"""GGUF quantization implementation using llama.cpp"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from .utils import run_command, ensure_directory


class GGUFQuantizer:
    """Handles GGUF quantization using llama.cpp"""

    def __init__(self, llama_cpp_dir: str = "llama.cpp", auto_setup: bool = True):
        """
        Initialize GGUF Quantizer

        Args:
            llama_cpp_dir: Path to llama.cpp directory
            auto_setup: Automatically clone and build llama.cpp if missing (default: True)
                       Set to False to require manual setup via setup_llama_cpp.py
        """
        self.llama_cpp_dir = llama_cpp_dir
        self.llama_cpp_path = Path(llama_cpp_dir)
        self.auto_setup = auto_setup
        self._ensure_llama_cpp()

    def _ensure_llama_cpp(self) -> None:
        """Ensure llama.cpp is available"""
        if not self.llama_cpp_path.exists():
            if not self.auto_setup:
                raise RuntimeError(
                    f"llama.cpp not found at {self.llama_cpp_dir}\n"
                    "Please run the setup script first:\n"
                    "  python setup_llama_cpp.py\n"
                    "Or set auto_setup=True to automatically build llama.cpp"
                )

            print("="*60)
            print("llama.cpp not found - performing first-time setup")
            print("="*60)
            print("This will clone and build llama.cpp (only needed once)")
            print("To skip auto-setup in the future, run: python setup_llama_cpp.py")
            print("="*60)

            print("\n[1/4] Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp"],
                check=True,
            )

            print("\n[2/4] Building llama.cpp with CMake...")
            subprocess.run(
                ["cmake", "-B", "build", "-DLLAMA_CURL=OFF"],
                cwd=self.llama_cpp_dir,
                check=True,
            )

            print("\n[3/4] Compiling (this may take a few minutes)...")
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"],
                cwd=self.llama_cpp_dir,
                check=True,
            )

            print("\n[4/4] Installing llama.cpp requirements...")
            subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=self.llama_cpp_dir,
                check=True,
            )
            print("\n✓ llama.cpp setup complete!\n")

        # Verify required binaries exist
        required_bins = ["llama-quantize", "llama-imatrix", "llama-gguf-split"]
        bin_dir = self.llama_cpp_path / "build" / "bin"

        missing_bins = []
        for bin_name in required_bins:
            if not (bin_dir / bin_name).exists():
                missing_bins.append(bin_name)

        if missing_bins:
            raise RuntimeError(
                f"llama.cpp binaries not found: {', '.join(missing_bins)}\n"
                "Please rebuild llama.cpp by running:\n"
                "  python setup_llama_cpp.py"
            )

    def convert_to_fp16(
        self,
        model_path: str,
        output_path: str,
        force: bool = False,
    ) -> str:
        """
        Convert model to FP16 GGUF format

        Args:
            model_path: Path to the model directory
            output_path: Path for the FP16 GGUF file
            force: Overwrite if file exists

        Returns:
            Path to the converted model
        """
        if os.path.exists(output_path) and not force:
            print(f"FP16 model already exists at {output_path}, skipping conversion.")
            return output_path

        print(f"Converting model to FP16 format...")
        ensure_directory(os.path.dirname(output_path))

        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        subprocess.run(
            [
                "python",
                str(convert_script),
                model_path,
                "--outtype", "f16",
                "--outfile", output_path,
            ],
            check=True,
        )
        print(f"Conversion complete: {output_path}")
        return output_path

    def generate_imatrix(
        self,
        model_path: str,
        output_path: str,
        train_data_path: Optional[str] = None,
        timeout: int = 600,
    ) -> str:
        """
        Generate importance matrix for better quantization

        Args:
            model_path: Path to the FP16 GGUF model
            output_path: Path to save the imatrix
            train_data_path: Path to training data (uses default if None)
            timeout: Timeout in seconds

        Returns:
            Path to the generated imatrix file
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print("Generating importance matrix...")

        # Use default calibration dataset if none provided
        if not train_data_path or not os.path.isfile(train_data_path):
            print("Using default calibration dataset...")
            train_data_path = str(self.llama_cpp_path / "groups_merged.txt")

        imatrix_bin = self.llama_cpp_path / "build" / "bin" / "llama-imatrix"
        command = [
            str(imatrix_bin),
            "-m", model_path,
            "-f", train_data_path,
            "-ngl", "99",
            "--output-frequency", "10",
            "-o", output_path,
        ]

        try:
            run_command(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Imatrix generation timed out but may have produced partial results.")

        if os.path.exists(output_path):
            print(f"Importance matrix generated: {output_path}")
            return output_path
        else:
            raise RuntimeError("Failed to generate importance matrix")

    def quantize(
        self,
        fp16_model_path: str,
        output_path: str,
        method: str,
        imatrix_path: Optional[str] = None,
    ) -> str:
        """
        Quantize a model using specified method

        Args:
            fp16_model_path: Path to FP16 GGUF model
            output_path: Path for quantized model
            method: Quantization method (e.g., 'q4_k_m', 'q5_k_m')
            imatrix_path: Path to imatrix file for imatrix quantization

        Returns:
            Path to the quantized model
        """
        print(f"Quantizing model with {method}...")

        quantize_bin = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
        command = [str(quantize_bin)]

        if imatrix_path:
            command.extend(["--imatrix", imatrix_path])

        command.extend([fp16_model_path, output_path, method])

        subprocess.run(command, check=True)
        print(f"Quantization complete: {output_path}")
        return output_path

    def split_model(
        self,
        model_path: str,
        split_max_tensors: int = 256,
    ) -> List[str]:
        """
        Split a large model into multiple shards

        Args:
            model_path: Path to the model to split
            split_max_tensors: Maximum tensors per shard

        Returns:
            List of shard file paths
        """
        print(f"Splitting model {model_path}...")

        model_path_prefix = '.'.join(model_path.split('.')[:-1])
        split_bin = self.llama_cpp_path / "build" / "bin" / "llama-gguf-split"

        command = [
            str(split_bin),
            "--split",
            "--split-max-tensors", str(split_max_tensors),
            model_path,
            model_path_prefix,
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error splitting model: {result.stderr}")

        print("Model split successfully!")

        # Get list of sharded files
        model_file_prefix = os.path.basename(model_path_prefix)
        model_dir = os.path.dirname(model_path)
        sharded_files = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.startswith(model_file_prefix) and f.endswith(".gguf")
        ]

        if not sharded_files:
            raise RuntimeError("No sharded files found after splitting")

        print(f"Created {len(sharded_files)} shards")

        # Remove original model to save space
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed original model file {model_path}")

        return sharded_files

    def quantize_model(
        self,
        model_path: str,
        output_dir: str,
        model_name: str,
        methods: List[str],
        use_imatrix: bool = False,
        train_data_path: Optional[str] = None,
        split_model: bool = False,
        split_max_tensors: int = 256,
    ) -> List[str]:
        """
        Complete quantization workflow for a model

        Args:
            model_path: Path to the model directory
            output_dir: Directory to save quantized models
            model_name: Name for the output files
            methods: List of quantization methods
            use_imatrix: Whether to use importance matrix
            train_data_path: Path to training data for imatrix
            split_model: Whether to split large models
            split_max_tensors: Max tensors per shard when splitting

        Returns:
            List of paths to quantized model files
        """
        ensure_directory(output_dir)
        quantized_files = []

        # Convert to FP16
        fp16_path = os.path.join(output_dir, f"{model_name.lower()}.fp16.gguf")
        self.convert_to_fp16(model_path, fp16_path)

        # Generate imatrix if requested
        imatrix_path = None
        if use_imatrix:
            imatrix_path = os.path.join(output_dir, "imatrix.dat")
            if not os.path.exists(imatrix_path):
                self.generate_imatrix(fp16_path, imatrix_path, train_data_path)
            else:
                print(f"Imatrix already exists at {imatrix_path}")

        # Quantize for each method
        for method in methods:
            quant_suffix = f"{method.lower()}-imat" if use_imatrix else method.lower()
            quant_path = os.path.join(output_dir, f"{model_name.lower()}.{quant_suffix}.gguf")

            # Skip if already exists
            if os.path.exists(quant_path):
                print(f"Quantized model {quant_path} already exists, skipping...")
                quantized_files.append(quant_path)
                continue

            # Quantize
            self.quantize(fp16_path, quant_path, method, imatrix_path)

            # Split if requested
            if split_model:
                shards = self.split_model(quant_path, split_max_tensors)
                quantized_files.extend(shards)
            else:
                quantized_files.append(quant_path)

        return quantized_files
