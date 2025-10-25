"""Main quantizer class"""

import os
import shutil
import tempfile
from typing import Optional, List

from .config import QuantConfig
from .model_manager import ModelManager
from .gguf import GGUFQuantizer
from .utils import ensure_directory


class Quantizer:
    """Main class for quantizing language models"""

    def __init__(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        llama_cpp_dir: str = "llama.cpp",
    ):
        """
        Initialize the Quantizer

        Args:
            model_id: HuggingFace model ID or local path
            hf_token: HuggingFace API token
            llama_cpp_dir: Directory for llama.cpp installation
        """
        self.model_id = model_id
        self.hf_token = hf_token
        self.model_manager = ModelManager(hf_token)
        self.gguf_quantizer = GGUFQuantizer(llama_cpp_dir)
        self.model_name = model_id.split('/')[-1]

    def quantize_gguf(
        self,
        methods: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        upload_to_hf: bool = False,
        hf_username: Optional[str] = None,
        use_imatrix: bool = False,
        train_data_path: Optional[str] = None,
        split_model: bool = False,
        split_max_tensors: int = 256,
        keep_intermediate_files: bool = False,
    ) -> dict:
        """
        Quantize a model to GGUF format

        Args:
            methods: List of quantization methods (default: ["q4_k_m"])
            output_dir: Local directory to save quantized models
            upload_to_hf: Whether to upload to HuggingFace
            hf_username: HuggingFace username for uploads
            use_imatrix: Use importance matrix for better quantization
            train_data_path: Path to training data for imatrix
            split_model: Split large models into shards
            split_max_tensors: Max tensors per shard
            keep_intermediate_files: Keep downloaded models and FP16 files (default: False)

        Returns:
            Dictionary with paths and repository info
        """
        if methods is None:
            methods = ["q4_k_m"]

        if not output_dir and not upload_to_hf:
            raise ValueError("Either output_dir or upload_to_hf must be specified")

        if upload_to_hf and not hf_username:
            raise ValueError("hf_username is required when upload_to_hf is True")

        # Use temporary directories for intermediate files
        with tempfile.TemporaryDirectory(prefix="ggez_quant_") as temp_base_dir:
            print(f"Using temporary directory for intermediate files: {temp_base_dir}")

            # Download model if it's from HuggingFace
            if not ModelManager.is_local_path(self.model_id):
                print(f"Downloading model from HuggingFace: {self.model_id}")
                model_download_dir = os.path.join(temp_base_dir, "model_download")
                model_path = self.model_manager.download_model(
                    self.model_id,
                    local_dir=model_download_dir,
                )
            else:
                print(f"Using local model: {self.model_id}")
                model_path = self.model_id

            # Create temporary directory for intermediate quantization files
            quant_temp_dir = os.path.join(temp_base_dir, "quantization")
            ensure_directory(quant_temp_dir)

            # Quantize the model
            print(f"Starting GGUF quantization with methods: {methods}")
            quantized_files = self.gguf_quantizer.quantize_model(
                model_path=model_path,
                output_dir=quant_temp_dir,
                model_name=self.model_name,
                methods=methods,
                use_imatrix=use_imatrix,
                train_data_path=train_data_path,
                split_model=split_model,
                split_max_tensors=split_max_tensors,
            )

            # Copy only final quantized files to output directory (if specified)
            final_output_dir = output_dir
            if output_dir:
                ensure_directory(output_dir)
                print(f"\nCopying final quantized files to {output_dir}...")

                final_files = []
                for file_path in quantized_files:
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(output_dir, filename)
                    shutil.copy2(file_path, dest_path)
                    final_files.append(dest_path)
                    print(f"  ✓ {filename}")

                quantized_files = final_files
            else:
                # Only uploading, use temp dir
                final_output_dir = quant_temp_dir

            # Copy imatrix file if it exists
            if use_imatrix and output_dir:
                imatrix_src = os.path.join(quant_temp_dir, "imatrix.dat")
                if os.path.exists(imatrix_src):
                    imatrix_dest = os.path.join(output_dir, "imatrix.dat")
                    shutil.copy2(imatrix_src, imatrix_dest)
                    print(f"  ✓ imatrix.dat")

            result = {
                "quantized_files": quantized_files,
                "output_dir": output_dir or final_output_dir,
                "methods": methods,
            }

            # Upload to HuggingFace if requested
            if upload_to_hf:
                print(f"\nUploading to HuggingFace...")

                # Determine which files to upload
                if use_imatrix:
                    allow_patterns = ["*.gguf", "*.md", "imatrix.dat"]
                else:
                    allow_patterns = ["*k.gguf", "*m.gguf", "*0.gguf", "*.md"]

                # Upload from the directory containing the files
                upload_dir = output_dir if output_dir else quant_temp_dir

                repo_id = self.model_manager.upload_quantized_model(
                    base_model_id=self.model_id,
                    quantized_model_name=f"{self.model_name}-GGUF",
                    quantization_type="gguf",
                    save_folder=upload_dir,
                    username=hf_username,
                    allow_patterns=allow_patterns,
                )
                result["repo_id"] = repo_id

            print("\n✅ Quantization completed successfully!")
            print(f"📦 Intermediate files automatically cleaned up")

            return result

    def quantize(
        self,
        config: QuantConfig,
    ) -> dict:
        """
        Quantize a model using a configuration object

        Args:
            config: QuantConfig instance with all settings

        Returns:
            Dictionary with paths and repository info
        """
        return self.quantize_gguf(
            methods=config.quantization_methods,
            output_dir=config.output_dir,
            upload_to_hf=config.upload_to_hf,
            hf_username=config.hf_username,
            use_imatrix=config.use_imatrix,
            train_data_path=config.train_data_path,
            split_model=config.split_model,
            split_max_tensors=config.split_max_tensors,
        )
