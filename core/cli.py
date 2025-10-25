"""Command-line interface for ggez_quant"""

import argparse
import os
import sys
from typing import List

from .quantizer import Quantizer
from .config import QuantConfig


def parse_methods(methods_str: str) -> List[str]:
    """Parse comma-separated quantization methods"""
    return [m.strip() for m in methods_str.split(",")]


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ggez_quant - Easily quantize LLMs to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize a HuggingFace model to q4_k_m and save locally
  ggez_quant meta-llama/Llama-2-7b-hf --methods q4_k_m --output ./quantized

  # Quantize and upload to HuggingFace
  ggez_quant meta-llama/Llama-2-7b-hf --methods q4_k_m,q5_k_m \\
    --upload --username your_username --token $HF_TOKEN

  # Quantize with importance matrix
  ggez_quant local/model/path --methods Q4_K_M,Q5_K_M \\
    --imatrix --output ./quantized

  # Quantize and split large models
  ggez_quant username/large-model --methods q4_k_m \\
    --split --split-max-tensors 256 --output ./quantized
        """,
    )

    # Required arguments
    parser.add_argument(
        "model_id",
        help="HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf) or local path",
    )

    # Quantization settings
    parser.add_argument(
        "-m", "--methods",
        default="q4_k_m",
        help="Comma-separated quantization methods (default: q4_k_m). "
             "Common: q2_k, q3_k_m, q4_k_m, q5_k_m, q6_k, q8_0",
    )

    # Output settings
    parser.add_argument(
        "-o", "--output",
        help="Local output directory for quantized models",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload quantized model to HuggingFace",
    )
    parser.add_argument(
        "-u", "--username",
        help="HuggingFace username (required if --upload is used)",
    )

    # Authentication
    parser.add_argument(
        "-t", "--token",
        help="HuggingFace API token (can also use HF_TOKEN environment variable)",
    )

    # GGUF-specific options
    parser.add_argument(
        "--imatrix",
        action="store_true",
        help="Use importance matrix for better quantization",
    )
    parser.add_argument(
        "--train-data",
        help="Path to custom training data for imatrix generation",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split large models into multiple files",
    )
    parser.add_argument(
        "--split-max-tensors",
        type=int,
        default=256,
        help="Maximum tensors per file when splitting (default: 256)",
    )

    # Advanced options
    parser.add_argument(
        "--llama-cpp-dir",
        default="llama.cpp",
        help="Directory for llama.cpp installation (default: llama.cpp)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.upload and not args.username:
        parser.error("--username is required when --upload is used")

    if not args.output and not args.upload:
        parser.error("Either --output or --upload must be specified")

    # Get HF token from args or environment
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if args.upload and not hf_token:
        print("Warning: No HuggingFace token provided. Upload may fail.")
        print("Provide token via --token or HF_TOKEN environment variable")

    # Parse quantization methods
    methods = parse_methods(args.methods)

    try:
        # Create quantizer
        print(f"Initializing quantizer for model: {args.model_id}")
        quantizer = Quantizer(
            model_id=args.model_id,
            hf_token=hf_token,
            llama_cpp_dir=args.llama_cpp_dir,
        )

        # Quantize
        result = quantizer.quantize_gguf(
            methods=methods,
            output_dir=args.output,
            upload_to_hf=args.upload,
            hf_username=args.username,
            use_imatrix=args.imatrix,
            train_data_path=args.train_data,
            split_model=args.split,
            split_max_tensors=args.split_max_tensors,
        )

        # Print results
        print("\n" + "="*60)
        print("Quantization Results:")
        print("="*60)
        print(f"Methods: {', '.join(methods)}")
        print(f"Output directory: {result['output_dir']}")
        print(f"Files created: {len(result['quantized_files'])}")
        if 'repo_id' in result:
            print(f"HuggingFace repository: https://huggingface.co/{result['repo_id']}")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
