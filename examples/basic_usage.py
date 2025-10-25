"""
Basic usage examples for ggez_quant
"""

import os
from core import Quantizer

# Set your HuggingFace token (or use environment variable)
HF_TOKEN = os.environ.get("HF_TOKEN", "your_token_here")
HF_USERNAME = "your_username"


def example_1_simple_quantization():
    """Example 1: Simple quantization to local directory"""
    print("\n" + "="*60)
    print("Example 1: Simple Quantization")
    print("="*60)

    quantizer = Quantizer(
        model_id="meta-llama/Llama-2-7b-hf",
        hf_token=HF_TOKEN
    )

    result = quantizer.quantize_gguf(
        methods=["q4_k_m"],
        output_dir="./quantized_models"
    )

    print(f"Quantized files: {result['quantized_files']}")


def example_2_multiple_methods():
    """Example 2: Quantize with multiple methods"""
    print("\n" + "="*60)
    print("Example 2: Multiple Quantization Methods")
    print("="*60)

    quantizer = Quantizer(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hf_token=HF_TOKEN
    )

    result = quantizer.quantize_gguf(
        methods=["q4_k_m", "q5_k_m", "q8_0"],
        output_dir="./quantized_models"
    )

    print(f"Created {len(result['quantized_files'])} quantized files")


def example_3_upload_to_hf():
    """Example 3: Quantize and upload to HuggingFace"""
    print("\n" + "="*60)
    print("Example 3: Upload to HuggingFace")
    print("="*60)

    quantizer = Quantizer(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hf_token=HF_TOKEN
    )

    result = quantizer.quantize_gguf(
        methods=["q4_k_m"],
        upload_to_hf=True,
        hf_username=HF_USERNAME
    )

    print(f"Uploaded to: https://huggingface.co/{result['repo_id']}")


def example_4_local_and_upload():
    """Example 4: Save locally AND upload to HuggingFace"""
    print("\n" + "="*60)
    print("Example 4: Save Locally and Upload")
    print("="*60)

    quantizer = Quantizer(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hf_token=HF_TOKEN
    )

    result = quantizer.quantize_gguf(
        methods=["q4_k_m", "q5_k_m"],
        output_dir="./my_quantized_models",
        upload_to_hf=True,
        hf_username=HF_USERNAME
    )

    print(f"Local files: {result['quantized_files']}")
    print(f"HuggingFace: https://huggingface.co/{result['repo_id']}")


def example_5_local_model():
    """Example 5: Quantize a local model"""
    print("\n" + "="*60)
    print("Example 5: Quantize Local Model")
    print("="*60)

    # Assuming you have a model downloaded locally
    quantizer = Quantizer(
        model_id="./path/to/local/model"
    )

    result = quantizer.quantize_gguf(
        methods=["q4_k_m"],
        output_dir="./quantized_models"
    )

    print(f"Quantized files: {result['quantized_files']}")


if __name__ == "__main__":
    # Run examples (uncomment the ones you want to try)

    # example_1_simple_quantization()
    # example_2_multiple_methods()
    # example_3_upload_to_hf()
    # example_4_local_and_upload()
    # example_5_local_model()

    print("\nUncomment the examples you want to run in the __main__ section")
