"""
Advanced usage examples for ggez_quant
"""

import os
from core import Quantizer, QuantConfig

HF_TOKEN = os.environ.get("HF_TOKEN", "your_token_here")
HF_USERNAME = "your_username"


def example_1_with_imatrix():
    """Example 1: Quantize with importance matrix for better quality"""
    print("\n" + "="*60)
    print("Example 1: Quantization with Importance Matrix")
    print("="*60)

    quantizer = Quantizer(
        model_id="meta-llama/Llama-2-7b-hf",
        hf_token=HF_TOKEN
    )

    # Imatrix methods provide better quantization quality
    result = quantizer.quantize_gguf(
        methods=["Q4_K_M", "Q5_K_M"],
        use_imatrix=True,
        output_dir="./quantized_models"
    )

    print(f"Quantized with imatrix: {result['quantized_files']}")


def example_2_split_large_model():
    """Example 2: Split a large model into multiple files"""
    print("\n" + "="*60)
    print("Example 2: Split Large Model")
    print("="*60)

    quantizer = Quantizer(
        model_id="meta-llama/Llama-2-70b-hf",
        hf_token=HF_TOKEN
    )

    result = quantizer.quantize_gguf(
        methods=["q4_k_m"],
        split_model=True,
        split_max_tensors=256,
        output_dir="./quantized_models"
    )

    print(f"Model split into {len(result['quantized_files'])} shards")


def example_3_custom_training_data():
    """Example 3: Use custom training data for imatrix"""
    print("\n" + "="*60)
    print("Example 3: Custom Training Data for Imatrix")
    print("="*60)

    quantizer = Quantizer(
        model_id="meta-llama/Llama-2-7b-hf",
        hf_token=HF_TOKEN
    )

    result = quantizer.quantize_gguf(
        methods=["Q4_K_M"],
        use_imatrix=True,
        train_data_path="./my_calibration_data.txt",
        output_dir="./quantized_models"
    )

    print(f"Quantized with custom imatrix data: {result['quantized_files']}")


def example_4_using_config_object():
    """Example 4: Use QuantConfig for complex configurations"""
    print("\n" + "="*60)
    print("Example 4: Using QuantConfig")
    print("="*60)

    # Create a reusable configuration
    config = QuantConfig(
        model_id="meta-llama/Llama-2-7b-hf",
        hf_token=HF_TOKEN,
        quantization_methods=["q4_k_m", "q5_k_m", "q8_0"],
        output_dir="./quantized_models",
        upload_to_hf=True,
        hf_username=HF_USERNAME,
        use_imatrix=True,
        split_model=False,
    )

    quantizer = Quantizer(
        model_id=config.model_id,
        hf_token=config.hf_token
    )

    result = quantizer.quantize(config)

    print(f"Quantization complete!")
    print(f"Local files: {result['quantized_files']}")
    print(f"HuggingFace: {result.get('repo_id')}")


def example_5_all_features():
    """Example 5: Using all features together"""
    print("\n" + "="*60)
    print("Example 5: All Features Combined")
    print("="*60)

    quantizer = Quantizer(
        model_id="meta-llama/Llama-2-13b-hf",
        hf_token=HF_TOKEN,
        llama_cpp_dir="./custom_llama_cpp"  # Custom llama.cpp directory
    )

    result = quantizer.quantize_gguf(
        methods=["Q4_K_M", "Q5_K_M"],
        output_dir="./quantized_models",
        upload_to_hf=True,
        hf_username=HF_USERNAME,
        use_imatrix=True,
        train_data_path="./calibration.txt",
        split_model=True,
        split_max_tensors=128,
    )

    print(f"Complete quantization:")
    print(f"  Methods: {result['methods']}")
    print(f"  Files: {len(result['quantized_files'])}")
    print(f"  Local: {result['output_dir']}")
    print(f"  HuggingFace: {result.get('repo_id')}")


def example_6_batch_quantization():
    """Example 6: Batch quantize multiple models"""
    print("\n" + "="*60)
    print("Example 6: Batch Quantization")
    print("="*60)

    models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        # Add more models as needed
    ]

    for model_id in models:
        print(f"\nQuantizing {model_id}...")

        quantizer = Quantizer(
            model_id=model_id,
            hf_token=HF_TOKEN
        )

        result = quantizer.quantize_gguf(
            methods=["q4_k_m"],
            output_dir=f"./quantized/{model_id.split('/')[-1]}",
            upload_to_hf=True,
            hf_username=HF_USERNAME
        )

        print(f"  ✓ Done: {result['repo_id']}")


if __name__ == "__main__":
    # Run examples (uncomment the ones you want to try)

    # example_1_with_imatrix()
    # example_2_split_large_model()
    # example_3_custom_training_data()
    # example_4_using_config_object()
    # example_5_all_features()
    # example_6_batch_quantization()

    print("\nUncomment the examples you want to run in the __main__ section")
