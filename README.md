# ggez_quant

A lightweight Python library for quantizing Large Language Models (LLMs) to GGUF format. Easily quantize models from HuggingFace or local files, and save them locally or upload directly to HuggingFace.

> **📝 Based on:** This is a Python library implementation of the [AutoQuant](https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing#scrollTo=Tj4rKt1UYA9q) Colab notebook by [Maxime Labonne](https://github.com/mlabonne). Check out his excellent [LLM Course](https://github.com/mlabonne/llm-course) for more resources on working with Large Language Models.

## Features

- **Simple API**: Quantize models with just a few lines of code
- **GGUF Support**: Full support for GGUF quantization using llama.cpp
- **Multiple Methods**: Support for various quantization methods (q2_k, q3_k_m, q4_k_m, q5_k_m, q6_k, q8_0, etc.)
- **Importance Matrix**: Optional imatrix generation for better quantization quality
- **Model Splitting**: Automatic splitting of large models into shards
- **HuggingFace Integration**: Download from and upload to HuggingFace Hub
- **CLI Tool**: Command-line interface for quick quantization tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ggez_quant.git
cd ggez_quant

# Install the package
pip install -e .

# Setup llama.cpp (one-time setup)
python setup_llama_cpp.py

# Or install from PyPI (when published)
pip install ggez_quant
```

### Prerequisites

- Python 3.8+
- Git
- CMake (for building llama.cpp)
  - Ubuntu/Debian: `sudo apt install cmake build-essential`
  - macOS: `brew install cmake`
  - Windows: Install from [cmake.org](https://cmake.org/download/)

### First-Time Setup

After installing the package, run the setup script to build llama.cpp:

```bash
python setup_llama_cpp.py
```

This will:
1. Clone the llama.cpp repository
2. Build the quantization tools
3. Install necessary dependencies

This only needs to be done once. The library will automatically use the built tools for all quantization tasks.

## Quick Start

### Python API

```python
from core import Quantizer

# Initialize quantizer with a HuggingFace model
quantizer = Quantizer(
    model_id="meta-llama/Llama-2-7b-hf",
    hf_token="your_hf_token"  # optional, for private models
)

# Quantize and save locally (only final GGUF files saved)
result = quantizer.quantize_gguf(
    methods=["q4_k_m", "q5_k_m"],
    output_dir="./quantized_models"
)

# Quantize and upload to HuggingFace (no local files saved)
result = quantizer.quantize_gguf(
    methods=["q4_k_m"],
    upload_to_hf=True,
    hf_username="your_username"
)

# Quantize with importance matrix for better quality
result = quantizer.quantize_gguf(
    methods=["Q4_K_M", "Q5_K_M"],
    output_dir="./quantized_models",
    use_imatrix=True
)
```

**Note:** Downloaded models and FP16 intermediate files are automatically stored in temporary directories and cleaned up after quantization. Only the final quantized GGUF files are saved to your output directory.

### CLI Usage

```bash
# Quantize a HuggingFace model
ggez_quant meta-llama/Llama-2-7b-hf --methods q4_k_m --output ./quantized

# Quantize and upload to HuggingFace
ggez_quant username/model-name \
  --methods q4_k_m,q5_k_m \
  --upload \
  --username your_username \
  --token $HF_TOKEN

# Quantize with importance matrix
ggez_quant local/model/path \
  --methods Q4_K_M,Q5_K_M \
  --imatrix \
  --output ./quantized

# Quantize and split large models
ggez_quant username/large-model \
  --methods q4_k_m \
  --split \
  --split-max-tensors 256 \
  --output ./quantized
```

## Advanced Usage

### Using Configuration Objects

```python
from core import Quantizer, QuantConfig

# Create a configuration
config = QuantConfig(
    model_id="meta-llama/Llama-2-7b-hf",
    hf_token="your_hf_token",
    quantization_methods=["q4_k_m", "q5_k_m"],
    output_dir="./quantized_models",
    upload_to_hf=True,
    hf_username="your_username",
    use_imatrix=True,
    split_model=False
)

# Quantize using the config
quantizer = Quantizer(
    model_id=config.model_id,
    hf_token=config.hf_token
)
result = quantizer.quantize(config)
```

### Quantizing Local Models

```python
# Quantize a model stored locally
quantizer = Quantizer(model_id="./path/to/local/model")

result = quantizer.quantize_gguf(
    methods=["q4_k_m"],
    output_dir="./quantized_models"
)
```

### Custom Training Data for Imatrix

```python
result = quantizer.quantize_gguf(
    methods=["Q4_K_M"],
    use_imatrix=True,
    train_data_path="./custom_calibration_data.txt",
    output_dir="./quantized_models"
)
```

## Quantization Methods

Common quantization methods (in order of quality/size):

- `q2_k`: 2-bit quantization (smallest, lowest quality)
- `q3_k_m`: 3-bit quantization (medium)
- `q4_k_m`: 4-bit quantization (recommended, good balance)
- `q5_k_m`: 5-bit quantization (high quality)
- `q6_k`: 6-bit quantization (very high quality)
- `q8_0`: 8-bit quantization (highest quality, larger size)

Imatrix methods (require `use_imatrix=True`):
- `IQ3_M`, `IQ3_XXS`, `Q4_K_M`, `Q4_K_S`, `IQ4_NL`, `IQ4_XS`, `Q5_K_M`, `Q5_K_S`

## Requirements

- Python 3.8+
- CMake (for building llama.cpp)
- Git
- HuggingFace account (for uploads)

The library will automatically:
- Clone and build llama.cpp if not present
- Download models from HuggingFace
- Handle authentication

## Project Structure

```
ggez_quant/
├── ggez_quant/
│   ├── __init__.py       # Main API exports
│   ├── config.py         # Configuration classes
│   ├── quantizer.py      # Main Quantizer class
│   ├── gguf.py          # GGUF quantization implementation
│   ├── model_manager.py # HuggingFace download/upload
│   ├── utils.py         # Helper functions
│   └── cli.py           # CLI interface
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── setup.py
├── requirements.txt
└── README.md
```

## Examples

See the `examples/` directory for more detailed examples:

- `basic_usage.py`: Simple quantization examples
- `advanced_usage.py`: Advanced features and configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Credits
The one and only [Maxime Labonne](https://github.com/mlabonne)

This library is built on top of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [HuggingFace Hub](https://github.com/huggingface/huggingface_hub)
- Inspired by [AutoQuant](https://github.com/mlabonne/llm-course) by Maxime Labonne

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
