"""Configuration classes for quantization"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuantConfig:
    """Configuration for model quantization"""

    # Model settings
    model_id: str  # HuggingFace model ID or local path
    model_name: Optional[str] = None  # Override model name (auto-detected if None)

    # Authentication
    hf_token: Optional[str] = None

    # Output settings
    output_dir: Optional[str] = None  # Local output directory
    upload_to_hf: bool = False
    hf_username: Optional[str] = None

    # GGUF-specific settings
    quantization_methods: List[str] = field(default_factory=lambda: ["q4_k_m"])
    use_imatrix: bool = False
    train_data_path: Optional[str] = None
    split_model: bool = False
    split_max_tensors: int = 256

    def __post_init__(self):
        """Validate configuration"""
        if self.upload_to_hf and not self.hf_username:
            raise ValueError("hf_username is required when upload_to_hf is True")

        if self.upload_to_hf and not self.hf_token:
            raise ValueError("hf_token is required when upload_to_hf is True")

        if not self.output_dir and not self.upload_to_hf:
            raise ValueError("Either output_dir or upload_to_hf must be specified")

        # Auto-detect model name from model_id if not provided
        if self.model_name is None:
            self.model_name = self.model_id.split('/')[-1]
