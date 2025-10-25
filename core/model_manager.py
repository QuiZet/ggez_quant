"""Model download and upload utilities"""

import os
from pathlib import Path
from typing import Optional, List

from huggingface_hub import (
    create_repo,
    HfApi,
    ModelCard,
    snapshot_download,
)


class ModelManager:
    """Handles model downloads and uploads to HuggingFace"""

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token) if hf_token else HfApi()

    def download_model(
        self,
        model_id: str,
        local_dir: str,
        ignore_patterns: Optional[List[str]] = None,
    ) -> str:
        """
        Download a model from HuggingFace Hub

        Args:
            model_id: HuggingFace model ID
            local_dir: Local directory to save the model
            ignore_patterns: File patterns to ignore during download

        Returns:
            Path to the downloaded model
        """
        if ignore_patterns is None:
            ignore_patterns = ["*.msgpack", "*.h5", "*.ot", "*.onnx"]

        print(f"Downloading model {model_id}...")
        model_path = snapshot_download(
            repo_id=model_id,
            token=self.hf_token,
            ignore_patterns=ignore_patterns,
            local_dir=local_dir,
        )
        print(f"Model downloaded to: {model_path}")
        return model_path

    def upload_quantized_model(
        self,
        base_model_id: str,
        quantized_model_name: str,
        quantization_type: str,
        save_folder: str,
        username: str,
        allow_patterns: Optional[List[str]] = None,
        bpw: Optional[float] = None,
    ) -> str:
        """
        Upload a quantized model to HuggingFace Hub

        Args:
            base_model_id: Original model ID
            quantized_model_name: Name for the quantized model
            quantization_type: Type of quantization (e.g., 'gguf', 'gptq', 'awq')
            save_folder: Folder containing the quantized model
            username: HuggingFace username
            allow_patterns: File patterns to upload
            bpw: Bits per weight (for EXL2)

        Returns:
            Repository ID of the uploaded model
        """
        # Define repository ID
        if quantization_type == 'exl2' and bpw is not None:
            repo_id = f"{username}/{quantized_model_name}-{bpw:.1f}bpw-exl2"
        else:
            repo_id = f"{username}/{quantized_model_name}"

        # Try to load existing model card
        try:
            existing_card = ModelCard.load(repo_id, token=self.hf_token)
            print(f"Model card already exists for {repo_id}. Skipping model card creation.")
        except Exception:
            # Create new model card
            try:
                card = ModelCard.load(base_model_id)
                card.data.tags = [] if card.data.tags is None else card.data.tags
            except Exception:
                card = ModelCard("")
                card.data.tags = []

            card.data.tags.extend(["ggez_quant", quantization_type])
            card.save(f'{save_folder}/README.md')
            print(f"Created new model card for {repo_id}")

        # Create or update repository
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=self.hf_token,
        )

        # Upload the model
        print(f"Uploading quantized model to {repo_id}...")
        self.api.upload_folder(
            folder_path=save_folder,
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            token=self.hf_token,
        )

        print(f"Successfully uploaded quantized model to {repo_id}")
        return repo_id

    @staticmethod
    def is_local_path(model_id: str) -> bool:
        """Check if model_id is a local path or HuggingFace ID"""
        return os.path.exists(model_id) or os.path.isabs(model_id)
