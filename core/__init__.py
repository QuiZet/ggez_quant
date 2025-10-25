"""
ggez_quant - A lightweight library for quantizing LLMs
"""

from .quantizer import Quantizer
from .config import QuantConfig

__version__ = "0.1.0"
__all__ = ["Quantizer", "QuantConfig"]
