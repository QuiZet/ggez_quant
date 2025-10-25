#!/usr/bin/env python3
"""
One-time setup script for llama.cpp

Run this script once after installing ggez_quant:
    python setup_llama_cpp.py
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_llama_cpp(install_dir: str = "llama.cpp"):
    """Clone and build llama.cpp"""
    install_path = Path(install_dir)

    if install_path.exists():
        print(f"llama.cpp already exists at {install_dir}")
        response = input("Rebuild? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping setup.")
            return

    print("="*60)
    print("Setting up llama.cpp for GGUF quantization")
    print("="*60)

    # Clone if needed
    if not install_path.exists():
        print("\n[1/4] Cloning llama.cpp repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", install_dir],
                check=True,
            )
            print("✓ Clone complete")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error cloning llama.cpp: {e}")
            return False

    # Build with CMake
    print("\n[2/4] Building llama.cpp with CMake...")
    try:
        subprocess.run(
            ["cmake", "-B", "build", "-DLLAMA_CURL=OFF"],
            cwd=install_dir,
            check=True,
        )
        print("✓ CMake configuration complete")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error configuring with CMake: {e}")
        print("\nMake sure CMake is installed:")
        print("  Ubuntu/Debian: sudo apt install cmake")
        print("  macOS: brew install cmake")
        return False

    print("\n[3/4] Compiling (this may take a few minutes)...")
    try:
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release"],
            cwd=install_dir,
            check=True,
        )
        print("✓ Compilation complete")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error compiling: {e}")
        return False

    # Install Python requirements
    print("\n[4/4] Installing Python requirements...")
    try:
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            cwd=install_dir,
            check=True,
        )
        print("✓ Python requirements installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

    print("\n" + "="*60)
    print("✓ llama.cpp setup complete!")
    print("="*60)
    print(f"\nInstalled to: {install_path.absolute()}")
    print("\nYou can now use ggez_quant for quantization.")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup llama.cpp for ggez_quant")
    parser.add_argument(
        "--dir",
        default="llama.cpp",
        help="Directory to install llama.cpp (default: llama.cpp)"
    )

    args = parser.parse_args()

    success = setup_llama_cpp(args.dir)
    sys.exit(0 if success else 1)
