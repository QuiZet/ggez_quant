"""Utility functions"""

import os
import subprocess
import signal
from typing import List


def run_command(
    command: List[str],
    timeout: int = 600,
    shell: bool = False,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a shell command with timeout handling

    Args:
        command: Command and arguments as a list
        timeout: Timeout in seconds
        shell: Whether to run in shell mode
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess instance
    """
    if capture_output:
        return subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    process = subprocess.Popen(command, shell=shell)
    try:
        process.wait(timeout=timeout)
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
        )
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s. Sending SIGINT for graceful termination...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Process still didn't terminate. Forcefully killing...")
            process.kill()
        raise


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
