from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def run_diagnostics() -> dict:
    checks: List[CheckResult] = []

    checks.append(_check_torch())
    checks.append(_check_cuda_toolkit())
    checks.append(_check_pynvvideocodec())
    checks.append(_check_nvidia_driver())

    failing = [check for check in checks if check.status != "ok"]
    advice: List[str] = []

    if any(check.name == "PyNvVideoCodec" and check.status != "ok" for check in checks):
        advice.append(
            "Install PyNvVideoCodec: pip install torchwindow[nvenc] or download the wheel"
            " from NVIDIA's developer site."
        )
    if any(check.name == "Torch CUDA" and check.status != "ok" for check in checks):
        advice.append(
            "Install a CUDA-enabled build of PyTorch (pip install torch --index-url"
            " https://download.pytorch.org/whl/cu121)."
        )
    if any(check.name == "NVENC Driver" and check.status != "ok" for check in checks):
        advice.append(
            "Ensure the NVIDIA driver is installed and nvidia-smi is in PATH."
        )

    report = {
        "status": "ok" if not failing else "error",
        "checks": [check.__dict__ for check in checks],
    }
    if advice:
        report["advice"] = advice
    return report


def _check_torch() -> CheckResult:
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckResult(
                name="Torch CUDA",
                status="fail",
                detail="torch.cuda.is_available() returned False",
            )
        device = torch.cuda.get_device_name(0)
        return CheckResult(
            name="Torch CUDA",
            status="ok",
            detail=f"CUDA device detected: {device}",
        )
    except ModuleNotFoundError:
        return CheckResult(
            name="Torch CUDA",
            status="fail",
            detail="torch is not installed",
        )
    except Exception as exc:
        return CheckResult(
            name="Torch CUDA",
            status="fail",
            detail=f"torch check failed: {exc}",
        )


def _check_cuda_toolkit() -> CheckResult:
    cudart = shutil.which("nvcc")
    if not cudart:
        return CheckResult(
            name="CUDA Toolkit",
            status="warn",
            detail="nvcc not found in PATH (not required but useful)",
        )
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return CheckResult(
            name="CUDA Toolkit",
            status="ok",
            detail=result.stdout.splitlines()[-1],
        )
    except Exception as exc:
        return CheckResult(
            name="CUDA Toolkit",
            status="warn",
            detail=f"Failed to query nvcc: {exc}",
        )


def _check_pynvvideocodec() -> CheckResult:
    try:
        import PyNvVideoCodec  # type: ignore  # noqa: F401

        return CheckResult(
            name="PyNvVideoCodec",
            status="ok",
            detail="PyNvVideoCodec module imported successfully",
        )
    except ModuleNotFoundError:
        return CheckResult(
            name="PyNvVideoCodec",
            status="fail",
            detail="PyNvVideoCodec is not installed",
        )
    except Exception as exc:
        return CheckResult(
            name="PyNvVideoCodec",
            status="fail",
            detail=f"Import failed: {exc}",
        )


def _check_nvidia_driver() -> CheckResult:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return CheckResult(
            name="NVENC Driver",
            status="warn",
            detail="nvidia-smi not found in PATH",
        )
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return CheckResult(
            name="NVENC Driver",
            status="ok",
            detail=result.stdout.strip(),
        )
    except Exception as exc:
        return CheckResult(
            name="NVENC Driver",
            status="warn",
            detail=f"nvidia-smi query failed: {exc}",
        )
