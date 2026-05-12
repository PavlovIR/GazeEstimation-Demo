from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "env.yaml"


@dataclass(slots=True)
class BackendSelection:
    name: str
    torch_backend: str | None
    reason: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the project environment and install a platform-appropriate "
            "PyTorch backend."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to env.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the detected backend and commands without executing them.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip `uv sync` and only reinstall torch/torchvision into .venv.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command))
    if dry_run:
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _version_key(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts)


def _detect_nvidia() -> tuple[str, str] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    summary = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    full = subprocess.run(
        ["nvidia-smi"],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"CUDA Version:\s*([0-9.]+)", full.stdout)
    if match is None:
        return None

    gpu_names = [line.strip() for line in summary.stdout.splitlines() if line.strip()]
    gpu_name = ", ".join(gpu_names) if gpu_names else "NVIDIA GPU"
    return gpu_name, match.group(1)


def _choose_nvidia_backend(config: dict[str, Any], cuda_version: str) -> str | None:
    backends = config["pytorch"]["nvidia"]["preferred_backends"]
    current = _version_key(cuda_version)
    for item in backends:
        if current >= _version_key(str(item["min_cuda"])):
            return str(item["torch_backend"])
    return None


def _detect_amd_linux() -> tuple[str, list[str]] | None:
    if platform.system() != "Linux":
        return None

    notes: list[str] = []
    for command in (["rocm-smi"], ["amd-smi", "list"]):
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            details = result.stdout.strip().splitlines()
            headline = details[0].strip() if details else command[0]
            notes.append(f"Detected AMD tooling via `{command[0]}`.")
            return headline, notes

    if shutil.which("lspci") is None:
        return None

    result = subprocess.run(["lspci"], text=True, capture_output=True)
    if result.returncode != 0:
        return None

    amd_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if re.search(r"(AMD|Advanced Micro Devices|ATI|Radeon)", line, re.IGNORECASE)
    ]
    if not amd_lines:
        return None

    notes.append("Detected AMD graphics hardware from `lspci`.")
    notes.append("ROCm wheels still require a working ROCm runtime on the host.")
    return amd_lines[0], notes


def select_backend(config: dict[str, Any]) -> BackendSelection:
    system = platform.system()
    if system == "Darwin":
        return BackendSelection(
            name="apple",
            torch_backend=config["pytorch"]["apple"]["torch_backend"],
            reason="Detected macOS; PyTorch MPS support ships in the standard macOS wheels.",
            notes=[],
        )

    nvidia = _detect_nvidia()
    if nvidia is not None:
        gpu_name, cuda_version = nvidia
        backend = _choose_nvidia_backend(config, cuda_version)
        if backend is None:
            return BackendSelection(
                name="cpu",
                torch_backend=config["pytorch"]["cpu"]["torch_backend"],
                reason=(
                    f"Detected NVIDIA GPU `{gpu_name}` but CUDA {cuda_version} is older "
                    "than the configured PyTorch CUDA backends."
                ),
                notes=["Falling back to CPU wheels."],
            )
        return BackendSelection(
            name="nvidia",
            torch_backend=backend,
            reason=(
                f"Detected NVIDIA GPU `{gpu_name}` with driver CUDA {cuda_version}; "
                f"selected PyTorch backend `{backend}`."
            ),
            notes=[],
        )

    amd = _detect_amd_linux()
    if amd is not None:
        gpu_name, notes = amd
        return BackendSelection(
            name="amd",
            torch_backend=config["pytorch"]["amd"]["linux_backend"],
            reason=(
                f"Detected AMD GPU `{gpu_name}` on Linux; selected ROCm backend "
                f"`{config['pytorch']['amd']['linux_backend']}`."
            ),
            notes=notes,
        )

    return BackendSelection(
        name="cpu",
        torch_backend=config["pytorch"]["cpu"]["torch_backend"],
        reason="No supported GPU runtime detected; selected CPU wheels.",
        notes=[],
    )


def venv_python_path(root: Path) -> Path:
    if platform.system() == "Windows":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def sync_base_environment(config: dict[str, Any], dry_run: bool) -> None:
    command = ["uv", "sync"]
    python_version = config.get("python")
    if python_version:
        command.extend(["--python", str(python_version)])
    for package_name in config["uv_sync"]["skip_packages"]:
        command.extend(["--no-install-package", str(package_name)])
    run_command(command, cwd=ROOT, dry_run=dry_run)


def install_torch_backend(
    config: dict[str, Any],
    selection: BackendSelection,
    *,
    dry_run: bool,
) -> None:
    packages = config["pytorch"]["packages"]
    python_path = venv_python_path(ROOT)
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python_path),
        "--reinstall",
        f"torch=={packages['torch']}",
        f"torchvision=={packages['torchvision']}",
    ]
    if selection.torch_backend is not None:
        command.extend(["--torch-backend", selection.torch_backend])
    run_command(command, cwd=ROOT, dry_run=dry_run)


def verify_environment(dry_run: bool) -> None:
    python_path = venv_python_path(ROOT)
    command = [
        str(python_path),
        "-c",
        (
            "import platform, torch, torchvision; "
            "mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(); "
            "print('platform=', platform.system()); "
            "print('torch=', torch.__version__); "
            "print('torchvision=', torchvision.__version__); "
            "print('cuda_available=', torch.cuda.is_available()); "
            "print('mps_available=', mps); "
            "print('hip=', getattr(torch.version, 'hip', None))"
        ),
    ]
    run_command(command, cwd=ROOT, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selection = select_backend(config)

    print(selection.reason)
    for note in selection.notes:
        print(note)

    if not args.skip_sync:
        sync_base_environment(config, args.dry_run)
    install_torch_backend(config, selection, dry_run=args.dry_run)
    verify_environment(args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
