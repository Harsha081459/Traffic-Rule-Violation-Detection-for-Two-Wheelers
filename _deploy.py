"""
_deploy.py - Deploy and run training on the remote GPU server.

Steps performed automatically
-------------------------------
1. SSH into the server and check environment.
2. Install required Python packages into the server's conda env.
3. SCP the project files (dataset_builder.py, train.py, traffic_violation/).
4. Create a screen session for each model and launch the pipeline.
5. Print live status / log tails so you can monitor from Windows.

Usage
-----
    python _deploy.py                         # full pipeline
    python _deploy.py --target helmet         # one model only
    python _deploy.py --status                # check running screens
    python _deploy.py --logs helmet           # tail logs for a model
    python _deploy.py --rf-key YOUR_KEY       # pass Roboflow API key
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import paramiko
from scp import SCPClient

# ── Server credentials ────────────────────────────────────────────────
HOST = "172.16.121.74"
PORT = 22
USER = "sem6"
PASS = "user@123"

# ── Remote paths ──────────────────────────────────────────────────────
REMOTE_BASE   = "/home/sem6/cv_project"
REMOTE_DATA   = "/home/sem6/data"
REMOTE_MODELS = "/home/sem6/cv_project/models"

# ── Files / directories to upload ────────────────────────────────────
LOCAL_BASE = Path(__file__).parent
UPLOAD_FILES = [
    "dataset_builder.py",
    "train.py",
    "run_export.py",
    "run_test.py",
    "solution.py",
]
UPLOAD_DIRS = [
    "traffic_violation",
]

# ── pip packages the server needs (torch + ultralytics already there) ─
PIP_PACKAGES = [
    "ultralytics",
    "albumentations>=1.3",
    "roboflow",
    "imagehash",
    "Pillow",
    "pyyaml",
    "tqdm",
    "onnxruntime",
    "onnx",
    "onnxsim",
    "scp",
    "paramiko",
    "fiftyone",       # for Open Images support
]


# ─────────────────────────────────────────────────────────────────────
# SSH / SCP helpers
# ─────────────────────────────────────────────────────────────────────

def _connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, PORT, USER, PASS, timeout=20)
    return client


def _run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[str, str, int]:
    """Run *cmd* on the remote and return (stdout, stderr, exit_code)."""
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    out  = stdout.read().decode(errors="replace").strip()
    err  = stderr.read().decode(errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    return out, err, code


def _run_print(client: paramiko.SSHClient, cmd: str, label: str = "", timeout: int = 300) -> int:
    """Run a command and stream output to stdout."""
    label = label or cmd[:60]
    print(f"\n>>> {label}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        err = stderr.read().decode(errors="replace").strip()
        if err:
            print(f"[stderr] {err}")
    return code


def _scp_upload(client: paramiko.SSHClient) -> None:
    """Upload project files and directories to the remote server."""
    print("\n=== Uploading project files ===")
    with SCPClient(client.get_transport(), progress=_scp_progress) as scp:
        for fname in UPLOAD_FILES:
            local = LOCAL_BASE / fname
            if local.exists():
                scp.put(str(local), remote_path=f"{REMOTE_BASE}/{fname}")
                print(f"  [OK] {fname}")
            else:
                print(f"  [-] {fname} not found locally, skipping")

        for dname in UPLOAD_DIRS:
            local = LOCAL_BASE / dname
            if local.exists():
                scp.put(str(local), remote_path=REMOTE_BASE, recursive=True)
                print(f"  [OK] {dname}/")
            else:
                print(f"  [-] {dname}/ not found locally, skipping")

    # Upload model weights if present locally
    models_local = LOCAL_BASE / "models"
    if models_local.exists():
        print("\n  Uploading model weights (.pt files)...")
        for pt in models_local.glob("*.pt"):
            with SCPClient(client.get_transport(), progress=_scp_progress) as scp:
                scp.put(str(pt), remote_path=f"{REMOTE_MODELS}/{pt.name}")
            print(f"    [OK] {pt.name}")


def _scp_progress(filename, size, sent):
    pct = int(100 * sent / max(size, 1))
    if pct % 25 == 0:
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(f"\r    [{bar}] {pct}% {filename.decode()}", end="", flush=True)
    if sent >= size:
        print()


# ─────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────

def setup_environment(client: paramiko.SSHClient) -> None:
    print("\n=== Setting up remote environment ===")

    # Create project directory structure
    dirs = [REMOTE_BASE, REMOTE_DATA, REMOTE_MODELS]
    _run(client, "mkdir -p " + " ".join(f'"{d}"' for d in dirs))
    print("  [OK] Remote directories created")

    # Install pip packages
    print("\n  Installing packages (may take ~2 min)…")
    pkgs = " ".join(f'"{p}"' for p in PIP_PACKAGES)
    cmd  = f"python3 -m pip install --quiet --upgrade {pkgs} 2>&1 | tail -5"
    _run_print(client, cmd, label="pip install", timeout=600)

    # Verify GPU + torch
    out, _, _ = _run(client,
        "python3 -c \"import torch; "
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); "
        "print('CUDA:', torch.version.cuda)\"",
    )
    print(f"\n  {out}")


# ─────────────────────────────────────────────────────────────────────
# Training launcher
# ─────────────────────────────────────────────────────────────────────

TRAIN_CONFIGS = {
    "helmet": {
        "epochs": 200, "batch": 32, "imgsz": 640, "patience": 40,
        "close_mosaic": 20, "freeze": 0,
    },
    "full": {
        "epochs": 150, "batch": 16, "imgsz": 640, "patience": 30,
        "close_mosaic": 15, "freeze": 0,
    },
    "plate": {
        "epochs": 150, "batch": 32, "imgsz": 480, "patience": 30,
        "close_mosaic": 15, "freeze": 0,
    },
}


def launch_training(
    client: paramiko.SSHClient,
    target: str,
    rf_api_key: str,
    tune: bool = False,
) -> None:
    """Build dataset and start training inside a detached screen session."""
    cfg        = TRAIN_CONFIGS[target]
    screen_name = f"train_{target}"
    log_file    = f"{REMOTE_BASE}/logs/{target}.log"
    data_yaml   = f"{REMOTE_DATA}/{target}/final/data.yaml"

    _run(client, f"mkdir -p {REMOTE_BASE}/logs")

    # The full pipeline: build data → train → export
    rf_flag    = f"--rf-api-key {rf_api_key}" if rf_api_key else "--no-roboflow"
    tune_flag  = "--tune --tune-iterations 30" if tune else ""
    train_cmd  = (
        f"cd {REMOTE_BASE} && "
        f"python3 dataset_builder.py "
        f"  --target {target} "
        f"  --out-dir {REMOTE_DATA} "
        f"  --target-per-class 6000 "
        f"  {rf_flag} && "
        f"python3 train.py "
        f"  --target {target} "
        f"  --data {data_yaml} "
        f"  --epochs {cfg['epochs']} "
        f"  --batch {cfg['batch']} "
        f"  --imgsz {cfg['imgsz']} "
        f"  --patience {cfg['patience']} "
        f"  --close-mosaic {cfg['close_mosaic']} "
        f"  {tune_flag} "
        f"  --device 0 "
        f"  --workers 8 "
        f"  2>&1 | tee {log_file}"
    )

    # Try screen first; fall back to nohup if screen is not installed.
    _run(client, f"screen -S {screen_name} -X quit 2>/dev/null; sleep 1")
    launch_screen = f"screen -dmS {screen_name} bash -c '{train_cmd}'"
    out_s, err_s, code_s = _run(client, launch_screen)

    if code_s == 0 and "command not found" not in err_s.lower():
        print(f"  [OK] screen '{screen_name}' launched   log -> {log_file}")
    else:
        # screen not available: use nohup + setsid so the job survives logout
        pid_file = f"{REMOTE_BASE}/logs/{target}.pid"
        launch_nohup = (
            f"nohup bash -c {chr(39)}{train_cmd}{chr(39)} "
            f"> {log_file} 2>&1 & echo $! > {pid_file}"
        )
        out_n, err_n, code_n = _run(client, launch_nohup)
        if code_n == 0:
            pid_out, _, _ = _run(client, f"cat {pid_file} 2>/dev/null")
            print(f"  [OK] nohup job '{target}' launched (PID {pid_out.strip()})   log -> {log_file}")
        else:
            print(f"  [FAIL] Could not launch '{target}': {err_n}")


def status(client: paramiko.SSHClient) -> None:
    """Print running jobs (screen or nohup) and the last 5 log lines per target."""
    print("\n=== Running screen sessions ===")
    out, _, _ = _run(client, "screen -ls 2>/dev/null")
    print(out or "(none - screen not installed, using nohup)")

    print("\n=== nohup PID files ===")
    for target in ("helmet", "full", "plate"):
        pid_file = f"{REMOTE_BASE}/logs/{target}.pid"
        pid_out, _, code = _run(client, f"cat {pid_file} 2>/dev/null")
        if code == 0 and pid_out.strip():
            alive, _, _ = _run(client, f"kill -0 {pid_out.strip()} 2>/dev/null && echo running || echo stopped")
            print(f"  {target}: PID {pid_out.strip()} - {alive.strip()}")

    for target in ("helmet", "full", "plate"):
        log_file = f"{REMOTE_BASE}/logs/{target}.log"
        out, _, code = _run(client, f"tail -8 {log_file} 2>/dev/null")
        if code == 0 and out:
            print(f"\n--- {target} (last 8 lines) ---")
            print(out.encode("cp1252", errors="replace").decode("cp1252"))


def tail_log(client: paramiko.SSHClient, target: str, lines: int = 40) -> None:
    """Print the last *lines* of a model's training log."""
    log_file = f"{REMOTE_BASE}/logs/{target}.log"
    out, err, _ = _run(client, f"tail -{lines} {log_file} 2>&1", timeout=10)
    print(out or err or f"(no log at {log_file})")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy and run training on the GPU server.")
    p.add_argument("--target", choices=["helmet", "full", "plate", "all"], default="all")
    p.add_argument("--rf-key", default=os.environ.get("RF_API_KEY", ""),
                   help="Roboflow API key (or set RF_API_KEY env var).")
    p.add_argument("--tune",   action="store_true",
                   help="Run hyperparameter tuning before full training.")
    p.add_argument("--status", action="store_true",
                   help="Print status of running training jobs and exit.")
    p.add_argument("--logs",   default=None, choices=["helmet", "full", "plate"],
                   help="Tail logs for a specific model and exit.")
    p.add_argument("--setup-only", action="store_true",
                   help="Only install dependencies + upload files, do not start training.")
    return p.parse_args()


import os   # noqa: E402 - placed after the main definitions intentionally


def main() -> int:
    args = _parse_args()

    print(f"Connecting to {USER}@{HOST}…")
    try:
        client = _connect()
    except Exception as exc:
        print(f"Connection failed: {exc}")
        return 1
    print("Connected.\n")

    try:
        # Status-only modes
        if args.status:
            status(client)
            return 0

        if args.logs:
            tail_log(client, args.logs)
            return 0

        # Full deployment
        setup_environment(client)
        _scp_upload(client)

        if args.setup_only:
            print("\nSetup complete. Training not started (--setup-only).")
            return 0

        targets = ["helmet", "full", "plate"] if args.target == "all" else [args.target]

        print(f"\n=== Launching training: {targets} ===")
        for t in targets:
            launch_training(client, t, args.rf_key, tune=args.tune)
            time.sleep(2)   # stagger launches slightly

        print("\n=== All jobs launched ===")
        print("Monitor with:  python _deploy.py --status")
        print("Tail logs with: python _deploy.py --logs helmet")
        print("\nTraining will run inside detached 'screen' sessions.")
        print("SSH in directly with:  ssh sem6@172.16.121.74")
        print("Attach screen:         screen -r train_helmet")

        # Show initial status
        time.sleep(5)
        status(client)

    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
