"""
Bootstrap script — run once after cloning to set up everything.
Usage: python setup.py [--skip-ollama]

What it does:
  1. Generates the synthetic MSME dataset
  2. Trains the Random Forest model
  3. Downloads Ollama (if not already installed)
  4. Pulls the Llama 3 model via Ollama
"""

import subprocess
import sys
import os
import platform
import urllib.request
import tarfile
import stat

BASE = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE, "bin")
OLLAMA_BIN = os.path.join(BIN_DIR, "ollama")
SKIP_OLLAMA = "--skip-ollama" in sys.argv


def run(cmd, desc, cwd=BASE):
    print(f"\n  {desc}")
    print(f"  {'─' * 50}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\n  [ERROR] Failed: {desc}")
        sys.exit(1)


def check_ollama():
    """Return path to ollama binary if available, else None."""
    # Check system PATH first
    result = subprocess.run("which ollama", shell=True, capture_output=True)
    if result.returncode == 0:
        return result.stdout.decode().strip()
    # Check local bin/
    if os.path.isfile(OLLAMA_BIN):
        return OLLAMA_BIN
    return None


def download_ollama():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        url = "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz"
    elif system == "linux" and "arm" in machine:
        url = "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64.tgz"
    elif system == "linux":
        url = "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz"
    else:
        print("  [WARNING] Automatic Ollama install not supported on Windows.")
        print("  Download manually from https://ollama.com/download and re-run setup.")
        return None

    os.makedirs(BIN_DIR, exist_ok=True)
    tgz_path = os.path.join(BIN_DIR, "ollama.tgz")

    print(f"  Downloading Ollama from GitHub releases...")
    urllib.request.urlretrieve(url, tgz_path)

    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name == "ollama" or member.name.endswith("/ollama"):
                member.name = "ollama"
                tar.extract(member, path=BIN_DIR)
                break

    os.remove(tgz_path)
    os.chmod(OLLAMA_BIN, os.stat(OLLAMA_BIN).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return OLLAMA_BIN


def is_ollama_server_running():
    try:
        import urllib.request as req
        req.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def setup_ollama(ollama_path):
    # Pull Llama 3 model
    print("  Pulling Llama 3 model (~4 GB). This will take a few minutes...")
    result = subprocess.run(f"{ollama_path} pull llama3", shell=True)
    if result.returncode != 0:
        print("  [WARNING] Could not pull llama3. You can do this manually: ollama pull llama3")


# ── Main ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 54)
print("  E-commerce MSME Credit Risk Screener — Setup")
print("=" * 54)

# Step 1: Generate data
run("python3 data/generate_synthetic.py",
    "Step 1/3  Generating synthetic MSME dataset (1,200 records)...")

# Step 2: Train model
run("python3 model/train.py",
    "Step 2/3  Training Random Forest credit risk model...")

# Step 3: Ollama
if SKIP_OLLAMA:
    print("\n  Step 3/3  Skipping Ollama setup (--skip-ollama flag set).")
    print("            The app will use rule-based reports instead of AI narratives.")
else:
    print("\n  Step 3/3  Setting up Ollama + Llama 3 (AI report generator)")
    print(f"  {'─' * 50}")

    ollama_path = check_ollama()

    if ollama_path:
        print(f"  Ollama already installed at: {ollama_path}")
    else:
        print("  Ollama not found — downloading now...")
        ollama_path = download_ollama()
        if ollama_path:
            print(f"  Ollama downloaded to: {ollama_path}")

    if ollama_path:
        if is_ollama_server_running():
            print("  Ollama server already running.")
            setup_ollama(ollama_path)
        else:
            # Start server in background then pull
            subprocess.Popen(
                f"{ollama_path} serve",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time
            print("  Starting Ollama server...", end="", flush=True)
            for _ in range(15):
                time.sleep(1)
                print(".", end="", flush=True)
                if is_ollama_server_running():
                    break
            print()
            if is_ollama_server_running():
                setup_ollama(ollama_path)
            else:
                print("  [WARNING] Ollama server did not start in time.")
                print("            Start it manually with:  bin/ollama serve")

# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 54)
print("  Setup complete!")
print("=" * 54)
print()
print("  Run the app:")
print("    streamlit run app.py")
print()
if not SKIP_OLLAMA:
    print("  If Ollama is not already running, start it first:")
    if os.path.isfile(OLLAMA_BIN):
        print("    bin/ollama serve")
    else:
        print("    ollama serve")
    print()
print("=" * 54 + "\n")
