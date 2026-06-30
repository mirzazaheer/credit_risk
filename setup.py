"""
Bootstrap script — run once after cloning.
Works on macOS, Linux (x86_64 / arm64), and Windows.

Usage:
    python setup.py              # full setup including Ollama + Llama 3
    python setup.py --skip-ollama  # skip the LLM step (app still works, uses rule-based reports)
"""

import os
import ssl
import sys
import stat
import time
import platform
import tarfile
import zipfile
import subprocess
import urllib.request
import urllib.error

# ── Constants ─────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
BIN_DIR    = os.path.join(BASE, "bin")
IS_WINDOWS = platform.system().lower() == "windows"
OLLAMA_EXE = "ollama.exe" if IS_WINDOWS else "ollama"
OLLAMA_BIN = os.path.join(BIN_DIR, OLLAMA_EXE)
PYTHON     = sys.executable          # use the exact Python running this script
SKIP_OLLAMA = "--skip-ollama" in sys.argv


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(text):
    print("\n" + "=" * 56)
    print(f"  {text}")
    print("=" * 56)

def step(text):
    print(f"\n  {text}")
    print(f"  {'─' * 52}")

def run(script_path, desc):
    """Run a Python script using the same interpreter as this setup script."""
    step(desc)
    result = subprocess.run([PYTHON, script_path], cwd=BASE)
    if result.returncode != 0:
        print(f"\n  [ERROR] Failed: {desc}")
        sys.exit(1)

def find_system_ollama():
    """Return path to system-installed ollama, or None."""
    check_cmd = "where" if IS_WINDOWS else "which"
    result = subprocess.run(
        [check_cmd, "ollama"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        path = result.stdout.strip().splitlines()[0].strip()
        if path:
            return path
    return None

def find_ollama():
    """Return ollama binary path (system or local bin/), or None."""
    sys_path = find_system_ollama()
    if sys_path:
        return sys_path
    if os.path.isfile(OLLAMA_BIN):
        return OLLAMA_BIN
    return None

def get_ollama_download_url():
    system  = platform.system().lower()
    machine = platform.machine().lower()
    base    = "https://github.com/ollama/ollama/releases/latest/download/"

    if system == "darwin":
        return base + "ollama-darwin.tgz", "tgz"
    if system == "linux":
        if "aarch64" in machine or "arm64" in machine:
            return base + "ollama-linux-arm64.tgz", "tgz"
        return base + "ollama-linux-amd64.tgz", "tgz"
    if system == "windows":
        return base + "ollama-windows-amd64.zip", "zip"
    return None, None

def download_ollama():
    url, fmt = get_ollama_download_url()
    if url is None:
        print(f"  [WARNING] Unsupported OS: {platform.system()}")
        print("  Download Ollama manually from https://ollama.com/download")
        return None

    os.makedirs(BIN_DIR, exist_ok=True)
    archive = os.path.join(BIN_DIR, f"ollama_dl.{fmt}")

    print(f"  Downloading Ollama ({platform.system()} {platform.machine()})...")

    def _do_download(ssl_ctx):
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        with opener.open(url) as resp, open(archive, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done * 100 // total}%  ({done // 1_000_000} / {total // 1_000_000} MB)   ", end="", flush=True)
        print()

    # Try verified → certifi → unverified (in that order)
    ssl_contexts = [ssl.create_default_context()]
    try:
        import certifi
        ssl_contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    ssl_contexts.append(unverified)

    downloaded = False
    for ctx in ssl_contexts:
        try:
            _do_download(ctx)
            downloaded = True
            break
        except ssl.SSLError:
            continue
        except urllib.error.URLError as e:
            if "CERTIFICATE" in str(e).upper():
                continue
            print(f"\n  [ERROR] Download failed: {e}")
            print("  Download Ollama manually from https://ollama.com/download")
            return None

    if not downloaded:
        print("\n  [ERROR] Could not download Ollama (SSL error on all attempts).")
        print("  Download manually from https://ollama.com/download")
        return None

    print("  Extracting...")
    if fmt == "tgz":
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name.split("/")[-1]
                if name == "ollama":
                    member.name = "ollama"
                    tar.extract(member, path=BIN_DIR, filter="data")
                    break
    else:  # zip (Windows)
        with zipfile.ZipFile(archive, "r") as zf:
            for entry in zf.namelist():
                if entry.lower().endswith("ollama.exe") and "/" not in entry.lstrip("/"):
                    zf.extract(entry, path=BIN_DIR)
                    extracted = os.path.join(BIN_DIR, entry)
                    target    = os.path.join(BIN_DIR, "ollama.exe")
                    if extracted != target:
                        os.replace(extracted, target)
                    break

    os.remove(archive)

    # Set executable bit on Unix
    if not IS_WINDOWS:
        current = os.stat(OLLAMA_BIN).st_mode
        os.chmod(OLLAMA_BIN, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return OLLAMA_BIN

def is_server_running():
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False

def start_ollama_server(ollama_path):
    """Start ollama serve in the background."""
    kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen([ollama_path, "serve"], **kwargs)

    print("  Starting Ollama server", end="", flush=True)
    for _ in range(20):
        time.sleep(1)
        print(".", end="", flush=True)
        if is_server_running():
            print(" ready.")
            return True
    print()
    return False

def pull_llama3(ollama_path):
    print("  Pulling Llama 3 model (~4 GB) — this takes a few minutes...")
    result = subprocess.run([ollama_path, "pull", "llama3"])
    if result.returncode != 0:
        print("  [WARNING] Could not pull llama3.")
        print(f"            Run manually: {ollama_path} pull llama3")

def ollama_serve_hint(ollama_path):
    rel = os.path.relpath(ollama_path, BASE)
    if IS_WINDOWS:
        return rel.replace("/", "\\")
    return rel if not os.path.isabs(rel) else ollama_path


# ── Main ─────────────────────────────────────────────────────────────────────
banner("E-commerce MSME Credit Risk Screener — Setup")

# Step 1: Generate dataset
run(os.path.join("data", "generate_synthetic.py"),
    "Step 1/3  Generating synthetic MSME dataset (1,200 records)...")

# Step 2: Train model
run(os.path.join("model", "train.py"),
    "Step 2/3  Training Random Forest credit risk model...")

# Step 3: Ollama + Llama 3
if SKIP_OLLAMA:
    step("Step 3/3  Skipping Ollama (--skip-ollama)")
    print("          App will use rule-based reports instead of AI narratives.")
else:
    step("Step 3/3  Setting up Ollama + Llama 3 (AI report generator)")

    ollama_path = find_ollama()

    if ollama_path:
        print(f"  Ollama found: {ollama_path}")
    else:
        print("  Ollama not found — downloading now...")
        ollama_path = download_ollama()
        if ollama_path:
            print(f"  Downloaded to: {ollama_path}")

    if ollama_path:
        if is_server_running():
            print("  Ollama server is already running.")
            pull_llama3(ollama_path)
        else:
            if start_ollama_server(ollama_path):
                pull_llama3(ollama_path)
            else:
                print("  [WARNING] Ollama server did not start in time.")
                hint = ollama_serve_hint(ollama_path)
                print(f"            Start it manually:  {hint} serve")
                print("            Then pull the model: ollama pull llama3")

# ── Done ─────────────────────────────────────────────────────────────────────
banner("Setup complete!")

sep = "\\" if IS_WINDOWS else "/"
ollama_local = f"bin{sep}{OLLAMA_EXE}"
system_ollama = find_system_ollama()

print("""
  HOW TO RUN
  ──────────""")

if not SKIP_OLLAMA:
    ollama_cmd = system_ollama if system_ollama else ollama_local
    print(f"  1. Start Ollama (keep this terminal open):")
    print(f"       {ollama_cmd} serve\n")
    print(f"  2. Launch the app (new terminal):")
else:
    print(f"  Launch the app:")

print(f"       streamlit run app.py")
print(f"\n  Then open: http://localhost:8501")
print()
