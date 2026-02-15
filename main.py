import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Commander handles object-detection subprocesses internally.
TARGETS = [
    ROOT / "EEG" / "eeg_visualizer.py",
    ROOT / "PythonInput" / "Commander.py",
]


def start_process(script_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, script_path.name],
        cwd=str(script_path.parent),
    )


def stop_all(procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 5
    for p in procs:
        if p.poll() is None:
            remaining = max(0, deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    missing = [str(p) for p in TARGETS if not p.exists()]
    if missing:
        print("Missing script(s):")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    procs = []
    running = True

    def _handle_stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        for target in TARGETS:
            proc = start_process(target)
            procs.append(proc)
            print(f"Started {target} (pid={proc.pid})")

        while running:
            for p in procs:
                code = p.poll()
                if code is not None:
                    print(f"Process exited (pid={p.pid}, code={code}). Stopping all.")
                    running = False
                    break
            time.sleep(0.2)
    finally:
        stop_all(procs)
        print("All processes stopped.")


if __name__ == "__main__":
    main()
