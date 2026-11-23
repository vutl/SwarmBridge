"""Run a WSL shell script while holding the shared SwarmBridge GPU lock (runs/.swarmbridge-gpu.lock, msvcrt byte lock).
Runs under the WINDOWS python (repo .venv), because the WSL -> Windows-python route of uav3d_comm/gpulock_run.py is
broken. Fails fast if another job holds the lock. Suitable for a detached WMI launch:
  <repo>\\.venv\\Scripts\\python.exe fd_gpulock_win.py <repo> <wsl-path-of-script.sh>"""
import msvcrt
import os
import subprocess
import sys


def main():
    repo, script = sys.argv[1], sys.argv[2]
    f = open(os.path.join(repo, "runs", ".swarmbridge-gpu.lock"), "a+b")
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("GPU lock busy (another job is running); not starting.", flush=True)
        return 2
    print("GPU lock acquired", flush=True)
    try:
        return subprocess.call(["wsl.exe", "-d", "Ubuntu", "-u", "uservutl", "--", "bash", script])
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        print("GPU lock released", flush=True)


if __name__ == "__main__":
    sys.exit(main())
