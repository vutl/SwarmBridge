"""Run a command while holding the shared SwarmBridge GPU lock (runs/.swarmbridge-gpu.lock), from WSL or Windows.
Standalone (python >= 3.6, no project imports) so the legacy py3.8 env can use it.

  python gpulock_run.py REPO_ROOT -- cmd args...
Under WSL the Windows msvcrt lock is taken through a child Windows python (same scheme as sb_audit/run.py gpu_lock).
Fails fast if another job holds the lock.
"""
import os
import platform
import subprocess
import sys

HOLDER = r"""
import msvcrt, sys
f = open(sys.argv[1], 'a+b'); f.seek(0)
try:
    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    print('BUSY', flush=True); sys.exit(1)
print('LOCKED', flush=True)
sys.stdin.read()
f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
"""


def main():
    repo = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    lock = os.path.join(repo, "runs", ".swarmbridge-gpu.lock")
    if "microsoft" in platform.release().lower():
        win_py = os.path.join(repo, ".venv", "Scripts", "python.exe")
        win_lock = "runs/.swarmbridge-gpu.lock"
        holder = subprocess.Popen([win_py, "-c", HOLDER, win_lock], cwd=repo, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, universal_newlines=True)
    else:
        holder = subprocess.Popen([sys.executable, "-c", HOLDER, lock], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, universal_newlines=True)
    if holder.stdout.readline().strip() != "LOCKED":
        holder.wait()
        print("GPU lock busy (another job is running); not starting.", flush=True)
        return 2
    print("GPU lock acquired", flush=True)
    try:
        return subprocess.call(cmd)
    finally:
        holder.stdin.close()
        holder.wait()
        print("GPU lock released", flush=True)


if __name__ == "__main__":
    sys.exit(main())
