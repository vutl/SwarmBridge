"""Minimal zip test / extract (WSL Ubuntu has no `unzip`). zipfile strips absolute paths and '..' on extract.
Usage: python -I zip_tool.py test ZIP      -> exit 0 if every member's CRC is OK
       python -I zip_tool.py x ZIP DEST    -> extract, skipping members that already exist with the same size
"""
import sys
import zipfile
from pathlib import Path


def main():
    mode, zp = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(zp) as z:
        if mode == "test":
            bad = z.testzip()
            print(zp, "members", len(z.infolist()), "bad", bad)
            return 0 if bad is None else 1
        dest = Path(sys.argv[3])
        n = 0
        for m in z.infolist():
            t = dest / m.filename
            if not m.is_dir() and t.exists() and t.stat().st_size == m.file_size:
                continue
            z.extract(m, dest)
            n += 1
        print(zp, "extracted", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
