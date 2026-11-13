"""List a remote zip (HTTP range requests) and optionally extract small non-image members. No bulk download.
Usage: python -I remote_zip.py URL OUT_DIR [list | get SUFFIXES MAX_BYTES_PER_FILE MAX_FILES]"""
import io
import json
import os
import sys
import urllib.request
import zipfile

UA = {"User-Agent": "python-urllib"}


class Remote(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.n_req, self.n_bytes = url, 0, 0, 0
        self.url0 = url
        r = urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"}), timeout=60)
        self.url = r.geturl()                                   # follow the redirect once
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.cache = {}

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, k, bs=1 << 18):
        if k not in self.cache:
            a, b = k * bs, min((k + 1) * bs, self.size) - 1
            for attempt in range(6):
                try:
                    # the signed redirect target expires quickly: on failure go back to the stable URL
                    u = self.url if attempt == 0 else self.url0
                    r = urllib.request.urlopen(urllib.request.Request(u, headers={**UA, "Range": f"bytes={a}-{b}"}),
                                               timeout=120)
                    self.url = r.geturl()
                    data = r.read()
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    import time
                    time.sleep(2 * (attempt + 1))
            self.cache[k] = data; self.n_req += 1; self.n_bytes += len(self.cache[k])
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[k]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out, bs = b"", 1 << 18
        while n > 0:
            k, o = divmod(self.pos, bs)
            chunk = self._block(k)[o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def main():
    url, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    f = Remote(url)
    z = zipfile.ZipFile(io.BufferedReader(f, 1 << 18))
    infos = z.infolist()
    if sys.argv[3] == "list":
        json.dump([(i.filename, i.file_size) for i in infos], open(os.path.join(out, "listing.json"), "w"))
        print("members", len(infos), "total uncompressed", sum(i.file_size for i in infos), "zip size", f.size)
    else:
        suf, mx, mxn = tuple(sys.argv[4].split(",")), int(sys.argv[5]), int(sys.argv[6])
        n = 0
        for i in infos:
            if i.is_dir() or not i.filename.lower().endswith(suf) or i.file_size > mx or n >= mxn:
                continue
            name = i.filename.replace("\\", "/")
            if ".." in name.split("/") or name.startswith("/"):
                continue
            p = os.path.join(out, "files", *name.split("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(z.read(i)); n += 1
        print("extracted", n)
    print("range requests", f.n_req, "bytes fetched", f.n_bytes)


if __name__ == "__main__":
    main()
