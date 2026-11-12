"""MDMT adapter for the H1 0B pilot (zip of the unofficial HF mirror, read in place; nothing is extracted).

Layout inside the zip (verified 2026-11-12, pilot protocol sec 9):
    <root>/{train,val,test}/{1,2}/<seq>-<view>/%08d.jpg      images, file index 1-based
    <root>/new_xml/{1,2}/<seq>-<view>.xml                    CVAT tracks; frame and track id 0-based
Image file numbers do NOT always start at 1 (test 52: 191..550, 55: 550..700, 56, 57, 61, 68 likewise), while XML
frames always run 0..n-1. So XML frame f belongs to the f-th image file in sorted order (rank), not to file f + 1
(the two coincide only when files start at 1). Both views must hold the same file number at every rank.
A sequence is `aligned` only if, in both views, the label frames are exactly 0..n_images-1 (registered 11-12 before
any image of these sequences was opened; test 55 view 1 has 151 images but 150 label frames -> excluded).
Track ids are shared by the two views of a sequence.

Rules enforced here (pilot protocol sec 2 / 8 / 9, review #8):
  * only .jpg and .xml members are ever opened; checkpoints (.pth) and scripts (.py) in the mirror are never read;
  * split access: only the splits in `allowed` (default: test = EXPL). `val` is kept unopened, `train` has no pairs;
  * images of the real dataset are only served when `rights_note` is given (a written statement of the basis for
    using the data); the note is returned by `provenance()` so every result file records it;
  * GT isolation: `image()` returns pixels only, `labels()` is a separate call used by the scorer / marked oracles.
"""
import io
import re
import zipfile
from hashlib import blake2b

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
CLASSES = ("person", "car", "bicycle", "bus")
_IMG = re.compile(r"(?:^|/)(train|val|test)/([12])/(\d+)-([12])/(\d{8})\.jpg$")
_XML = re.compile(r"(?:^|/)new_xml/([12])/(\d+)-([12])\.xml$")
_TRACK = re.compile(r'<track\b([^>]*)>(.*?)</track>', re.S)
_BOX = re.compile(r'<box\b([^>]*)/?>')
_ATTR = re.compile(r'(\w+)="([^"]*)"')


class AccessError(RuntimeError):
    pass


def seq_hash(seq):
    return int(blake2b(str(seq).encode(), digest_size=8).hexdigest(), 16)


def is_hold(seq):
    """HOLD rule registered in pilot protocol sec 2 (applies to the full dataset; unused by the EXPL pilot)."""
    return seq_hash(seq) % 4 == 0


def parse_cvat(text):
    """-> {frame(0-based): {track_id: (x1, y1, x2, y2, class_index)}}; boxes with outside=1 are skipped, tracks of
    unknown labels raise (no silent class loss)."""
    out = {}
    for head, body in _TRACK.findall(text):
        a = dict(_ATTR.findall(head))
        if a["label"] not in CLASSES:
            raise ValueError(f"unknown label {a['label']!r}")
        tid, c = int(a["id"]), CLASSES.index(a["label"])
        for b in _BOX.findall(body):
            d = dict(_ATTR.findall(b))
            if d.get("outside", "0") == "1":
                continue
            x1, y1, x2, y2 = float(d["xtl"]), float(d["ytl"]), float(d["xbr"]), float(d["ybr"])
            if x2 <= x1 or y2 <= y1:
                continue
            out.setdefault(int(d["frame"]), {})[tid] = (x1, y1, x2, y2, c)
    return out


class MDMT:
    def __init__(self, zip_path, allowed=("test",), rights_note=None):
        bad = set(allowed) - set(SPLITS)
        if bad:
            raise ValueError(bad)
        self.path, self.allowed, self.rights_note = str(zip_path), tuple(allowed), rights_note
        self.z = zipfile.ZipFile(zip_path)
        self._img, self._xml, self._split = {}, {}, {}
        for n in self.z.namelist():                      # names only; no member is opened here
            m = _IMG.search(n)
            if m:
                sp, d, seq, v, f = m.groups()
                if d != v:
                    continue
                self._img.setdefault((int(seq), int(v)), {})[int(f)] = n
                self._split[int(seq)] = sp
                continue
            m = _XML.search(n)
            if m and m.group(1) == m.group(3):
                self._xml[(int(m.group(2)), int(m.group(3)))] = n

    # ---- listing (names only)
    def split_of(self, seq):
        return self._split[seq]

    def pairs(self, split):
        """sequences of `split` with images AND labels for both views."""
        return sorted(s for s, sp in self._split.items() if sp == split
                      and all((s, v) in self._img and (s, v) in self._xml for v in (1, 2)))

    def _files(self, seq, view):
        return sorted(self._img[(seq, view)])

    def frames(self, seq):
        """0-based frames (= ranks) present in both views; the file numbers of the two views must agree."""
        a, b = self._files(seq, 1), self._files(seq, 2)
        n = min(len(a), len(b))
        if a[:n] != b[:n]:
            raise ValueError(f"sequence {seq}: views are not synchronised by file number")
        return list(range(n))

    def aligned(self, seq):
        """in both views the last annotated frame (outside boxes included) is n_images - 1 (frames with no visible
        object may be absent). Reads the XML of an opened split only."""
        for v in (1, 2):
            self._check(seq, False)
            fr = [int(x) for x in re.findall(r'<box\b[^>]*\bframe="(\d+)"', self.z.read(self._xml[(seq, v)]).decode())]
            if not fr or max(fr) != len(self._files(seq, v)) - 1:
                return False
        return True

    def epochs(self, seq, step=10, max_epochs=60):
        return self.frames(seq)[::step][:max_epochs]

    def pilot_order(self, split):
        """aligned pairs, deterministic order (smallest hash first) for picking the sequences of a small pilot."""
        return sorted((s for s in self.pairs(split) if self.aligned(s)), key=seq_hash)

    # ---- guarded access
    def _check(self, seq, pixels):
        sp = self._split[seq]
        if sp not in self.allowed:
            raise AccessError(f"sequence {seq} is in split {sp!r}, which is not opened ({self.allowed})")
        if pixels and not self.rights_note:
            raise AccessError("images are only served with a rights_note (basis for using the data)")

    def image(self, seq, view, frame):
        """RGB uint8 HxWx3 of 0-based `frame`. Pixels only: no label is touched."""
        self._check(seq, True)
        name = self._img[(seq, view)][self._files(seq, view)[frame]]
        return np.asarray(Image.open(io.BytesIO(self.z.read(name))).convert("RGB"))

    def labels(self, seq, view):
        """GT for the scorer / marked oracles only."""
        self._check(seq, False)
        return parse_cvat(self.z.read(self._xml[(seq, view)]).decode("utf-8"))

    def provenance(self):
        return dict(source=self.path, allowed=list(self.allowed), rights_note=self.rights_note)
