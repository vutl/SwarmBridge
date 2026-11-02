"""Extract a page-indexed local reading cache without changing any PDF."""
import hashlib
import json
import logging
import re
import tempfile
from pathlib import Path

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)
root = Path(__file__).resolve().parent.parent
cache = Path(tempfile.gettempdir()) / "swarmbridge_literature_20261030"
cache.mkdir(exist_ok=True)
rows = []
for index, file in enumerate(sorted((root / "pdf").glob("*.pdf"))):
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    dest = cache / f"{digest[:16]}.txt"
    reader = PdfReader(file)
    if not dest.exists() or dest.stat().st_size < 100:
        extracted = "\n\n".join(f"[PAGE {i+1}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
        dest.write_text(extracted, encoding="utf-8", errors="replace")
    text = dest.read_text(encoding="utf-8")
    links = []
    for page in reader.pages:
        for reference in page.get("/Annots", []):
            annotation = reference.get_object()
            action = annotation.get("/A", {})
            if action and action.get("/URI"):
                links.append(str(action["/URI"]))
    links += re.findall(r"https?://[^\s<>]+", text)
    rows.append(dict(id=index, file=file.name, pages=len(reader.pages), sha256=digest,
                     text=str(dest), chars=len(text), links=list(dict.fromkeys(links))))
(cache / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Indexed {len(rows)} PDFs, {sum(r['pages'] for r in rows)} pages; cache: {cache}")
for row in rows:
    print(row["id"], row["pages"], row["file"])
