"""Download the user's reading list, verifying PDFs and preserving existing files."""
import concurrent.futures
import hashlib
import io
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "pdf"
REPOS = PDF / "GithubModels"
OUT = ROOT / "reports" / "metadata" / "danh-sach-tai-lieu-2026-10-30.json"
A = "https://arxiv.org/pdf/"
papers = [
 ("01", "Wireless Communications with Unmanned Aerial Vehicles Opportunities and Challenges", [A+"1602.03602"]),
 ("02", "Accessing From the Sky A Tutorial on UAV Communications for 5G and Beyond", [A+"1903.05289"]),
 ("03", "Deep Joint Source-Channel Coding for Wireless Image Transmission", [A+"1809.01733"]),
 ("04", "Deep Learning Enabled Semantic Communication Systems", [A+"2006.10685"]),
 ("05", "Semantic Communications in Networked Systems A Data Significance Perspective", [A+"2103.05391"]),
 ("06", "The Age of Incorrect Information An Enabler of Semantics-Empowered Communication", [A+"2012.13214"]),
 ("07", "Cooperative Internet of UAVs Distributed Trajectory Design by Multi-Agent Deep Reinforcement Learning", [A+"2007.14297"]),
 ("08", "Multi-Agent Deep Reinforcement Learning Based Trajectory Planning for Multi-UAV Assisted Mobile Edge Computing", [A+"2009.11277"]),
 ("09", "Multi-Agent Reinforcement Learning Based Resource Allocation for UAV Networks", [A+"1810.10408"]),
 ("10", "NOMA in UAV-aided Cellular Offloading A Machine Learning Approach", [A+"2011.14776"]),
 ("11", "Performance Optimization for Semantic Communications An Attention-Based Reinforcement Learning Approach", [A+"2208.08239","https://oar.princeton.edu/bitstream/88435/pr1mk65820/1/2208.08239.pdf"]),
 ("12", "UAV-assisted Semantic Communication with Hybrid Action Reinforcement Learning", [A+"2309.16713"]),
 ("13", "Joint Optimization Trajectory and Resource Allocation for UAV-Assisted Semantic Communications", ["https://www.sciencedirect.com/science/article/pii/S1874490724002738/pdfft?isDTMRedir=true&download=true"]),
 ("14", "Resource Allocation for Multi-Modal Semantic Communication in UAV Collaborative Networks", ["https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10930657"]),
 ("15", "Resource Allocation for UAV-Assisted Anti-Jamming Semantic D2D Networks A Graph Reinforcement Learning Approach", ["https://www.sciencedirect.com/science/article/pii/S138912862500430X/pdfft?isDTMRedir=true&download=true"]),
 ("16", "Decision Transformer Reinforcement Learning via Sequence Modeling", [A+"2106.01345"]),
 ("17", "Conservative Q-Learning for Offline Reinforcement Learning", [A+"2006.04779"]),
 ("18", "Decision Transformers for Wireless Communications A New Paradigm of Resource Management", [A+"2404.05199"]),
 ("19", "Decision Transformer-Based Drone Trajectory Planning with Dynamic Safety Efficiency Trade-Offs", [A+"2507.21506"]),
 ("20", "Constrained Decision Transformer for Offline Safe Reinforcement Learning", ["https://proceedings.mlr.press/v202/liu23m/liu23m.pdf"]),
 ("21", "When Should We Prefer Decision Transformers for Offline Reinforcement Learning", ["https://openreview.net/pdf?id=vpV7fOFQy4",A+"2305.14550"]),
 ("22", "Adaptive Task-Oriented Semantic Communication for UAVs With Joint Trajectory and Resource Optimization", ["https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11664598"]),
 ("23", "Predictive and Adaptive Semantic Communication for QoE Optimization in Multi-UAV Networks", ["https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11615274"]),
 ("24", "Adaptive Cross-Modal UAV Semantic Communications via Hierarchical DRL", ["https://link.springer.com/content/pdf/10.1186/s13638-026-02664-z_reference.pdf"]),
 ("25", "Deep Graph Reinforcement Learning for UAV-Enabled Multi-User Secure Communications", [A+"2504.01446"]),
 ("26", "Collaborative Computation and Migration in Multi-UAV Networks with Semantic Communication", [A+"2609.14476"]),
 ("R5", "Reinforcement Learning-powered Semantic Communication via Semantic Similarity", [A+"2108.12121"]),
 ("R6", "UavNetSim-v1 A Python-based Simulation Platform for UAV Communication Networks", [A+"2507.09852"]),
 ("R3", "Computation Offloading Optimization for UAV-assisted Mobile Edge Computing A Deep Deterministic Policy Gradient Approach", ["https://link.springer.com/content/pdf/10.1007/s11276-021-02632-z.pdf"]),
]
repos = [
 ("vale817", "DeepJSCC"), ("amanjaiswal-07", "UAV-PATH-OPTIMIZATION"),
 ("fangvv", "UAV-DDPG"), ("13274086", "DeepSC"),
 ("lukun199", "SemanticRL"), ("Zihao-Felix-Zhou", "UavNetSim-v1"),
]

def fetch(url, limit=120_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research reading list)", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = r.read(limit + 1)
        if len(data) > limit:
            raise ValueError("download exceeds 120 MB")
        return data, r.geturl()

def normalized(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def get_paper(item):
    number, title, urls = item
    path = PDF / (title + ".pdf")
    result = {"id": number, "title": title, "urls": urls}
    if path.exists():
        result.update(status="already exists", file=str(path.relative_to(ROOT)))
        return result
    errors = []
    for url in urls:
        try:
            data, final_url = fetch(url)
            if not data.startswith(b"%PDF-"):
                raise ValueError("response is not a PDF")
            reader = PdfReader(io.BytesIO(data))
            first = " ".join((p.extract_text() or "") for p in reader.pages[:2])
            # Catch incorrect IEEE links and publisher login/error PDFs.
            words = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 3]
            compact = normalized(first)
            overlap = sum(w in compact for w in words) / max(len(words), 1)
            if overlap < 0.45:
                raise ValueError(f"PDF title does not match expected title (overlap {overlap:.2f})")
            path.write_bytes(data)
            result.update(status="downloaded", file=str(path.relative_to(ROOT)), source=final_url,
                          pages=len(reader.pages), sha256=hashlib.sha256(data).hexdigest())
            return result
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:180]}")
    result.update(status="failed", errors=errors)
    return result

def get_repo(item):
    owner, name = item
    path = REPOS / name
    result = {"repository": f"https://github.com/{owner}/{name}"}
    if path.exists():
        result.update(status="already exists", directory=str(path.relative_to(ROOT)))
        return result
    errors = []
    try:
        metadata, _ = fetch(f"https://api.github.com/repos/{owner}/{name}", limit=1_000_000)
        default_branch = json.loads(metadata)["default_branch"]
        branches = [default_branch]
    except Exception as exc:
        result.update(status="failed", errors=[f"Could not determine default branch: {type(exc).__name__}: {exc}"])
        return result
    for branch in branches:
        url = f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/{branch}"
        try:
            data, final = fetch(url)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                entries = z.infolist()
                if sum(i.file_size for i in entries) > 500_000_000:
                    raise ValueError("extracted archive exceeds 500 MB")
                for info in entries:
                    parts = Path(info.filename.replace("\\", "/")).parts
                    if len(parts) < 2 or any(p in ("..", ".git") for p in parts):
                        continue
                    dest = path.joinpath(*parts[1:])
                    if not dest.resolve().is_relative_to(path.resolve()):
                        raise ValueError("unsafe archive path")
                    if info.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(info) as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            result.update(status="downloaded", directory=str(path.relative_to(ROOT)),
                          branch=branch, source=final, zip_bytes=len(data),
                          note="Source archive; no Git history; Git LFS pointers may require separate downloads.")
            return result
        except Exception as exc:
            if path.exists():
                # Preserve a partial extraction for inspection instead of deleting it.
                result.update(status="failed", directory=str(path.relative_to(ROOT)),
                              errors=errors + [f"{url}: {type(exc).__name__}: {str(exc)[:180]}"])
                return result
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:180]}")
    result.update(status="failed", errors=errors)
    return result

def main():
    PDF.mkdir(exist_ok=True)
    REPOS.mkdir(exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        paper_results = list(pool.map(get_paper, papers))
        repo_results = list(pool.map(get_repo, repos))
    if OUT.exists():
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        for category, rows, key in (("papers", paper_results, "id"), ("repositories", repo_results, "repository")):
            saved = {row[key]: row for row in previous[category]}
            for index, row in enumerate(rows):
                if row["status"] == "already exists" and row[key] in saved:
                    rows[index] = saved[row[key]]
    paper_results.append({
        "id": "R2",
        "title": "DDQN-Enabled UAV Path Planning in Non-Terrestrial Networks Under ICAS Constraints",
        "status": "included in repository",
        "file": "pdf/GithubModels/UAV-PATH-OPTIMIZATION/DDQN-Enabled_UAV_Path_Planning_in_Non-Terrestrial_Networks_Under_ICAS_Constraints.pdf",
        "note": "The title in the source repository differs from the user's pasted description."
    })
    OUT.write_text(json.dumps({"papers": paper_results, "repositories": repo_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in paper_results + repo_results:
        print(item.get("id", item.get("repository")), item["status"], flush=True)
    print("Manifest:", OUT)

if __name__ == "__main__":
    main()
