# SwarmBridge

Research status, consolidated results, protocols and document index:
[reports/README.md](reports/README.md).

Some research data files exceed GitHub's 100 MiB limit. They are stored as
numbered parts alongside their original locations, with sizes and SHA-256 hashes
recorded in `large_file_manifest.json`. To reconstruct and verify them after
cloning, run:

```powershell
python scripts/reassemble_large_files.py restore
```

Existing files are verified and never overwritten if their contents differ.
Model checkpoints and `pdf/GithubModels/` are excluded from the repository.
