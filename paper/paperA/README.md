# Paper A: bản thảo

- `main.tex`: bản nháp đầu (IEEEtran journal).
  - Đã viết: System Model, Controllers, Experimental Setup, Results.
  - Còn khung: Introduction, Related Work, Conclusion.
  - Chỗ chưa kiểm với code được đánh dấu `\todo{}` (in đỏ).
- `figs/`: hình từ `swarmbridge/experiments/uav_semcom/paperA_figures.py`, chỉ đọc JSON kết quả, không chạy lại thí
  nghiệm. Tạo lại bằng lệnh (WSL env):

  ```
  python paperA_figures.py <repo>/paper/paperA/figs
  ```
- Số liệu Bảng I và các hình lấy từ `runs/uav_semcom/paperA_r1/summary.json` (R1, seed 9800–9899).
- **Biên dịch:** máy này không có LaTeX. Tải `main.tex` và `figs/` lên Overleaf, rồi biên dịch bằng pdfLaTeX.
- **Kế hoạch và nguồn số liệu:**
  - [paperA-outline.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-outline.md);
  - kết quả R1: [paperA-r1-result.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-r1-result.md).
