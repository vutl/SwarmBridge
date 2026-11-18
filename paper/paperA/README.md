# Paper A: bản thảo cho ICT Express

- **`main.tex`:** bài dạng letter theo template ICT Express (`elsarticle`, 3p, hai cột), lấy từ
  `paper/ICT Express Submission Template_191231.zip`.
  - Giới hạn của tạp chí: khuyến nghị 4 trang tính cả tài liệu tham khảo (100 USD mỗi trang vượt); hình + bảng ≤ 4;
    abstract 75–100 từ; 3–5 từ khóa xếp theo bảng chữ cái.
  - Bản này: abstract 96 từ, 1 bảng + 3 hình, thân bài khoảng 1.200 từ.
  - Chỗ còn phải kiểm được đánh dấu `\todo{}` (in đỏ): tác giả, thông tin tham khảo chưa chắc, mức phạt khi trễ
    deadline, định nghĩa cổng của MPC.
- **Kèm theo** (lấy nguyên từ template): `elsarticle.cls`, `elsarticle-num.bst`, `numcompress.sty`.
- **`figs/`:** hình từ `swarmbridge/experiments/uav_semcom/paperA_figures.py`, chỉ đọc JSON kết quả. Tạo lại bằng lệnh
  (WSL env):

  ```
  python paperA_figures.py <repo>/paper/paperA/figs
  ```
- **Số liệu:** `runs/uav_semcom/paperA_r1/summary.json` (R1, seed 9800–9899).
- **Biên dịch:** máy này không có LaTeX. Tải cả thư mục `paperA/` lên Overleaf và biên dịch bằng pdfLaTeX. Kiểm tra số
  trang: phải ≤ 4.
- **Kế hoạch và kết quả:**
  - [paperA-outline.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-outline.md);
  - [paperA-r1-result.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-r1-result.md).
