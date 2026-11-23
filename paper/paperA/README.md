# Paper A: bản thảo cho ICT Express (method letter, viết lại 23/11)

- **`main.tex`:** bài dạng letter theo template ICT Express (`elsarticle`, 3p, hai cột), lấy từ
  `paper/ICT Express Submission Template_191231.zip`.
  - Giới hạn của tạp chí: khuyến nghị 4 trang tính cả tài liệu tham khảo (100 USD mỗi trang vượt); hình + bảng ≤ 4;
    abstract 75–100 từ; 3–5 từ khóa xếp theo bảng chữ cái.
  - Bản này: 4 trang, abstract khoảng 95 từ, 1 hình + 2 bảng, 15 tài liệu tham khảo.
  - Chỗ còn thiếu được đánh dấu `\todo{}` (in đỏ): tác giả, đơn vị, tài trợ.
- **Kèm theo** (lấy nguyên từ template): `elsarticle.cls`, `elsarticle-num.bst`, `numcompress.sty`.
- **Hình và bảng:**
  - Hình 1 `figs/fig_method.pdf`: tạo bằng `swarmbridge/experiments/uav_semcom/paperA_letter_figures.py`. Script
    này cũng tạo `fig_recall.pdf` (return so với recall), hiện không dùng trong bài vì giới hạn 4 trang.
  - Bảng 1 `table_robust.tex` và Bảng 2 `table_e7.tex`: tạo bằng `paperA_letter_numbers.py`. Script chỉ đọc file kết
    quả, không chạy lại thí nghiệm. Nó ghi `runs/uav_semcom/paperA_r2/letter_numbers.json`.
  - Các hình và bảng cũ (`fig_misalign`, `fig_r1_*`, `fig_system`, `table_main.tex`) thuộc bản 20/11, giữ lại để đối
    chiếu.
- **Nguồn của từng con số:** [CLAIMS.md](CLAIMS.md).
- **Tạo lại** (môi trường WSL):

  ```
  python paperA_letter_numbers.py
  python paperA_letter_figures.py <repo>/paper/paperA/figs
  bash paper/paperA/build.sh
  ```

  Lệnh cuối tạo `paper/paperA/main.pdf`. Phải kiểm tra số trang ≤ 4.
- **Kết quả nền:**
  - [paperA-r2-robustness-result.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-r2-robustness-result.md);
  - [paperA-e4-result.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-e4-result.md) (E4–E6);
  - [paperA-e7-result.md](../../reports/phases/2026-11-18_paperA-writeup/paperA-e7-result.md).
