# Paper A: bản thảo cho ICT Express (method letter, viết lại 23/11)

> **Cảnh báo (25/11): bản thảo này đã LỖI THỜI. Không nộp.**
> - Đánh giá full-data đăng ký trước ([consolidated](../../reports/phases/2026-11-23_paperA-fulldata/paperA-fd-consolidated.md))
>   cho thấy: so với MPC danh định có cùng cổng + dải, MPC hiệu chỉnh không hơn ở cấu hình chuẩn (−0,10 n.s.). Vì vậy
>   claim "12/12" trong tiêu đề và abstract đến từ việc so với planner không bảo vệ.
> - Cổng và H = 4 không cần thiết.
> - Thời gian chạy tuần tự vượt 50 ms.
> - Chờ phản biện chọn C / B / D và venue rồi mới viết lại ([CHATGPT_HANDOFF.md](../../reports/CHATGPT_HANDOFF.md)).

- **`main.tex`:** bài dạng letter theo template ICT Express (`elsarticle`, 3p, hai cột), lấy từ
  `paper/ICT Express Submission Template_191231.zip`.
  - Giới hạn của tạp chí: khuyến nghị 4 trang tính cả tài liệu tham khảo (100 USD mỗi trang vượt); hình + bảng ≤ 4;
    abstract 75–100 từ; 3–5 từ khóa xếp theo bảng chữ cái.
  - Bản này: **6 trang** (hình kiểu IEEE/MATLAB, kích thước đầy đủ), abstract khoảng 95 từ, 4 hình, không có bảng, 15
    tài liệu tham khảo. Phải chọn: nộp ICT Express 4 trang (bớt hình) hay nhắm venue dài hơn.
  - Chỗ còn thiếu được đánh dấu `\todo{}` (in đỏ): tác giả, đơn vị, tài trợ.
- **Kèm theo** (lấy nguyên từ template): `elsarticle.cls`, `elsarticle-num.bst`, `numcompress.sty`.
- **Hình và bảng:**
  - Hình 1 `figs/fig_method.pdf` (sơ đồ phương pháp): tạo bằng `paperA_letter_figures.py`.
  - Hình 2–4: kiểu IEEE/MATLAB (khung kín, lưới, Arial, marker), tạo bằng
    `swarmbridge/experiments/uav_semcom/paperA_ieee_figures.py`:
    - Hình 2 `figs/fig_sweeps.pdf`: 4 đồ thị đường theo băng thông, số UAV, số kênh và độ trễ báo cáo.
    - Hình 3 `figs/fig_phy.pdf`: đồ thị cột cho các biến thể PHY, cùng chênh recall.
    - Hình 4 `figs/fig_rl.pdf`: learning curve, cùng cột B1/B2 cho return và recall.
  - Các phiên bản hình cũ hơn (`fig_lines`, `fig_e7bars`, `fig_robust`, `fig_e7`, `fig_recall`) vẫn được
    `paperA_letter_figures.py` tạo ra, nhưng không dùng trong bài.
  - Số liệu: `paperA_letter_numbers.py` chỉ đọc file kết quả, không chạy lại thí nghiệm. Nó ghi
    `runs/uav_semcom/paperA_r2/letter_numbers.json`, cùng hai bảng `table_robust.tex` và `table_e7.tex`. Hai bảng này
    có cùng số liệu với Hình 2–3 nhưng không đưa vào PDF.
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
