# Paper A: bản thảo cho ICT Express (viết lại 25/11 trên dữ liệu full-data)

**Trạng thái:** bản nháp 4 trang, chờ ChatGPT phản biện lần cuối. **Chưa nộp.**

**Bài:**
- **Tiêu đề:** "Uncertainty-aware mode selection for multi-UAV semantic image offloading".
- **File:** `main.tex` → `main.pdf`, template ICT Express (`elsarticle`, 3p, hai cột; lấy từ
  `paper/ICT Express Submission Template_191231.zip`).
- **Khuôn khổ:** 4 trang tính cả tài liệu tham khảo; abstract 100 từ; 5 từ khóa; 2 hình + 1 bảng; 13 tài liệu tham
  khảo.
- **Chỗ còn thiếu** (đánh dấu `\todo{}`, in đỏ): tác giả, đơn vị, email, tài trợ.
- **Bản method letter cũ** (ngày 23/11, tiêu đề "ACK-calibrated selective MPC") nằm trong lịch sử git ở commit
  `1225e287`.

**Hình và bảng** (chỉ đọc kết quả đã lưu, không chạy lại thí nghiệm):
- `swarmbridge/experiments/uav_semcom/paperA_fd_numbers.py` ghi `runs/uav_semcom/paperA_fd/paper/paper_numbers.json`
  và `table_fd.tex` (Bảng 1).
- `swarmbridge/experiments/uav_semcom/paperA_fd_figures.py` vẽ hai hình:
  - `figs/fig_fd_system.pdf` (Hình 1: hệ thống và bộ điều khiển);
  - `figs/fig_fd_results.pdf` (Hình 2: ablation, return so với recall, phân rã return).
- Các file `fig_*`, `table_*.tex` khác là của các bản cũ, giữ lại để đối chiếu.

**Nguồn của từng con số:** [CLAIMS.md](CLAIMS.md). Mỗi số được đánh dấu là kiểm định đăng ký trước, hỗ trợ, hay hậu
kiểm.

**Báo cáo quá trình viết và các vấn đề còn mở:**
[paperA-fd-writing-report.md](../../reports/phases/2026-11-23_paperA-fulldata/paperA-fd-writing-report.md).

**Tạo lại** (môi trường WSL, `semcom-a`; chạy hai lệnh python trong `swarmbridge/experiments/uav_semcom`):

```
python paperA_fd_numbers.py
python paperA_fd_figures.py <repo>/paper/paperA/figs
bash paper/paperA/build.sh        # main.pdf; kiểm tra <= 4 trang
```

**Kết quả nền:**
- [paperA-fd-consolidated.md](../../reports/phases/2026-11-23_paperA-fulldata/paperA-fd-consolidated.md);
- [paperA-fd-posthoc-result.md](../../reports/phases/2026-11-23_paperA-fulldata/paperA-fd-posthoc-result.md).
