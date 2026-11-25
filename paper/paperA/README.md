# Paper A: bản thảo cho ICT Express (viết lại 25/11 trên dữ liệu full-data)

**Trạng thái (26/11):** bản sửa cuối, **6 trang**, chờ phản biện trước khi nộp và giáo sư xem. **Chưa nộp.**
- **Tiêu đề mới:** "Predictive mode selection under wireless rate uncertainty for multi-UAV semantic image
  offloading".
- **Hình và bảng:**
  - Hình 1 vector: `figs/fig_fd_system.pdf`, nguồn là hàm `system2` trong `paperA_fd_figures.py`.
  - Hình 2: `figs/fig_fd_ablation.pdf` (một cột).
  - Hình 3: `figs/fig_fd_tradeoff.pdf`.
  - Bảng I nằm trong `main.tex`; Bảng II là `table_fd.tex`.
- **Gói tái lập:** `make_snapshot.sh` (từ tag `paperA-v1.0`), mô tả trong `SNAPSHOT_README.md`. DOI còn để TODO,
  chờ tác giả tải lên.
- **Bản 6 trang trước** ở commit `0a27d406`.

(Các ghi chú bên dưới thuộc các bản trước.)
- **Hình và bảng:** 3 hình (`fig_fd_system`, `fig_fd_mechanism`, `fig_fd_tradeoff`) + 2 bảng (Bảng I tham số
  trong `main.tex`; Bảng II là `table_fd.tex`).
- **Bản 4 trang** ở commit `484953df`.
- **Lưu ý:** template 2019 trong repo ghi khuyến nghị 4 trang, tối đa 4 hình + bảng, và 100 USD mỗi trang vượt quá
  4. Cần kiểm lại với Guide for Authors hiện hành trước khi nộp.

Các dòng dưới đây mô tả bản 4 trang; phần nào khác với bản 6 trang thì theo ghi chú ở trên.

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
