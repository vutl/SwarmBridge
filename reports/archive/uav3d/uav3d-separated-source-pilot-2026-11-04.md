# UAV3D mini: pilot tách feature của ego và đồng đội

## Câu hỏi

Trong thí nghiệm trước, feature hiện tại của ego và feature có thể bị trễ của các UAV khác được lấy trung bình trước khi đưa vào decoder. Pilot này thử giữ hai nguồn ở hai nhóm kênh riêng và hỏi liệu đưa thêm tuổi message theo số frame có giúp decoder dùng message trễ tốt hơn không.

Hai nhánh dùng cùng encoder đóng băng, checkpoint, dữ liệu, decoder, số epoch, phân phối trễ/mất gói và ba seed 0–2. Cả hai nhận feature ego hiện tại và feature trung bình của các helper còn lại; một nhánh nhận thêm một kênh tuổi, nhánh còn lại không. Khi không còn helper, feature helper bằng 0. Trọng số của các kênh mới được khởi tạo bằng 0. Đây là phép kiểm tra vai trò của tuổi *trong kiến trúc tách nguồn*, không phải phép so chỉ khác một yếu tố với kiến trúc trộn trung bình cũ.

Code: `swarmbridge/experiments/uav3d_comm/train_separate.py`. Lịch chạy: `scripts/uav3d_separate_pilot.sh`. Log: `runs/uav3d_separate_pilot.log`. Checkpoint và kết quả: `runs/uav3d_separate_pilot/`. Script dùng khóa GPU chung; khi chạy lại sẽ bỏ qua từng model đã hoàn thành. Sau sáu model, script tự chạy đánh giá detection và ghi `det_heldout.json`.

## Giới hạn diễn giải

- Chỉ có ba scene held-out của mini; không khẳng định tổng quát hoặc novelty từ kết quả này.
- Trễ và rơi gói là mô phỏng; truyền feature chưa được nén hoặc đo theo cùng bitrate với box.
- Dùng số frame trễ, chưa quy đổi ra giây: timestamp metadata không khớp trực tiếp nhận định 0,2 giây/frame trong report trước; cần kiểm tra ý nghĩa timestamp và thời gian mô phỏng trước khi quy đổi.
- Cùng một tuổi được áp cho tất cả helper trong mỗi mẫu; chưa kiểm tra trễ khác nhau theo từng UAV. Trên đầu scene, số frame trễ có thể nhỏ hơn mức yêu cầu.
- Không gọi đây là triển khai CERF chính thức hoặc là thử nghiệm SB.
