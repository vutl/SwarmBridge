# Pilot ảnh thật — cổng A (06/11/2026)

Đã inference trên **216 ảnh dev, 30 nhóm filename**. Không train detector; chưa mở holdout.

| Phương án | AP50 | AP50:95 | Mean F1 | kbit/ảnh | xử lý p50 ms |
|---|---:|---:|---:|---:|---:|
| UAV nano → box codec | 0.3306 | 0.2220 | 0.4625 | 11.96 | 11.6 |
| UAV large → box codec | 0.4841 | 0.3525 | 0.5608 | 4.96 | 39.7 |
| Ảnh gốc → large tại trạm | 0.4841 | 0.3525 | 0.5608 | 1275.88 | 39.6 |
| JPEG Q10 → large tại trạm | 0.1795 | 0.1125 | 0.2514 | 264.74 | 40.0 |
| JPEG Q70 → large tại trạm | 0.4581 | 0.3332 | 0.5357 | 1006.92 | 39.7 |

Xử lý p50 là latency workstation GPU gồm preprocess/inference/postprocess; codec JPEG được đo riêng, không gồm radio.
Không dùng các số này làm latency hoặc năng lượng của phần cứng UAV.

| Model | CPU tham chiếu p50 / p95 ms (4 threads, 8 ảnh) |
|---|---:|
| yolov8n.pt | 85.0 / 88.6 |
| yolov8l.pt | 1073.9 / 1159.0 |

Box từ cùng model large so với ảnh gốc: ΔF1 = **+0.000000**, payload nhỏ hơn **257.5×**; so JPEG Q70 nhỏ hơn **203.2×**.

**Cổng A chưa qua.** Trong kịch bản UAV chạy được model large, gửi box gần như giữ nguyên kết quả nhưng ít bit hơn rất nhiều. Nếu chỉ cho UAV chạy nano, phải biện minh bằng compute/energy thực tế; chưa có ngân sách onboard độc lập nên chưa thể xác nhận nhu cầu offloading.

**Quyết định:** dừng mở correction, mobility hoặc RL cho bộ ba này. Không train tiếp để cứu kết quả. Kết quả không bác bỏ UAV mobility/SemCom nói chung; nó kiểm tra đúng tác vụ detection và codec đã chọn.

Các giới hạn:
- Ba lớp car/bus/truck, AP từ COCOeval với maxDets500 và spatial ignored-region filter theo toolkit VisDrone; không phải điểm leaderboard 10 lớp.
- Group là prefix filename, chưa xác minh độc lập hoàn toàn theo chuyến bay. Split đã ghi trước metric; 168 ảnh holdout chưa được chấm.
- Một cặp detector COCO đóng băng, domain ảnh UAV; kết quả chưa phải đánh giá detector tốt nhất hoặc generative recovery.
- Không có power measurement, channel experiment hoặc phần cứng UAV; chưa chứng minh trade-off energy hay deadline end-to-end.

Job giữ khóa chung mất 1.93 phút (gồm CPU benchmark và chấm), peak VRAM 0.38 GiB. Không có training.

Tái lập: `bash scripts/active_recovery/run_gate_a.sh` trong WSL; từng mode và benchmark CPU có resume.
Dataset/checkpoint: `assets.json`; split/config/prediction hash: thư mục này.
