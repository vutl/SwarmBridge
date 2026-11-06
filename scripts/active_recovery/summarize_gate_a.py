"""Render measured Gate-A evidence, keeping deployment feasibility conditional."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/active_recovery/gate_a'


def main():
    r=json.loads((OUT/'summary.json').read_text())
    lines=['# Pilot ảnh thật — cổng A (06/11/2026)', '',
           f"Đã inference trên **{r['dev_images']} ảnh dev, {r['dev_groups']} nhóm filename**. Không train detector; chưa mở holdout.", '',
           '| Phương án | AP50 | AP50:95 | Mean F1 | kbit/ảnh | xử lý p50 ms |',
           '|---|---:|---:|---:|---:|---:|']
    names={'nano_boxes':'UAV nano → box codec','large_boxes':'UAV large → box codec','large_original':'Ảnh gốc → large tại trạm','large_jpeg10':'JPEG Q10 → large tại trạm','large_jpeg70':'JPEG Q70 → large tại trạm'}
    for mode,s in r['scores'].items():
        lines.append('| %s | %.4f | %.4f | %.4f | %.2f | %.1f |'%(names[mode],s['AP50'],s['AP50_95'],s['mean_image_F1'],s['payload_bits_mean']/1000,s['processing_ms_p50']))
    lines+=['','Xử lý p50 là latency workstation GPU gồm preprocess/inference/postprocess; codec JPEG được đo riêng, không gồm radio.',
            'Không dùng các số này làm latency hoặc năng lượng của phần cứng UAV.','',
            '| Model | CPU tham chiếu p50 / p95 ms (4 threads, 8 ảnh) |','|---|---:|']
    for model,c in r['cpu_reference'].items():lines.append('| %s | %.1f / %.1f |'%(model,c['p50_ms'],c['p95_ms']))
    b=r['scores']['large_boxes'];full=r['scores']['large_original'];q=r['scores']['large_jpeg70']
    lines += ['',f"Box từ cùng model large so với ảnh gốc: ΔF1 = **{b['mean_image_F1']-full['mean_image_F1']:+.6f}**, "
              f"payload nhỏ hơn **{full['payload_bits_mean']/b['payload_bits_mean']:.1f}×**; so JPEG Q70 nhỏ hơn **{q['payload_bits_mean']/b['payload_bits_mean']:.1f}×**.", '',
              '**Cổng A: chưa kết luận về triển khai.** Khi bỏ chi phí tính toán ra khỏi so sánh, gửi box từ cùng model giữ chất lượng và ít bit hơn. '
              'Điều này không chứng minh offloading ảnh vô ích: CPU tham chiếu chạy large chậm hơn GPU tại trạm. Cần so đủ compute, truyền và deadline.', '',
              '**Quyết định sau đính chính:** chưa mở correction, mobility hoặc RL. Giữ offloading là ứng viên có điều kiện; '
              'không coi bộ ba đã thất bại. Phân tích không inference mới nằm ở [offloading/summary.md](../offloading/summary.md). '
              'Năng lượng và compute onboard thực tế vẫn chưa được xác nhận.', '',
              'Các giới hạn:',
              '- Ba lớp car/bus/truck, AP từ COCOeval với maxDets500 và spatial ignored-region filter theo toolkit VisDrone; không phải điểm leaderboard 10 lớp.',
              '- Group là prefix filename, chưa xác minh độc lập hoàn toàn theo chuyến bay. Split đã ghi trước metric; 168 ảnh holdout chưa được chấm.',
              '- Một cặp detector COCO đóng băng, domain ảnh UAV; kết quả chưa phải đánh giá detector tốt nhất hoặc generative recovery.',
              '- Không có power measurement, channel experiment hoặc phần cứng UAV; chưa chứng minh trade-off energy hay deadline end-to-end.', '',
              f"Job giữ khóa chung mất {r['resources']['job_wall_seconds']/60:.2f} phút (gồm CPU benchmark và chấm), "
              f"peak VRAM {r['resources']['peak_vram_bytes']/1024**3:.2f} GiB. Không có training.", '',
              'Tái lập: `bash scripts/active_recovery/run_gate_a.sh` trong WSL; từng mode và benchmark CPU có resume.',
              'Dataset/checkpoint: `assets.json`; split/config/prediction hash: thư mục này.']
    text='\n'.join(lines)+'\n'
    (OUT/'summary.md').write_text(text,encoding='utf-8')
    report=ROOT/'reports/wireless-abc.md';old=report.read_text(encoding='utf-8')
    marker='\n## Pilot ảnh thật — cổng A đã chạy (06/11/2026)\n'
    if marker in old:
        before,section=old.split(marker,1)
        end=section.find('\n## ')
        after=section[end:] if end>=0 else ''
        old=before
    else:after=''
    report_text=text.replace('# Pilot ảnh thật — cổng A (06/11/2026)\n','',1).replace('(../offloading/summary.md)','(../runs/active_recovery/offloading/summary.md)')
    report.write_text(old+marker+'\n'+report_text+after,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
