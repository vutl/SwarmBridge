# Tình trạng tải tài liệu ngày 30/10/2026

- Reading map: hiện có 24/26 bài trong `pdf/`: 21 bài đã tải ban đầu và 3 PDF IEEE (#14, #22, #23) đã được bổ sung tại máy sau đó. Các bản arXiv là bản preprint của tác giả; bài #24 là bản accepted manuscript do Springer cung cấp.
- Bốn bài gắn với các repo nhưng không nằm riêng trong reading map: paper của SemanticRL, UavNetSim và UAV-DDPG nằm trong `pdf/`; paper UAV Path Optimization nằm sẵn trong thư mục mã nguồn của repo. Bài DeepJSCC và DeepSC đã được tính trong reading map.
- Sáu thư mục GitHub nằm trong `pdf/GithubModels/`. Đây là bản giải nén source archive, không có lịch sử Git hay remote để `git pull`.
- Đã tải và giải nén checkpoint DeepJSCC từ GitHub Releases vào `pdf/GithubModels/DeepJSCC/runs/` theo README. Chưa chạy thử code hay kiểm tra chất lượng checkpoint.
- Đính chính SemanticRL: lần đầu script đã chọn nhánh `main`, chỉ có README cũ; nhánh mặc định thực tế là `master` và có code. Đã tải lại 110 file của nhánh `master` vào `pdf/GithubModels/SemanticRL/`. Năm repo khác đã được đối chiếu và tải đúng nhánh mặc định. Script tải nay kiểm tra nhánh mặc định qua GitHub API.
- Paper nằm sẵn trong repo UAV Path Optimization có tên **DDQN-Enabled UAV Path Planning in Non-Terrestrial Networks Under ICAS Constraints**. Tên thực tế khác tên paper trong đoạn giới thiệu được gửi. Giữ file trong repo để tránh tạo một bản trùng.

## Những paper chưa tải được

Tác vụ tải chạy trong một tiến trình riêng, không tự chia sẻ phiên đăng nhập trình duyệt của bạn. Các kết quả HTTP dưới đây chỉ phản ánh lần tải này; chúng không chứng minh bạn không có quyền qua trường. Bạn có thể mở những link sau trên trình duyệt đã đăng nhập và lưu PDF vào `pdf/`:

Ngày 30/10: bạn cho biết không tải được hai bài ScienceDirect (#13 và #15). Đã tìm lại theo tiêu đề, DOI, trang tác giả và OpenAlex; chưa xác minh được bản toàn văn công khai. Có thể hỏi thư viện trường về quyền truy cập/liên thư viện hoặc chủ động xin manuscript từ tác giả. Bản ghi của hai bài: [Xu và cộng sự](https://doi.org/10.1016/j.phycom.2024.102555), [Xie và cộng sự](https://pure.qub.ac.uk/en/publications/resource-allocation-for-uav-assisted-anti-jamming-semantic-d2d-ne/).

| Số | Paper | Link nhà xuất bản | Kết quả lần thử |
|---|---|---|---|
| 13 | Joint optimization trajectory and resource allocation for UAV-assisted semantic communications | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1874490724002738) | PDF trả HTTP 403 |
| 15 | Resource allocation for UAV-assisted anti-jamming semantic D2D networks: A graph reinforcement learning approach | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S138912862500430X) | PDF trả HTTP 403 |

Danh mục đầy đủ các bài và repo, link nguồn, số trang và SHA-256 của các PDF mới tải nằm trong [danh-sach-tai-lieu-2026-10-30.json](../../metadata/danh-sach-tai-lieu-2026-10-30.json). Với các PDF mới tải, script kiểm tra định dạng, đọc trang và đối chiếu từ khóa tiêu đề. Kiểm tra này không thay thế việc đọc nội dung.

File [`.gitignore`](../../../.gitignore) được tạo để Git bỏ qua dữ liệu, PDF/repo tham khảo, kết quả chạy, checkpoint, môi trường Python, cache và cấu hình máy. [`.gitignores`](../../../.gitignores) cũng được điền theo yêu cầu nhưng Git không tự đọc file tên này. Máy hiện không nhận lệnh `git` trong PATH, nên chưa kiểm tra bằng `git check-ignore`.
