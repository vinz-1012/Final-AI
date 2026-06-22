# 📅 HỆ THỐNG XẾP THỜI KHÓA BIỂU TỰ ĐỘNG BẰNG AI (UNIVERSITY TIMETABLE)

Dự án này là một ứng dụng demo và so sánh hiệu năng giữa **12 thuật toán trí tuệ nhân tạo (AI)** phổ biến để giải quyết bài toán xếp thời khóa biểu tự động (University Timetabling Problem) — một bài toán NP-hard điển hình trong thực tế.

Hệ thống được phát triển bằng **Python** và tích hợp giao diện trực quan **Streamlit**, cho phép người dùng cấu hình tham số, chạy thử nghiệm, so sánh hiệu năng và lập kế hoạch dự phòng đối kháng.

---

## 🛠️ Công nghệ sử dụng
* **Ngôn ngữ**: Python 3.x
* **Thư viện chính**: 
  * `Streamlit` (Xây dựng UI tương tác)
  * `Matplotlib` & `Pandas` (Vẽ biểu đồ và quản lý dữ liệu hiệu năng)
  * `NetworkX` (Xây dựng và phân tích đồ thị xung đột)

---

## 📂 Cấu trúc dự án

```text
ProjectAI/
├── config/
│   └── settings.py              # Các cấu hình mặc định (Đường dẫn dữ liệu, kích thước cửa sổ...)
├── data/
│   ├── sample_data/             # Các bộ dữ liệu mẫu JSON
│   │   ├── small_timetable.json  # Dữ liệu nhỏ (5 lớp HP)
│   │   ├── medium_timetable.json # Dữ liệu vừa (8 lớp HP)
│   │   └── timetable_data.json   # Dữ liệu chuẩn (40 lớp HP)
│   └── timetable_loader.py      # Bộ phân tích cú pháp tải JSON chuyển sang cấu trúc đồ thị bài toán
├── src/
│   ├── algorithms/              # Triển khai các thuật toán AI xếp lịch
│   │   ├── uninformed/          # Tìm kiếm mù (BFS, DFS)
│   │   ├── informed/            # Tìm kiếm thông tin có Heuristic (Greedy Best-First, A*)
│   │   ├── local/               # Tìm kiếm cục bộ (Hill Climbing, Simulated Annealing)
│   │   ├── csp/                 # Ràng buộc thỏa mãn (CSP Backtracking, CSP Min-Conflicts)
│   │   ├── complex/             # Tìm kiếm phức tạp/Online (AND-OR Search, LRTA*)
│   │   └── adversarial/         # Phân tích đối kháng đối phó sự cố (Minimax, Alpha-Beta Pruning)
│   ├── core/                    # Các thực thể dữ liệu cốt lõi (Timetable, Section, Lecturer, Room, Period...)
│   └── visualization/           # Render kết quả thời khóa biểu dưới dạng HTML/CSS tương tác đẹp mắt
├── streamlit_app.py             # Giao diện chính của ứng dụng Streamlit
├── main.py                      # File chạy kiểm thử CLI
└── benchmark.py                 # File so sánh hiệu năng trực tiếp của các thuật toán
```

---

## 💡 Cách hoạt động của Hệ thống

Bài toán xếp lịch được ánh xạ sang **Bài toán Tìm kiếm Không gian trạng thái** hoặc **Bài toán Thỏa mãn Ràng buộc (CSP)** trên đồ thị xung đột:
* **Nút (Nodes)**: Lớp học phần cần được gán ca học và phòng học.
* **Cạnh (Edges)**: Đại diện cho sự xung đột giữa hai lớp học phần (ví dụ: chung giảng viên).

### 1. Ràng buộc cứng (Hard Constraints) - Bắt buộc phải thỏa mãn:
1. **Không trùng lịch giảng viên**: Giảng viên không được phép dạy 2 lớp khác nhau trong cùng một ca.
2. **Không trùng phòng học**: Một phòng học không được chứa 2 lớp khác nhau trong cùng một ca.
3. **Phù hợp sức chứa**: Sức chứa của phòng học phải lớn hơn hoặc bằng sĩ số sinh viên của lớp học phần.
4. **Phù hợp loại phòng**: Lớp học thực hành phải xếp vào phòng máy (lab), lớp lý thuyết xếp vào phòng học thường (lecture/seminar).
5. **Tránh ca bận**: Không xếp lịch dạy của giảng viên vào những ca mà giảng viên đã đăng ký bận.

### 2. Ràng buộc mềm (Soft Constraints) - Khuyến khích tối ưu:
1. Hạn chế tối đa số ca dạy trong một ngày của giảng viên (tránh quá tải).
2. Tối ưu phân bố phòng học theo tòa nhà.
3. Phân bố đều lịch học trong tuần cho sinh viên.

---

## 🧠 Tóm tắt 6 Nhóm thuật toán AI được tích hợp

| Nhóm Thuật toán | Các thuật toán đại diện | Cách giải quyết & Đặc điểm |
| :--- | :--- | :--- |
| **Uninformed Search** | `BFS`, `DFS` | Tìm kiếm mù có hệ thống. Phù hợp dữ liệu cực nhỏ ($\le 5$ lớp). Dễ bị bùng nổ tổ hợp trên dữ liệu lớn. |
| **Informed Search** | `Greedy`, `A*` | Sử dụng thông tin hàm heuristic đánh giá khoảng cách/xung đột để hướng tìm kiếm về đích nhanh hơn. |
| **Local Search** | `Hill Climbing`, `Simulated Annealing` | Khởi tạo TKB ngẫu nhiên, sau đó liên tục thực hiện các điều chỉnh nhỏ (leo núi) hoặc chấp nhận bước đi tệ hơn với xác suất giảm dần (luyện kim) để thoát khỏi cực trị địa phương. |
| **Constraint Satisfaction (CSP)** | `CSP Backtracking`, `Min-Conflicts` | Xem các vị trí xếp là các biến ràng buộc. **Min-Conflicts** là thuật toán thực tế vượt trội nhất, có thể giải quyết lịch 40 lớp hoặc hàng trăm lớp trong tích tắc bằng cách liên tục giảm số xung đột hiện tại. |
| **Complex Search** | `AND-OR Search`, `LRTA*` | Lập kế hoạch dự phòng (AND-OR) khi có sự cố phát sinh và học hỏi môi trường thời gian thực (LRTA*). |
| **Adversarial Search** | `Minimax`, `Alpha-Beta` | Xem bài toán dưới dạng đối kháng giữa Người xếp lịch (MAX) muốn tối ưu hóa lịch và Môi trường/Sự cố đột xuất (MIN) muốn phá hỏng lịch để chọn ra vị trí robust (bền bỉ) nhất. |

---

## 🚀 Hướng dẫn chạy chương trình

### Cách 1: Chạy giao diện Web Streamlit (Khuyên dùng)
Giao diện Web cung cấp đầy đủ bảng tương tác, bản đồ TKB tuần trực quan và tab so sánh benchmark:
```bash
streamlit run streamlit_app.py
```

### Cách 2: Chạy kiểm thử dòng lệnh (CLI)
Chạy kịch bản kiểm thử mặc định từ dòng lệnh:
```bash
python main.py
```

### Cách 3: Chạy chương trình Benchmark
So sánh hiệu năng đo lường trực tiếp tốc độ chạy và chất lượng TKB giữa các thuật toán:
```bash
python benchmark.py
```

---

## 🗓️ Các chế độ lọc hiển thị Thời khóa biểu
Sau khi chạy thuật toán, bạn có thể lọc hiển thị TKB theo:
1. **Tất cả**: Xem toàn cảnh tất cả các phòng và ca học hoạt động trong tuần.
2. **Phòng học**: Xem chi tiết lịch sử dụng của một phòng học cụ thể để tránh lãng phí hoặc kiểm tra chồng chéo.
3. **Giảng viên**: Xem lịch đi dạy cá nhân của từng giảng viên trong tuần.
4. **Học phần**: Theo dõi lịch phân bổ của tất cả các lớp thuộc cùng một môn học.
