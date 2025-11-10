# video2frames.py — Xuất ảnh từ video & Phân tích bong bóng

Tool đa năng để xử lý video: xuất ảnh từ video hoặc phân tích bong bóng (đếm số lượng, tính đường kính và vận tốc).

## Tính năng chính

### 📸 Chế độ xuất ảnh (mặc định)

- **Mặc định 30 ảnh/giây** (`--fps 30`) để đúng yêu cầu _1 giây -> 30 ảnh_.
- **Hai chế độ**:
  - `--native`: xuất **mọi frame gốc** (nhanh nhất). Dùng khi video là **30 FPS** chuẩn, mỗi giây sẽ ra 30 ảnh.
  - Mặc định (không `--native`): **lấy mẫu theo FPS mục tiêu** (30 ảnh/giây). Hữu ích nếu video **không đúng 30 FPS** nhưng bạn vẫn muốn 30 ảnh/giây ổn định.
- Giới hạn **khoảng thời gian** bằng `--start` / `--end` (giây).
- Chọn **định dạng** (`--ext jpg|png|webp|...`) và **chất lượng** (`--quality`).
- Cho phép **overwrite** file cũ, **prefix** tên file, **dry-run** để ước lượng.

### 🫧 Chế độ phân tích bong bóng (`--analyze-bubbles`)

- **Phát hiện bong bóng** trong video bằng HoughCircles hoặc Contour Detection
- **Đếm số lượng bong bóng** duy nhất trong toàn bộ video
- **Tính đường kính** (pixel) của từng quả bong bóng
- **Tính vận tốc** (pixel/giây) theo trục X, Y và độ lớn tổng
- **Tracking bong bóng** qua các frame để theo dõi chuyển động
- **Xuất kết quả** ra JSON (dữ liệu chi tiết) và CSV (thống kê tổng hợp)
- **Tùy chọn visualization**: tạo video với bong bóng được đánh dấu (vòng tròn, ID, vector vận tốc)

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Nếu bạn chưa có `ffmpeg` cũng không sao — tool dùng OpenCV, không yêu cầu ffmpeg.

## Cách dùng nhanh

### 📸 Xuất ảnh từ video

```bash
# 1) Xuất 30 ảnh mỗi giây (chuẩn yêu cầu)
python video2frames.py input.mp4 -o ./frames

# 2) Video chắc chắn là 30FPS? Dùng native cho nhanh (xuất mọi frame)
python video2frames.py input.mp4 -o ./frames --native

# 3) Chọn PNG & giới hạn đoạn 10s -> 25s
python video2frames.py input.mp4 -o ./frames --ext png --start 10 --end 25

# 4) Đặt prefix tên file và ghi đè nếu trùng
python video2frames.py input.mp4 -o ./frames --prefix clipA --overwrite
```

### 🫧 Phân tích bong bóng

```bash
# 1) Phân tích cơ bản (xuất JSON và CSV)
python video2frames.py input.mp4 --analyze-bubbles

# 2) Phân tích với video visualization
python video2frames.py input.mp4 --analyze-bubbles --visualize

# 3) Tùy chỉnh phương pháp phát hiện và kích thước
python video2frames.py input.mp4 --analyze-bubbles --detection-method contour --min-radius 10 --max-radius 50

# 4) Phân tích đoạn video cụ thể với ngưỡng thấp hơn
python video2frames.py input.mp4 --analyze-bubbles --start 5 --end 15 --threshold 30

# 5) Tùy chỉnh tracking (khoảng cách tối đa, số frame biến mất)
python video2frames.py input.mp4 --analyze-bubbles --max-tracking-distance 100 --max-disappeared 10
```

### Kết quả xuất ảnh

Sau khi chạy, thư mục `./frames` sẽ chứa file dạng:

```
<video_stem>_000000.jpg
<video_stem>_000001.jpg
...
```

### Kết quả phân tích bong bóng

Sau khi chạy với `--analyze-bubbles`, sẽ tạo các file:

- `<video_stem>_bubbles.json`: Dữ liệu chi tiết theo từng frame
- `<video_stem>_bubbles.csv`: Thống kê tổng hợp cho từng bong bóng
- `<video_stem>_analyzed.mp4`: Video visualization (nếu dùng `--visualize`)

**Cấu trúc JSON:**

- `bubble_statistics`: Thống kê cho từng bong bóng (ID, đường kính, vận tốc, lifetime, ...)
- `frame_by_frame_data`: Dữ liệu chi tiết theo từng frame

**Cấu trúc CSV:**

- Cột: `id`, `avg_diameter_pixels`, `avg_velocity_x_pixels_per_sec`, `avg_velocity_y_pixels_per_sec`, `avg_velocity_magnitude_pixels_per_sec`, `lifetime_seconds`, ...

## Tham số đầy đủ

### Tham số chung

- `input` (bắt buộc): đường dẫn file video (`.mp4`, `.mov`, ...).
- `--fps`: FPS mục tiêu (mặc định **30**).
- `--start`, `--end`: mốc thời gian theo giây để xử lý một đoạn.

### Tham số xuất ảnh

- `-o, --output`: thư mục lưu ảnh (mặc định: `./<tên_video>_frames`).
- `--native`: xuất mọi frame gốc (dùng khi video 30FPS và muốn mỗi giây 30 ảnh, nhanh hơn).
- `--ext`: `jpg|jpeg|png|webp|bmp|tif|tiff` (mặc định `jpg`).
- `--quality`: chất lượng ảnh (JPG/WEBP: 1–100; PNG tự quy đổi sang mức nén).
- `--overwrite`: ghi đè file nếu đã tồn tại.
- `--prefix`: tiền tố tên file (mặc định dùng tên video).
- `--dry-run`: chạy thử (không ghi file) để xem số lượng ảnh sẽ xuất.

### Tham số phân tích bong bóng

- `--analyze-bubbles`: Bật chế độ phân tích bong bóng (thay vì xuất ảnh).
- `--detection-method`: `hough` hoặc `contour` (mặc định `hough`).
  - `hough`: Sử dụng HoughCircles (tốt cho bong bóng tròn đều)
  - `contour`: Sử dụng contour detection (linh hoạt hơn, tốt cho bong bóng không hoàn toàn tròn)
- `--min-radius`: Bán kính nhỏ nhất (pixel, mặc định `5`).
- `--max-radius`: Bán kính lớn nhất (pixel, mặc định `100`).
- `--threshold`: Ngưỡng phát hiện cho HoughCircles (mặc định `50`, thấp hơn = phát hiện nhiều hơn).
- `--max-tracking-distance`: Khoảng cách tối đa để tracking bong bóng giữa các frame (pixel, mặc định `50.0`).
- `--max-disappeared`: Số frame tối đa bong bóng biến mất trước khi xóa khỏi tracking (mặc định `5`).
- `--visualize`: Tạo video visualization với bong bóng được đánh dấu.
- `--output-video`: Đường dẫn video output cho visualization (mặc định: `<tên_video>_analyzed.mp4`).
- `--output-json`: Đường dẫn file JSON (mặc định: `<tên_video>_bubbles.json`).
- `--output-csv`: Đường dẫn file CSV (mặc định: `<tên_video>_bubbles.csv`).

## Gợi ý sử dụng theo nhu cầu

### 📸 Xuất ảnh

- **Video 30FPS chuẩn**: dùng `--native` để xuất nhanh toàn bộ frames (mỗi giây sẽ được 30 ảnh).
- **Video không đúng 30FPS**: bỏ `--native` để tool tự **resample** về đúng `--fps 30`.
- **Muốn đúng số lượng ảnh = thời lượng(giây) × 30**: không dùng `--native` (resample).

### 🫧 Phân tích bong bóng

- **Bong bóng tròn đều, nền tương phản rõ**: dùng `--detection-method hough` (mặc định).
- **Bong bóng không hoàn toàn tròn hoặc nền phức tạp**: thử `--detection-method contour`.
- **Bong bóng nhỏ**: giảm `--min-radius` (vd: `--min-radius 3`).
- **Bong bóng lớn**: tăng `--max-radius` (vd: `--max-radius 200`).
- **Phát hiện quá nhiều false positive**: tăng `--threshold` (vd: `--threshold 70`).
- **Phát hiện thiếu bong bóng**: giảm `--threshold` (vd: `--threshold 30`).
- **Bong bóng di chuyển nhanh**: tăng `--max-tracking-distance` (vd: `--max-tracking-distance 100`).
- **Bong bóng biến mất tạm thời**: tăng `--max-disappeared` (vd: `--max-disappeared 10`).
- **Muốn xem kết quả trực quan**: dùng `--visualize` để tạo video với bong bóng được đánh dấu.

---

Chúc bạn xử lý video và phân tích bong bóng mượt mà! 🎈
