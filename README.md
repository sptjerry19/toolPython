# video2frames.py — Xuất ảnh từ video (30 ảnh/giây)

## Tính năng chính
- **Mặc định 30 ảnh/giây** (`--fps 30`) để đúng yêu cầu *1 giây -> 30 ảnh*.
- **Hai chế độ**:
  - `--native`: xuất **mọi frame gốc** (nhanh nhất). Dùng khi video là **30 FPS** chuẩn, mỗi giây sẽ ra 30 ảnh.
  - Mặc định (không `--native`): **lấy mẫu theo FPS mục tiêu** (30 ảnh/giây). Hữu ích nếu video **không đúng 30 FPS** nhưng bạn vẫn muốn 30 ảnh/giây ổn định.
- Giới hạn **khoảng thời gian** bằng `--start` / `--end` (giây).
- Chọn **định dạng** (`--ext jpg|png|webp|...`) và **chất lượng** (`--quality`).
- Cho phép **overwrite** file cũ, **prefix** tên file, **dry-run** để ước lượng.

## Cài đặt
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Nếu bạn chưa có `ffmpeg` cũng không sao — tool dùng OpenCV, không yêu cầu ffmpeg.

## Cách dùng nhanh
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

Sau khi chạy, thư mục `./frames` sẽ chứa file dạng:
```
<video_stem>_000000.jpg
<video_stem>_000001.jpg
...
```

## Tham số đầy đủ
- `input` (bắt buộc): đường dẫn file video (`.mp4`, `.mov`, ...).
- `-o, --output`: thư mục lưu ảnh (mặc định: `./<tên_video>_frames`).
- `--fps`: FPS mục tiêu (mặc định **30**).
- `--native`: xuất mọi frame gốc (dùng khi video 30FPS và muốn mỗi giây 30 ảnh, nhanh hơn).
- `--ext`: `jpg|jpeg|png|webp|bmp|tif|tiff` (mặc định `jpg`).
- `--quality`: chất lượng ảnh (JPG/WEBP: 1–100; PNG tự quy đổi sang mức nén).
- `--overwrite`: ghi đè file nếu đã tồn tại.
- `--start`, `--end`: mốc thời gian theo giây để xuất một đoạn.
- `--prefix`: tiền tố tên file (mặc định dùng tên video).
- `--dry-run`: chạy thử (không ghi file) để xem số lượng ảnh sẽ xuất.

## Gợi ý sử dụng theo nhu cầu
- **Video 30FPS chuẩn**: dùng `--native` để xuất nhanh toàn bộ frames (mỗi giây sẽ được 30 ảnh).
- **Video không đúng 30FPS**: bỏ `--native` để tool tự **resample** về đúng `--fps 30`.
- **Muốn đúng số lượng ảnh = thời lượng(giây) × 30**: không dùng `--native` (resample).

---

Chúc bạn xử lý video mượt mà!