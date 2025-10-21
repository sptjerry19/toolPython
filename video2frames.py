#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2frames.py
----------------
Tool xuất ảnh từ video theo FPS mục tiêu (mặc định 30 ảnh/giây).
- Hỗ trợ hai chế độ:
  1) --native: Xuất mọi frame gốc của video (nhanh). Thích hợp khi video là 30 FPS chuẩn và bạn muốn 1s -> 30 ảnh.
  2) Mặc định (không --native): Lấy mẫu theo --fps (mặc định 30). Hữu ích khi video không đúng 30 FPS nhưng bạn vẫn muốn 30 ảnh/giây ổn định.

Cài đặt:
    pip install -r requirements.txt

Ví dụ:
    python video2frames.py input.mp4 -o ./frames
    # Ép 30 ảnh/giây, đặt tên theo index, chất lượng JPEG 95

    python video2frames.py input.mp4 -o ./frames --native
    # Xuất tất cả frame gốc (nếu video 30FPS thì 1s -> 30 ảnh)

    python video2frames.py input.mp4 -o ./frames --fps 30 --ext png
    # Lấy mẫu 30 ảnh/giây, xuất PNG

    python video2frames.py input.mp4 -o ./frames --start 10 --end 25
    # Chỉ xuất đoạn [10s, 25s]

Tác giả: ChatGPT
"""

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


def human_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    if h > 0:
        return f"{h}h{m:02d}m{s:05.2f}s"
    if m > 0:
        return f"{m}m{s:05.2f}s"
    return f"{s:.2f}s"


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def imwrite_ext(path: Path, image: np.ndarray, ext: str, quality: int) -> bool:
    ext = ext.lower().lstrip(".")
    if ext == "jpg" or ext == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))]
        return cv2.imwrite(str(path), image, params)
    elif ext == "png":
        # 0 = none (fast), 9 = max (small)
        level = int(np.clip(round((100 - quality) / 10), 0, 9))
        params = [cv2.IMWRITE_PNG_COMPRESSION, level]
        return cv2.imwrite(str(path), image, params)
    elif ext == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(np.clip(quality, 1, 100))]
        return cv2.imwrite(str(path), image, params)
    else:
        # bmp, tif, ...
        return cv2.imwrite(str(path), image)


def build_src_indices(frame_count: int, fps_src: float, target_fps: float,
                      start_idx: int, end_idx: int) -> List[int]:
    """
    Tạo danh sách index frame nguồn cần xuất để đạt target_fps trong đoạn [start_idx, end_idx].
    Sử dụng bước float để đảm bảo phủ đều theo thời gian.
    """
    if end_idx < 0 or end_idx >= frame_count:
        end_idx = frame_count - 1

    if start_idx < 0:
        start_idx = 0

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if end_idx < start_idx:
        return []

    # Thời lượng đoạn chọn
    duration = (end_idx - start_idx + 1) / max(fps_src, 1e-6)
    total_to_export = int(round(duration * target_fps))

    if total_to_export <= 0:
        return []

    # i chạy từ 0 -> total_to_export-1
    # Map sang index nguồn trong [start_idx, end_idx] theo t = i/target_fps
    # src = start_idx + t*fps_src
    step = fps_src / target_fps
    indices = []
    acc = float(start_idx)
    for _ in range(total_to_export):
        idx = int(round(acc))
        idx = max(start_idx, min(end_idx, idx))
        if indices and idx < indices[-1]:
            idx = indices[-1]  # đảm bảo không giảm
        indices.append(idx)
        acc += step

    # Đảm bảo không vượt quá end_idx
    indices = [min(end_idx, i) for i in indices]
    return indices


def export_frames(video_path: Path, output_dir: Path, target_fps: float = 30.0,
                  image_ext: str = "jpg", quality: int = 95, overwrite: bool = False,
                  assume_native: bool = False, start_sec: Optional[float] = None,
                  end_sec: Optional[float] = None, prefix: Optional[str] = None,
                  dry_run: bool = False) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Một số container không cho biết chính xác FPS/FRAME_COUNT.
    if not fps_src or math.isnan(fps_src) or fps_src <= 0:
        fps_src = 30.0  # fallback hợp lý cho video 30FPS
    if frame_count <= 0:
        # fallback ước lượng bằng thời lượng (nếu có)
        dur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Không đáng tin, giữ nguyên 0 -> sẽ read tới khi hết
        pass

    # Tính khoảng index theo start/end (nếu có)
    if start_sec is None:
        start_frame = 0
    else:
        start_frame = max(0, int(round(start_sec * fps_src)))

    if end_sec is None:
        end_frame = frame_count - 1 if frame_count > 0 else -1
    else:
        end_frame = int(round(end_sec * fps_src))

    # Seek gần vị trí bắt đầu
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ensure_outdir(output_dir)

    stem = prefix if prefix is not None else video_path.stem
    n_digits = max(6, len(str(max(frame_count, 1))))

    saved = 0
    written_files: List[Path] = []

    # Chế độ native: xuất mọi frame gốc trong đoạn chọn
    if assume_native:
        # Đọc tuần tự từ start_frame tới end_frame
        current_idx = start_frame - 1
        pbar_total = (end_frame - start_frame + 1) if end_frame >= start_frame >= 0 else frame_count
        pbar = tqdm(total=max(0, pbar_total), desc="Exporting frames (native)", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_idx += 1
            if end_frame >= 0 and current_idx > end_frame:
                break

            fname = f"{stem}_{current_idx:0{n_digits}d}.{image_ext}"
            fpath = output_dir / fname
            if dry_run:
                saved += 1
                written_files.append(fpath)
            else:
                if fpath.exists() and not overwrite:
                    # bỏ qua nếu đã tồn tại
                    pass
                ok = imwrite_ext(fpath, frame, image_ext, quality)
                if not ok:
                    raise RuntimeError(f"Lưu ảnh thất bại: {fpath}")
                saved += 1
                written_files.append(fpath)
            pbar.update(1)
        pbar.close()

        # Tính duration dựa trên số frame đã xử lý
        actual_frames = (end_frame - start_frame + 1) if end_frame >= start_frame >= 0 else current_idx + 1
        duration = actual_frames / fps_src if fps_src > 0 else 0.0

        cap.release()
        return {
            "mode": "native",
            "fps_src": fps_src,
            "frame_count": frame_count,
            "exported": saved,
            "duration_s": duration,
            "output_dir": str(output_dir),
            "files": [str(p) for p in written_files],
        }

    # Chế độ lấy mẫu theo target_fps
    # Xây dựng danh sách index nguồn cần xuất
    if frame_count > 0 and end_frame < 0:
        end_frame = frame_count - 1

    src_indices = build_src_indices(
        frame_count=frame_count if frame_count > 0 else 10**9,
        fps_src=fps_src,
        target_fps=target_fps,
        start_idx=start_frame,
        end_idx=end_frame if end_frame >= 0 else (frame_count - 1 if frame_count > 0 else -1),
    )

    if not src_indices:
        # Fallback: đọc tuần tự và xuất theo bước gần đúng
        # (Trường hợp frame_count không xác định; vẫn cố gắng đọc theo thời gian thực)
        step = max(fps_src / target_fps, 1.0)
        target_next = start_frame
        current_idx = start_frame - 1
        exported = 0
        pbar = tqdm(desc=f"Exporting @ {target_fps:.3f} FPS", unit="img")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_idx += 1
            if end_frame >= 0 and current_idx > end_frame:
                break
            if current_idx >= target_next - 0.5:
                fname = f"{stem}_{exported:0{n_digits}d}.{image_ext}"
                fpath = output_dir / fname
                if not dry_run:
                    ok = imwrite_ext(fpath, frame, image_ext, quality)
                    if not ok:
                        raise RuntimeError(f"Lưu ảnh thất bại: {fpath}")
                exported += 1
                pbar.update(1)
                target_next += step
        pbar.close()
        cap.release()
        return {
            "mode": "resample(fallback)",
            "fps_src": fps_src,
            "frame_count": frame_count,
            "exported": exported,
            "duration_s": (end_frame - start_frame + 1) / fps_src if end_frame >= start_frame >= 0 and fps_src > 0 else None,
            "output_dir": str(output_dir),
        }

    # Đọc tuần tự & xuất khi gặp index khớp
    pbar = tqdm(total=len(src_indices), desc=f"Exporting @ {target_fps:.3f} FPS", unit="img")
    current_idx = start_frame - 1
    target_ptr = 0
    target_total = len(src_indices)

    while target_ptr < target_total:
        ret, frame = cap.read()
        if not ret:
            break
        current_idx += 1
        # Bỏ qua frame trước chỉ số đích
        if current_idx < src_indices[target_ptr]:
            continue
        # Với các duplicate chỉ số (do round), xuất từng lần để đảm bảo đủ số lượng ảnh
        while target_ptr < target_total and current_idx == src_indices[target_ptr]:
            fname = f"{stem}_{target_ptr:0{n_digits}d}.{image_ext}"
            fpath = output_dir / fname
            if dry_run:
                saved += 1
                written_files.append(fpath)
            else:
                if fpath.exists() and not overwrite:
                    pass
                ok = imwrite_ext(fpath, frame, image_ext, quality)
                if not ok:
                    raise RuntimeError(f"Lưu ảnh thất bại: {fpath}")
                saved += 1
                written_files.append(fpath)
            pbar.update(1)
            target_ptr += 1

        if end_frame >= 0 and current_idx >= end_frame:
            break

    pbar.close()
    cap.release()

    duration_sel = (end_frame - start_frame + 1) / fps_src if end_frame >= start_frame >= 0 and fps_src > 0 else None
    return {
        "mode": "resample",
        "fps_src": fps_src,
        "frame_count": frame_count,
        "exported": saved,
        "duration_s": duration_sel,
        "output_dir": str(output_dir),
        "files": [str(p) for p in written_files],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export frames from video at target FPS (default 30)."
    )
    p.add_argument("input", help="Đường dẫn file video (vd: input.mp4)")
    p.add_argument("-o", "--output", default=None, help="Thư mục xuất ảnh (mặc định: ./<tên_video>_frames)")
    p.add_argument("--fps", type=float, default=30.0, help="FPS mục tiêu (ảnh/giây), mặc định 30")
    p.add_argument("--native", action="store_true", help="Xuất tất cả frame gốc (nhanh). Dùng khi bạn chắc video là 30FPS và muốn 1s->30 ảnh.")
    p.add_argument("--ext", choices=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"], default="jpg", help="Định dạng ảnh xuất")
    p.add_argument("--quality", type=int, default=95, help="Chất lượng ảnh (JPG/WEBP: 1-100; PNG tự chuyển sang mức nén)")
    p.add_argument("--overwrite", action="store_true", help="Ghi đè nếu file đã tồn tại")
    p.add_argument("--start", type=float, default=None, help="Giây bắt đầu (vd: 10.0)")
    p.add_argument("--end", type=float, default=None, help="Giây kết thúc (vd: 25.0)")
    p.add_argument("--prefix", type=str, default=None, help="Tiền tố tên file xuất (mặc định dùng tên video)")
    p.add_argument("--dry-run", action="store_true", help="Chạy thử, không ghi file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    if not video_path.exists():
        print(f"❌ Không tìm thấy video: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else Path(f"./{video_path.stem}_frames")
    try:
        stats = export_frames(
            video_path=video_path,
            output_dir=out_dir,
            target_fps=args.fps,
            image_ext=args.ext,
            quality=args.quality,
            overwrite=args.overwrite,
            assume_native=args.native,
            start_sec=args.start,
            end_sec=args.end,
            prefix=args.prefix,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n⛔ Bị hủy bởi người dùng.")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Lỗi: {e}", file=sys.stderr)
        sys.exit(2)

    # In thống kê
    mode = stats.get("mode")
    fps_src = stats.get("fps_src")
    frame_count = stats.get("frame_count")
    exported = stats.get("exported")
    duration_s = stats.get("duration_s")

    print("\n=== Hoàn tất ===")
    print(f"Chế độ          : {mode}")
    print(f"FPS nguồn       : {fps_src:.3f}")
    if frame_count and frame_count > 0:
        print(f"Tổng frame video: {frame_count}")
    if duration_s is not None:
        print(f"Thời lượng đã xử lý: {human_time(duration_s)}")
    print(f"Đã xuất         : {exported} ảnh")
    print(f"Thư mục output  : {stats.get('output_dir')}")


if __name__ == "__main__":
    main()
