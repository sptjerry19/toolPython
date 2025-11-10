#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2frames.py
----------------
Tool xuất ảnh từ video theo FPS mục tiêu (mặc định 30 ảnh/giây) hoặc phân tích bong bóng.

CHẾ ĐỘ XUẤT FRAME:
- Hỗ trợ hai chế độ:
  1) --native: Xuất mọi frame gốc của video (nhanh). Thích hợp khi video là 30 FPS chuẩn và bạn muốn 1s -> 30 ảnh.
  2) Mặc định (không --native): Lấy mẫu theo --fps (mặc định 30). Hữu ích khi video không đúng 30 FPS nhưng bạn vẫn muốn 30 ảnh/giây ổn định.

CHẾ ĐỘ PHÂN TÍCH BONG BÓNG:
- Phát hiện và theo dõi bong bóng trong video
- Đếm số lượng bong bóng duy nhất
- Tính đường kính của từng bong bóng
- Tính vận tốc (theo pixel/giây) của từng bong bóng
- Xuất kết quả ra JSON và CSV
- Tùy chọn tạo video visualization với bong bóng được đánh dấu

Cài đặt:
    pip install -r requirements.txt

Ví dụ xuất frame:
    python video2frames.py input.mp4 -o ./frames
    # Ép 30 ảnh/giây, đặt tên theo index, chất lượng JPEG 95

    python video2frames.py input.mp4 -o ./frames --native
    # Xuất tất cả frame gốc (nếu video 30FPS thì 1s -> 30 ảnh)

    python video2frames.py input.mp4 -o ./frames --fps 30 --ext png
    # Lấy mẫu 30 ảnh/giây, xuất PNG

    python video2frames.py input.mp4 -o ./frames --start 10 --end 25
    # Chỉ xuất đoạn [10s, 25s]

Ví dụ phân tích bong bóng:
    python video2frames.py input.mp4 --analyze-bubbles
    # Phân tích toàn bộ video, xuất kết quả JSON và CSV

    python video2frames.py input.mp4 --analyze-bubbles --visualize
    # Phân tích và tạo video visualization

    python video2frames.py input.mp4 --analyze-bubbles --detection-method contour --min-radius 10 --max-radius 50
    # Sử dụng phương pháp contour detection với kích thước bong bóng tùy chỉnh

    python video2frames.py input.mp4 --analyze-bubbles --start 5 --end 15 --threshold 30
    # Phân tích đoạn [5s, 15s] với ngưỡng phát hiện thấp hơn

Tác giả: ChatGPT
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class Bubble:
    """Thông tin về một quả bong bóng"""
    id: int
    center_x: float
    center_y: float
    diameter: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_magnitude: float = 0.0
    frame_idx: int = 0
    first_seen: int = 0
    last_seen: int = 0


class BubbleTracker:
    """Theo dõi bong bóng qua các frame"""
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 5):
        self.next_id = 1
        self.bubbles: Dict[int, Bubble] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.history: Dict[int, List[Bubble]] = defaultdict(list)
    
    def update(self, detections: List[Tuple[float, float, float]], frame_idx: int, fps: float) -> List[Bubble]:
        """
        Cập nhật tracking với các phát hiện mới.
        detections: List[(center_x, center_y, diameter)]
        """
        if not detections:
            # Tăng counter cho các bong bóng biến mất
            for bubble_id in list(self.bubbles.keys()):
                self.disappeared[bubble_id] = self.disappeared.get(bubble_id, 0) + 1
                if self.disappeared[bubble_id] > self.max_disappeared:
                    # Xóa bong bóng đã biến mất quá lâu
                    del self.bubbles[bubble_id]
                    del self.disappeared[bubble_id]
            return list(self.bubbles.values())
        
        # Nếu chưa có bong bóng nào, tạo mới tất cả
        if not self.bubbles:
            for center_x, center_y, diameter in detections:
                bubble = Bubble(
                    id=self.next_id,
                    center_x=center_x,
                    center_y=center_y,
                    diameter=diameter,
                    frame_idx=frame_idx,
                    first_seen=frame_idx,
                    last_seen=frame_idx
                )
                self.bubbles[self.next_id] = bubble
                self.history[self.next_id].append(bubble)
                self.next_id += 1
            return list(self.bubbles.values())
        
        # Tính khoảng cách giữa detections và bubbles hiện tại
        detections_array = np.array([(d[0], d[1]) for d in detections])
        bubbles_array = np.array([(b.center_x, b.center_y) for b in self.bubbles.values()])
        bubble_ids = list(self.bubbles.keys())
        
        # Greedy matching: gán detection gần nhất cho mỗi bubble
        used_detections = set()
        matched_bubbles = set()
        
        # Sắp xếp theo khoảng cách
        distances = []
        for i, det in enumerate(detections_array):
            for j, bubble_pos in enumerate(bubbles_array):
                dist = np.linalg.norm(det - bubble_pos)
                if dist <= self.max_distance:
                    distances.append((dist, i, j))
        
        distances.sort()
        
        # Gán các cặp gần nhất
        for dist, det_idx, bubble_idx in distances:
            if det_idx not in used_detections and bubble_ids[bubble_idx] not in matched_bubbles:
                bubble_id = bubble_ids[bubble_idx]
                old_bubble = self.bubbles[bubble_id]
                center_x, center_y, diameter = detections[det_idx]
                
                # Tính vận tốc (pixel/frame)
                dt = frame_idx - old_bubble.frame_idx
                if dt > 0:
                    velocity_x = (center_x - old_bubble.center_x) / dt
                    velocity_y = (center_y - old_bubble.center_y) / dt
                    velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                else:
                    velocity_x = old_bubble.velocity_x
                    velocity_y = old_bubble.velocity_y
                    velocity_magnitude = old_bubble.velocity_magnitude
                
                # Cập nhật bubble
                new_bubble = Bubble(
                    id=bubble_id,
                    center_x=center_x,
                    center_y=center_y,
                    diameter=diameter,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    velocity_magnitude=velocity_magnitude,
                    frame_idx=frame_idx,
                    first_seen=old_bubble.first_seen,
                    last_seen=frame_idx
                )
                self.bubbles[bubble_id] = new_bubble
                self.history[bubble_id].append(new_bubble)
                self.disappeared[bubble_id] = 0
                used_detections.add(det_idx)
                matched_bubbles.add(bubble_id)
        
        # Tạo bubble mới cho các detection chưa được gán
        for i, (center_x, center_y, diameter) in enumerate(detections):
            if i not in used_detections:
                bubble = Bubble(
                    id=self.next_id,
                    center_x=center_x,
                    center_y=center_y,
                    diameter=diameter,
                    frame_idx=frame_idx,
                    first_seen=frame_idx,
                    last_seen=frame_idx
                )
                self.bubbles[self.next_id] = bubble
                self.history[self.next_id].append(bubble)
                self.next_id += 1
        
        # Xử lý các bubble không được match
        for bubble_id in list(self.bubbles.keys()):
            if bubble_id not in matched_bubbles:
                self.disappeared[bubble_id] = self.disappeared.get(bubble_id, 0) + 1
                if self.disappeared[bubble_id] > self.max_disappeared:
                    del self.bubbles[bubble_id]
                    del self.disappeared[bubble_id]
        
        return list(self.bubbles.values())
    
    def get_all_tracked(self) -> List[Bubble]:
        """Lấy tất cả bong bóng đã được track"""
        return list(self.bubbles.values())


def detect_bubbles(frame: np.ndarray, 
                   method: str = "hough",
                   min_radius: int = 5,
                   max_radius: int = 100,
                   threshold: int = 50) -> List[Tuple[float, float, float]]:
    """
    Phát hiện bong bóng trong frame.
    Trả về List[(center_x, center_y, diameter)]
    
    Args:
        frame: Frame ảnh BGR
        method: "hough" hoặc "contour"
        min_radius: Bán kính nhỏ nhất (pixel)
        max_radius: Bán kính lớn nhất (pixel)
        threshold: Ngưỡng cho HoughCircles hoặc binary threshold
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if method == "hough":
        # Sử dụng HoughCircles để phát hiện hình tròn
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius * 2,
            param1=50,
            param2=threshold,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detections.append((float(x), float(y), float(r * 2)))
        return detections
    
    else:  # method == "contour"
        # Sử dụng contour detection
        # Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Adaptive threshold hoặc Otsu
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < math.pi * min_radius ** 2:
                continue
            
            # Tính bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if min_radius <= radius <= max_radius:
                # Kiểm tra độ tròn (circularity)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter ** 2)
                    if circularity > 0.5:  # Đủ tròn
                        detections.append((float(x), float(y), float(radius * 2)))
        
        return detections


def analyze_bubbles(video_path: Path,
                    fps: Optional[float] = None,
                    start_sec: Optional[float] = None,
                    end_sec: Optional[float] = None,
                    detection_method: str = "hough",
                    min_radius: int = 5,
                    max_radius: int = 100,
                    threshold: int = 50,
                    max_tracking_distance: float = 50.0,
                    max_disappeared_frames: int = 5,
                    visualize: bool = False,
                    output_video: Optional[Path] = None) -> Dict:
    """
    Phân tích video để đếm bong bóng, tính đường kính và vận tốc.
    
    Returns:
        Dict chứa thống kê và danh sách bong bóng
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or math.isnan(video_fps) or video_fps <= 0:
        video_fps = 30.0
    
    if fps is None:
        fps = video_fps
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tính frame range
    if start_sec is None:
        start_frame = 0
    else:
        start_frame = max(0, int(round(start_sec * video_fps)))
    
    if end_sec is None:
        end_frame = frame_count - 1 if frame_count > 0 else -1
    else:
        end_frame = int(round(end_sec * video_fps))
    
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    tracker = BubbleTracker(max_distance=max_tracking_distance, max_disappeared=max_disappeared_frames)
    
    # Video writer cho visualization
    out_writer = None
    if visualize and output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_video), fourcc, video_fps, (width, height))
    
    frame_idx = start_frame - 1
    all_bubbles_data = []
    
    pbar = tqdm(desc="Phân tích bong bóng", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if end_frame >= 0 and frame_idx > end_frame:
            break
        
        # Phát hiện bong bóng
        detections = detect_bubbles(
            frame,
            method=detection_method,
            min_radius=min_radius,
            max_radius=max_radius,
            threshold=threshold
        )
        
        # Cập nhật tracking
        tracked_bubbles = tracker.update(detections, frame_idx, fps)
        
        # Lưu dữ liệu frame hiện tại
        frame_data = {
            "frame_idx": frame_idx,
            "time_sec": frame_idx / video_fps,
            "bubbles": [asdict(b) for b in tracked_bubbles]
        }
        all_bubbles_data.append(frame_data)
        
        # Visualization
        if visualize:
            vis_frame = frame.copy()
            for bubble in tracked_bubbles:
                x, y = int(bubble.center_x), int(bubble.center_y)
                r = int(bubble.diameter / 2)
                
                # Vẽ vòng tròn
                cv2.circle(vis_frame, (x, y), r, (0, 255, 0), 2)
                
                # Vẽ ID và thông tin
                label = f"ID:{bubble.id} D:{bubble.diameter:.1f}px"
                cv2.putText(vis_frame, label, (x - 50, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Vẽ vector vận tốc
                if bubble.velocity_magnitude > 0:
                    vx, vy = bubble.velocity_x * 5, bubble.velocity_y * 5  # Scale để dễ nhìn
                    cv2.arrowedLine(vis_frame, (x, y), 
                                   (int(x + vx), int(y + vy)), 
                                   (255, 0, 0), 2)
            
            if out_writer:
                out_writer.write(vis_frame)
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if out_writer:
        out_writer.release()
    
    # Tổng hợp thống kê
    all_tracked = tracker.get_all_tracked()
    unique_bubbles = set(b.id for b in all_tracked)
    
    # Tính thống kê cho từng bong bóng
    bubble_stats = []
    for bubble_id in unique_bubbles:
        history = tracker.history[bubble_id]
        if not history:
            continue
        
        # Tính đường kính trung bình
        avg_diameter = np.mean([b.diameter for b in history])
        
        # Tính vận tốc trung bình (pixel/frame -> pixel/second)
        velocities_x = [b.velocity_x for b in history if b.velocity_x != 0]
        velocities_y = [b.velocity_y for b in history if b.velocity_y != 0]
        velocities_mag = [b.velocity_magnitude for b in history if b.velocity_magnitude > 0]
        
        avg_velocity_x = np.mean(velocities_x) * fps if velocities_x else 0.0
        avg_velocity_y = np.mean(velocities_y) * fps if velocities_y else 0.0
        avg_velocity_magnitude = np.mean(velocities_mag) * fps if velocities_mag else 0.0
        
        # Vị trí đầu và cuối
        first = history[0]
        last = history[-1]
        
        bubble_stats.append({
            "id": bubble_id,
            "first_seen_frame": first.first_seen,
            "last_seen_frame": last.last_seen,
            "lifetime_frames": last.last_seen - first.first_seen + 1,
            "lifetime_seconds": (last.last_seen - first.first_seen + 1) / fps,
            "avg_diameter_pixels": float(avg_diameter),
            "avg_velocity_x_pixels_per_sec": float(avg_velocity_x),
            "avg_velocity_y_pixels_per_sec": float(avg_velocity_y),
            "avg_velocity_magnitude_pixels_per_sec": float(avg_velocity_magnitude),
            "start_position": {"x": float(first.center_x), "y": float(first.center_y)},
            "end_position": {"x": float(last.center_x), "y": float(last.center_y)},
            "total_distance_pixels": float(np.sqrt((last.center_x - first.center_x)**2 + 
                                                   (last.center_y - first.center_y)**2))
        })
    
    return {
        "video_path": str(video_path),
        "fps": float(fps),
        "total_frames_analyzed": frame_idx - start_frame + 1,
        "total_unique_bubbles": len(unique_bubbles),
        "bubble_statistics": bubble_stats,
        "frame_by_frame_data": all_bubbles_data
    }


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
        description="Export frames from video at target FPS (default 30) hoặc phân tích bong bóng."
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
    
    # Thêm các tham số cho phân tích bong bóng
    p.add_argument("--analyze-bubbles", action="store_true", help="Phân tích bong bóng: đếm số lượng, tính đường kính và vận tốc")
    p.add_argument("--detection-method", choices=["hough", "contour"], default="hough", help="Phương pháp phát hiện bong bóng: hough (HoughCircles) hoặc contour")
    p.add_argument("--min-radius", type=int, default=5, help="Bán kính nhỏ nhất của bong bóng (pixel)")
    p.add_argument("--max-radius", type=int, default=100, help="Bán kính lớn nhất của bong bóng (pixel)")
    p.add_argument("--threshold", type=int, default=50, help="Ngưỡng phát hiện (cho HoughCircles)")
    p.add_argument("--max-tracking-distance", type=float, default=50.0, help="Khoảng cách tối đa để tracking bong bóng giữa các frame (pixel)")
    p.add_argument("--max-disappeared", type=int, default=5, help="Số frame tối đa bong bóng biến mất trước khi xóa khỏi tracking")
    p.add_argument("--visualize", action="store_true", help="Tạo video visualization với bong bóng được đánh dấu")
    p.add_argument("--output-video", type=str, default=None, help="Đường dẫn video output cho visualization (mặc định: <tên_video>_analyzed.mp4)")
    p.add_argument("--output-json", type=str, default=None, help="Đường dẫn file JSON để lưu kết quả phân tích (mặc định: <tên_video>_bubbles.json)")
    p.add_argument("--output-csv", type=str, default=None, help="Đường dẫn file CSV để lưu thống kê bong bóng (mặc định: <tên_video>_bubbles.csv)")
    
    return p.parse_args()


def save_bubble_results_json(results: Dict, output_path: Path) -> None:
    """Lưu kết quả phân tích bong bóng ra file JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã lưu kết quả JSON: {output_path}")


def save_bubble_results_csv(results: Dict, output_path: Path) -> None:
    """Lưu thống kê bong bóng ra file CSV"""
    import csv
    
    bubble_stats = results.get("bubble_statistics", [])
    if not bubble_stats:
        print("⚠️  Không có dữ liệu bong bóng để xuất CSV")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "first_seen_frame", "last_seen_frame", "lifetime_frames", "lifetime_seconds",
            "avg_diameter_pixels", "avg_velocity_x_pixels_per_sec", "avg_velocity_y_pixels_per_sec",
            "avg_velocity_magnitude_pixels_per_sec", "start_position_x", "start_position_y",
            "end_position_x", "end_position_y", "total_distance_pixels"
        ])
        writer.writeheader()
        
        for stat in bubble_stats:
            row = {
                "id": stat["id"],
                "first_seen_frame": stat["first_seen_frame"],
                "last_seen_frame": stat["last_seen_frame"],
                "lifetime_frames": stat["lifetime_frames"],
                "lifetime_seconds": stat["lifetime_seconds"],
                "avg_diameter_pixels": stat["avg_diameter_pixels"],
                "avg_velocity_x_pixels_per_sec": stat["avg_velocity_x_pixels_per_sec"],
                "avg_velocity_y_pixels_per_sec": stat["avg_velocity_y_pixels_per_sec"],
                "avg_velocity_magnitude_pixels_per_sec": stat["avg_velocity_magnitude_pixels_per_sec"],
                "start_position_x": stat["start_position"]["x"],
                "start_position_y": stat["start_position"]["y"],
                "end_position_x": stat["end_position"]["x"],
                "end_position_y": stat["end_position"]["y"],
                "total_distance_pixels": stat["total_distance_pixels"]
            }
            writer.writerow(row)
    
    print(f"✅ Đã lưu thống kê CSV: {output_path}")


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    if not video_path.exists():
        print(f"❌ Không tìm thấy video: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Chế độ phân tích bong bóng
    if args.analyze_bubbles:
        try:
            output_video = None
            if args.visualize:
                if args.output_video:
                    output_video = Path(args.output_video)
                else:
                    output_video = Path(f"./{video_path.stem}_analyzed.mp4")
            
            results = analyze_bubbles(
                video_path=video_path,
                fps=args.fps,
                start_sec=args.start,
                end_sec=args.end,
                detection_method=args.detection_method,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                threshold=args.threshold,
                max_tracking_distance=args.max_tracking_distance,
                max_disappeared_frames=args.max_disappeared,
                visualize=args.visualize,
                output_video=output_video
            )
            
            # Lưu kết quả JSON
            if args.output_json:
                json_path = Path(args.output_json)
            else:
                json_path = Path(f"./{video_path.stem}_bubbles.json")
            save_bubble_results_json(results, json_path)
            
            # Lưu kết quả CSV
            if args.output_csv:
                csv_path = Path(args.output_csv)
            else:
                csv_path = Path(f"./{video_path.stem}_bubbles.csv")
            save_bubble_results_csv(results, csv_path)
            
            # In thống kê
            print("\n=== Kết quả phân tích bong bóng ===")
            print(f"Video            : {results['video_path']}")
            print(f"FPS              : {results['fps']:.3f}")
            print(f"Tổng frame đã phân tích: {results['total_frames_analyzed']}")
            print(f"Tổng số bong bóng: {results['total_unique_bubbles']}")
            
            if results['bubble_statistics']:
                avg_diameter = np.mean([b['avg_diameter_pixels'] for b in results['bubble_statistics']])
                avg_velocity = np.mean([b['avg_velocity_magnitude_pixels_per_sec'] for b in results['bubble_statistics'] if b['avg_velocity_magnitude_pixels_per_sec'] > 0])
                print(f"Đường kính trung bình: {avg_diameter:.2f} pixel")
                if avg_velocity > 0:
                    print(f"Vận tốc trung bình: {avg_velocity:.2f} pixel/giây")
            
            if output_video:
                print(f"Video visualization: {output_video}")
            
        except KeyboardInterrupt:
            print("\n⛔ Bị hủy bởi người dùng.")
            sys.exit(130)
        except Exception as e:
            print(f"❌ Lỗi: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(2)
        return

    # Chế độ xuất frame (mặc định)
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
