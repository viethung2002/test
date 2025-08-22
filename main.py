# main.py
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import BoostTrack
from utils.draw import draw_bbox, draw_keypoints, draw_arrow, draw_zone
from utils.zone import count_in_zone
import config
import datetime
import time
import psutil
from collections import deque  # thêm để lưu history
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------
# Hàm phản chiếu vector qua vector tham chiếu
# --------------------
def reflect_vector(vec, ref_vec):
    ref_unit = ref_vec / np.linalg.norm(ref_vec)
    dot = np.dot(vec, ref_unit)
    return 2*dot*ref_unit - vec

def compute_direction_from_keypoints(kpts):
    """
    kpts: (K,2) keypoints pixel
    Trả về vector hướng tổng (unit) từ keypoint 1
    """
    if len(kpts) < 5:
        return None

    head = np.array(kpts[0], dtype=np.float32)  # keypoint 1
    pt2 = np.array(kpts[1], dtype=np.float32)
    pt3 = np.array(kpts[2], dtype=np.float32)
    pt4 = np.array(kpts[3], dtype=np.float32)
    pt5 = np.array(kpts[4], dtype=np.float32)

    # Vector 3->1 (tham chiếu)
    vec_3to1 = head - pt3

    vectors = [vec_3to1]

    # Các vector khác 2->1,4->1,5->1 phản chiếu qua 3->1
    for pt in [pt2, pt4, pt5]:
        if np.any(pt > 0):
            vec = head - pt
            if np.linalg.norm(vec_3to1) > 0:
                vec = reflect_vector(vec, vec_3to1)
            vectors.append(vec)

    # Vector tổng
    vec_total = np.sum(vectors, axis=0)
    norm = np.linalg.norm(vec_total)
    if norm > 0:
        vec_unit = vec_total / norm
    else:
        vec_unit = None

    return vec_unit



def main():
    print("🔹 Loading YOLO model...")
    model = YOLO(config.YOLO_WEIGHTS)

    print("🔹 Initializing tracker...")
    tracker = BoostTrack(
        reid_weights=Path(config.REID_WEIGHTS),
        device=device,
        half=False
    )

    # Lưu history điểm đầu cho mỗi con tằm
    track_history = {}  # track_id -> deque of head points

    cap = cv2.VideoCapture(config.VIDEO_INPUT)
    frame_count = 0
    start_time = time.time()
    process = psutil.Process()
    max_ram = 0
    if not cap.isOpened():
        print(f"❌ Cannot open video: {config.VIDEO_INPUT}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.VIDEO_OUTPUT, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        if results.keypoints is not None:
            keypoints = results.keypoints.xy.cpu().numpy()
        else:
            keypoints = []

        detections = np.column_stack((boxes, confs, classes)) if len(boxes) > 0 else np.empty((0, 6))
        tracks = tracker.update(detections, frame)

        # Duyệt qua từng track
        for t in tracks:
            x1, y1, x2, y2, track_id, cls_id, conf = t[:7]
            draw_bbox(frame, (x1, y1, x2, y2), track_id=track_id)

            # tìm keypoint head trong bbox
            for kps in keypoints:
                head = kps[0]  # điểm đầu là keypoint 1 (index 0)
                if x1 <= head[0] <= x2 and y1 <= head[1] <= y2:
                    # Lưu vào history
                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=10)  # lưu tối đa 50 điểm
                    track_history[track_id].append(tuple(head.astype(int)))

                    # Vẽ keypoints
                    draw_keypoints(frame, kps)

                    # Vẽ vector hướng
                    dir_vec = compute_direction_from_keypoints(kps)
                    if dir_vec is not None:
                        start_pt = tuple(head.astype(int))
                        end_pt = tuple((head + dir_vec*50).astype(int))
                        cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 2, tipLength=0.3)

                    # Vẽ quỹ đạo của riêng track_id này
                    pts = list(track_history[track_id])
                    if len(pts) > 1:
                        for i in range(1, len(pts)):
                            cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 2)

                    break  # chỉ cần 1 bộ keypoint khớp với bbox

        # Vùng
        for zone_name, zone_points in config.REGIONS.items():
            draw_zone(frame, zone_points)
            bboxes = tracks[:, :4] if len(tracks) > 0 else np.empty((0, 4))
            count = count_in_zone(tracks, zone_points, bboxes)
            cv2.putText(
                frame,
                f"{zone_name}: {count}",
                (zone_points[0][0], zone_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        frame_count += 1
        ram_usage = process.memory_info().rss / (1024 * 1024)
        if ram_usage > max_ram:
            max_ram = ram_usage

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n---\nProcessed {frame_count} frames in {elapsed:.2f} seconds.")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Peak RAM usage: {max_ram:.2f} MB")


if __name__ == "__main__":
    main()
