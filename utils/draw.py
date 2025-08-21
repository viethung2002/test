# Drawing functions (bbox, keypoints, arrow, zone)
import cv2
import numpy as np

# ======================
# Vẽ bounding box
# ======================

def draw_bbox(frame, bbox, keypoints=None, track_id=None, color=(0, 255, 0), thickness=2):
    """
    Vẽ bounding box, keypoints và track_id lên frame.
    
    Args:
        frame: ảnh gốc
        bbox: [x1, y1, x2, y2]
        keypoints: [[x, y], ...] (optional)
        track_id: ID của đối tượng (optional)
        color: màu bbox
        thickness: độ dày bbox
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Vẽ track ID nếu có
    if track_id is not None and track_id != -1:
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Vẽ keypoints nếu có
    if keypoints is not None:
        for kp in keypoints:
            if kp is not None and len(kp) >= 2:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # đỏ



# ======================
# Vẽ keypoints
# ======================
# Vẽ keypoints (theo thứ tự nối 1-3-4-5-2)
# ======================
def draw_keypoints(image, keypoints, color=(0, 0, 255), radius=3, line_color=(255, 0, 255)):
    """
    Vẽ keypoints và nối theo thứ tự 1-3-4-5-2.
    Args:
        image: ảnh gốc.
        keypoints: ndarray/list [[x, y], ...].
        color: màu điểm (B, G, R).
        radius: bán kính điểm.
        line_color: màu đường nối.
    """
    kpts_list = []
    for (x, y) in keypoints:
        if x > 0 and y > 0:  # chỉ vẽ điểm hợp lệ
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
            kpts_list.append((int(x), int(y)))
        else:
            kpts_list.append(None)

    # Nối theo thứ tự 1-3-4-5-2 (index bắt đầu từ 1)
    connect_order = [1, 3, 4, 5, 2]
    for a, b in zip(connect_order[:-1], connect_order[1:]):
        if (kpts_list[a-1] is not None) and (kpts_list[b-1] is not None):
            cv2.line(image, kpts_list[a-1], kpts_list[b-1], line_color, 2)

    return image


# ======================
# Vẽ mũi tên chỉ hướng
# ======================
def draw_arrow(image, start_point, end_point, color=(255, 0, 0), thickness=2):
    """
    Vẽ mũi tên từ start_point đến end_point.
    Args:
        start_point: (x, y).
        end_point: (x, y).
        color: màu (B, G, R).
        thickness: độ dày đường.
    """
    start_point = tuple(map(int, start_point))
    end_point = tuple(map(int, end_point))
    cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.3)
    return image


# ======================
# Vẽ zone (đa giác)
# ======================
def draw_zone(image, points, color=(0, 255, 255), alpha=0.3):
    """
    Vẽ zone (đa giác) bán trong suốt.
    Args:
        image: ảnh gốc.
        points: list [(x, y), ...] định nghĩa polygon.
        color: màu (B, G, R).
        alpha: độ trong suốt (0.0 -> 1.0).
    """
    overlay = image.copy()
    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    # blend vào ảnh gốc
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image
