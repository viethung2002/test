import cv2
import numpy as np

def count_in_zone(frame, zone_points, bboxes, zone_id=0, color=(0, 0, 255)):
    """
    Đếm số lượng bbox nằm trong zone (đa giác).
    
    Args:
        frame: ảnh gốc để vẽ
        zone_points: list các điểm [(x1,y1), (x2,y2), ...] tạo thành polygon
        bboxes: list bbox [[x1,y1,x2,y2], ...]
        zone_id: id của zone (số thứ tự)
        color: màu vẽ (BGR)

    Returns:
        count: số bbox nằm trong zone
    """
    zone_poly = np.array(zone_points, np.int32).reshape((-1, 1, 2))

    count = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if cv2.pointPolygonTest(zone_poly, (cx, cy), False) >= 0:
            count += 1

    # Vẽ zone
    cv2.polylines(frame, [zone_poly], True, color, 2)
    cv2.putText(frame, f"Zone {zone_id}: {count}", 
                (zone_points[0][0], zone_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return count
