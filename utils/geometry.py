# bbox_iou, PCA direction, choose_direction_towards_head
import numpy as np

def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def pca_direction(points_xy):
    pts = np.array(points_xy, dtype=np.float32)
    mask = ~np.all(pts == 0, axis=1)
    pts_valid = pts[mask]
    if len(pts_valid) < 2:
        return None, None, 0

    centroid = pts_valid.mean(axis=0)
    X = pts_valid - centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    dir_vec = Vt[0]
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        return None, centroid, len(pts_valid)
    return dir_vec / norm, centroid, len(pts_valid)

def choose_direction_towards_head(dir_unit, centroid, head_xy):
    if dir_unit is None or centroid is None or head_xy is None:
        return None
    vec_head = np.array(head_xy, dtype=np.float32) - np.array(centroid, dtype=np.float32)
    if np.dot(dir_unit, vec_head) < 0:
        dir_unit = -dir_unit
    return dir_unit

def combine_direction(dir_pca, head, tail_avg, alpha=0.6):
    """
    Kết hợp vector PCA với vector head->tail.
    alpha=0 -> hoàn toàn head->tail
    alpha=1 -> hoàn toàn PCA
    """
    v52 = np.array(tail_avg, dtype=np.float32) - np.array(head, dtype=np.float32)
    norm_v52 = np.linalg.norm(v52)
    v52_unit = v52 / norm_v52 if norm_v52>0 else np.array([0.0,0.0])
    
    if dir_pca is None:
        dir_unit = v52_unit
    else:
        dir_unit = dir_pca
    
    # interpolation
    dir_final = alpha * dir_unit + (1-alpha) * v52_unit
    norm_final = np.linalg.norm(dir_final)
    if norm_final > 0:
        dir_final /= norm_final
    return dir_final