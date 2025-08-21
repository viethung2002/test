# Keypoints processing, direction, head prediction
import numpy as np
import cv2
from utils.geometry import pca_direction, choose_direction_towards_head

def process_keypoints(frame, kpts, tid, pose_head_hist, pose_head_ema_speed,
                      EMA_ALPHA=0.5, DEFAULT_STEP=30.0, MAX_HISTORY=10):
    """
    Trả về: head_xy, dir_unit, pred_head
    """
    kpts_list = []
    head_xy = None
    for j, (kx, ky) in enumerate(kpts, start=1):
        if kx > 0 and ky > 0:
            cv2.circle(frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)
            cv2.putText(frame, str(j), (int(kx)+4, int(ky)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            kpts_list.append([int(kx), int(ky)])
        else:
            kpts_list.append([0, 0])

    # nối chuỗi 1-3-4-5-2
    connect_order = [1,3,4,5,2]
    for a,b in zip(connect_order[:-1], connect_order[1:]):
        if kpts_list[a-1][0] > 0 and kpts_list[a-1][1] > 0 and \
           kpts_list[b-1][0] > 0 and kpts_list[b-1][1] > 0:
            cv2.line(frame, tuple(kpts_list[a-1]), tuple(kpts_list[b-1]), (255,0,255), 2)

    if kpts_list and kpts_list[0] != [0,0]:
        head_xy = np.array(kpts_list[0], dtype=np.float32)

    dir_unit, centroid, _ = pca_direction(kpts_list)
    dir_unit = choose_direction_towards_head(dir_unit, centroid, head_xy)

    # update history
    if head_xy is not None:
        pose_head_hist.setdefault(tid, [])
        pose_head_hist[tid].append(tuple(head_xy))
        if len(pose_head_hist[tid]) > MAX_HISTORY:
            pose_head_hist[tid].pop(0)

    step_px = DEFAULT_STEP
    if head_xy is not None and len(pose_head_hist[tid]) >= 2:
        last = np.array(pose_head_hist[tid][-2], dtype=np.float32)
        cur  = np.array(pose_head_hist[tid][-1], dtype=np.float32)
        inst_speed = np.linalg.norm(cur-last)
        prev = pose_head_ema_speed.get(tid, inst_speed)
        ema = EMA_ALPHA*inst_speed + (1-EMA_ALPHA)*prev
        pose_head_ema_speed[tid] = float(ema)
        step_px = max(10.0, min(ema*1.5, 80.0))

    pred_head = None
    if dir_unit is not None and head_xy is not None:
        pred_head = (head_xy + dir_unit*step_px).astype(int)

        # vẽ
        tail = (head_xy - dir_unit*25).astype(int)
        tip  = (head_xy + dir_unit*25).astype(int)
        cv2.arrowedLine(frame, tuple(tail), tuple(tip), (0,255,255),2,tipLength=0.35)
        cv2.circle(frame, tuple(pred_head), 5, (0,255,255), -1)
        cv2.putText(frame,"pred",(int(pred_head[0])+6,int(pred_head[1])-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

    return head_xy, dir_unit, pred_head
