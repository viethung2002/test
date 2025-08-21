# SimpleKalman2D
import cv2
import numpy as np

class SimpleKalman2D:
    def __init__(self, x0, y0, dt=1.0, process_var=1e-2, meas_var=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],
                                             [0,1,0,dt],
                                             [0,0,1,0],
                                             [0,0,0,1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_var
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_var
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[x0],[y0],[0],[0]], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return pred[:2].ravel(), pred[2:].ravel()

    def correct(self, x, y):
        z = np.array([[x],[y]], dtype=np.float32)
        upd = self.kf.correct(z)
        return upd[:2].ravel(), upd[2:].ravel()
