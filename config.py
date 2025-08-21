# Config file
from pathlib import Path
import torch

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
YOLO_WEIGHTS = Path("best.pt")
REID_WEIGHTS = Path("osnet_x0_25_msmt17.pt")

# Video input/output
VIDEO_INPUT = "test1.mp4"
VIDEO_OUTPUT = "output_slikworm_17.mp4"

# Tracker config
USE_KALMAN = False
EMA_ALPHA = 0.5
DEFAULT_STEP = 30.0
MAX_HISTORY = 10

# Regions
REGIONS = {
    "zone1": [(50, 50), (700, 50), (700, 500), (50, 500)]
}
