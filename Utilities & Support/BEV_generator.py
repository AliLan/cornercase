import os
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# === CONFIGURATION ===
MODEL_PATH = "frcnn_nuscenes_epoch_50.pth"  # <--- UPDATE THIS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCORE_THRESHOLD = 0.3

# === CLASS MAPPING
CLASS_TO_IDX = {
    'car': 1, 'truck': 2, 'bus': 3, 'trailer': 4, 'construction': 5,
    'pedestrian': 6, 'motorcycle': 7, 'bicycle': 8,
    'van': 9, 'person_sitting': 10, 'tram': 11
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# === BEV PARAMETERS
X_RANGE = (-70.4, 70.4)
Y_RANGE = (-70.4, 70.4)
Z_RANGE = (-3, 3)
RES = 0.25

def load_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_TO_IDX) + 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    if points.shape[1] == 3:
        intensity = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, intensity], axis=1)
    return points

def point_cloud_to_bev(points):
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    mask = (x > X_RANGE[0]) & (x < X_RANGE[1]) & \
           (y > Y_RANGE[0]) & (y < Y_RANGE[1]) & \
           (z > Z_RANGE[0]) & (z < Z_RANGE[1])
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    x_img = ((x - X_RANGE[0]) / RES).astype(np.int32)
    y_img = (((y * -1) - Y_RANGE[0]) / RES).astype(np.int32)

    x_max = int((X_RANGE[1] - X_RANGE[0]) / RES)
    y_max = int((Y_RANGE[1] - Y_RANGE[0]) / RES)

    bev_img = np.zeros((3, y_max, x_max), dtype=np.float32)
    x_img = np.clip(x_img, 0, x_max - 1)
    y_img = np.clip(y_img, 0, y_max - 1)

    bev_img[0, y_img, x_img] = z
    bev_img[1, y_img, x_img] = intensity
    bev_img[2, y_img, x_img] = 0
    return bev_img

def draw_boxes_on_bev(bev_img, boxes, labels, scores):
    # Use Z-channel for base visualization
    z_img = bev_img[0]
    norm_img = ((z_img - z_img.min()) / (z_img.ptp() + 1e-5) * 255).astype(np.uint8)
    vis_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

    for box, label, score in zip(boxes, labels, scores):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        cls = IDX_TO_CLASS.get(label, 'unknown')
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"{cls} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis_img

def convert_ply_to_png(ply_path: str, out_path: str =None):

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"File not found: {ply_path}")
    
    model = load_model()

    points = load_ply(ply_path)
    bev_img = point_cloud_to_bev(points)
    input_tensor = torch.tensor(bev_img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model([input_tensor.squeeze(0)])[0]

    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().tolist()
    scores = pred['scores'].cpu().tolist()

    vis_img = draw_boxes_on_bev(bev_img, boxes, labels, scores)
    if out_path is None:
        out_path = ply_path.replace(".ply", ".png")
    cv2.imwrite(out_path, vis_img)
    return out_path

