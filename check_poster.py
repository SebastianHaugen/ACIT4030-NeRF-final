"""
Quick check of the poster data before training: image size and where the cameras are.
Run from the repo root:  python check_poster.py
"""
import glob
import json

import numpy as np
from PIL import Image

with open("data/poster/transforms.json", encoding="utf-8") as f:
    meta = json.load(f)

frames = meta["frames"]
first = frames[0]["file_path"].replace("./", "")
img = Image.open(f"data/poster/{first}")
print(f"Images: {len(frames)}, size (width x height): {img.size[0]} x {img.size[1]}")
print(f"Image files found: {len(glob.glob('data/poster/images/*'))}")

# Camera centres are the translation part of each camera-to-world matrix
centers = np.array([np.array(fr["transform_matrix"])[:3, 3] for fr in frames])
# Viewing direction is the camera's -z axis (OpenGL convention in transforms.json)
dirs = np.array([-np.array(fr["transform_matrix"])[:3, 2] for fr in frames])

mid = centers.mean(axis=0)
dist_origin = np.linalg.norm(centers, axis=1)
spread = np.linalg.norm(centers - mid, axis=1)
print(f"Camera centre mean:            {np.round(mid, 3)}")
print(f"Distance camera -> origin:     min {dist_origin.min():.3f}  mean {dist_origin.mean():.3f}  max {dist_origin.max():.3f}")
print(f"Camera spread around the mean: max {spread.max():.3f}")
print(f"Bounding box of cameras:       min {np.round(centers.min(0), 3)}  max {np.round(centers.max(0), 3)}")

# Point where all viewing rays come closest together = roughly what the cameras look at
A = np.zeros((3, 3))
b = np.zeros(3)
for c, d in zip(centers, dirs):
    d = d / np.linalg.norm(d)
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ c
focus = np.linalg.solve(A, b)
dist_focus = np.linalg.norm(centers - focus, axis=1)
print(f"Point the cameras look at:     {np.round(focus, 3)}")
print(f"Distance camera -> that point: min {dist_focus.min():.3f}  mean {dist_focus.mean():.3f}  max {dist_focus.max():.3f}")
print("Extra keys in transforms.json:", [k for k in meta if k != "frames"])