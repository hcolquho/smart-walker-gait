"""
Standard OpenCV checkerboard camera calibration -> produces a
CameraIntrinsics YAML file consumable by any FrameSource backend.

Run this exact script today against your webcam to validate the procedure,
then re-run it unchanged against the Femto Bolt's RGB stream once it arrives
(and optionally read the Bolt's own factory intrinsics via the Orbbec SDK as
a cross-check). Zero throwaway work either way.

Usage:
    python -m walker_gait.calibration.calibrate_camera \
        --images "calib_images/*.jpg" \
        --board-cols 9 --board-rows 6 --square-size-m 0.025 \
        --out configs/camera_intrinsics_measured.yaml
"""
from __future__ import annotations
import argparse
import glob
import sys

import numpy as np
import yaml


def calibrate(image_paths, board_cols, board_rows, square_size_m):
    import cv2

    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_m

    obj_points, img_points = [], []
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  skip (unreadable): {path}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (board_cols, board_rows), None)
        if not found:
            print(f"  no checkerboard found: {path}", file=sys.stderr)
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 5:
        raise RuntimeError(
            f"Only {len(obj_points)} usable calibration images found -- need at least "
            "~10-15 varied views (different angles/distances/positions in frame) for a "
            "reliable calibration."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    reproj_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        reproj_errors.append(err)
    mean_reproj_error = float(np.mean(reproj_errors))

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "depth_scale": 1.0,
        "width": img_shape[0], "height": img_shape[1],
        "_distortion_coeffs": dist_coeffs.flatten().tolist(),
        "_mean_reprojection_error_px": mean_reproj_error,
        "_n_images_used": len(obj_points),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Glob pattern for checkerboard images, e.g. 'calib/*.jpg'")
    p.add_argument("--board-cols", type=int, default=9, help="Inner corners per row")
    p.add_argument("--board-rows", type=int, default=6, help="Inner corners per column")
    p.add_argument("--square-size-m", type=float, default=0.025, help="Checkerboard square size in meters")
    p.add_argument("--out", default="camera_intrinsics_measured.yaml")
    args = p.parse_args()

    image_paths = sorted(glob.glob(args.images))
    print(f"Found {len(image_paths)} candidate images.")
    result = calibrate(image_paths, args.board_cols, args.board_rows, args.square_size_m)

    print(f"Mean reprojection error: {result['_mean_reprojection_error_px']:.4f} px "
          f"(good calibration is typically < 0.5 px; > 1.0 px means re-shoot images)")

    with open(args.out, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"Wrote intrinsics to {args.out}")


if __name__ == "__main__":
    main()
