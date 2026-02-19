#!/usr/bin/env python3
"""
Trajectory resampling tool for COLMAP camera poses.

Given 228 COLMAP images (every 7th frame from ~1200 robot images), this script:
1. Parses camera poses from COLMAP's images.txt (quaternion + translation).
2. Converts COLMAP's camera-to-world representation to world-frame camera positions.
3. Interpolates the trajectory back to the full ~1200 frames using cubic splines.
4. Resamples the trajectory at equal arc-length intervals.
5. Outputs the frame indices (from the full 1200) that best match the resampled positions.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline
from pathlib import Path
import argparse
import shutil


def parse_colmap_images_txt(filepath: str) -> dict:
    """
    Parse COLMAP images.txt file.

    Returns dict mapping frame index (int) -> {quat: [qw, qx, qy, qz], tvec: [tx, ty, tz]}
    COLMAP stores the world-to-camera transform: t and R such that X_cam = R @ X_world + t
    So the camera center in world coords is: C = -R^T @ t
    """
    images = {}
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Skip comment lines (lines starting with #)
    data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]

    # Every two lines: first is pose, second is 2D points
    for i in range(0, len(data_lines), 2):
        parts = data_lines[i].split()
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        # Extract frame number from name like "frame_000227.jpg"
        frame_idx = int(name.split("_")[1].split(".")[0])

        images[frame_idx] = {
            "image_id": image_id,
            "quat": np.array([qx, qy, qz, qw]),  # scipy uses [x, y, z, w]
            "tvec": np.array([tx, ty, tz]),
            "name": name,
        }

    return images


def colmap_pose_to_world_position(quat_xyzw: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP pose (world-to-camera) to camera center in world coordinates.
    C_world = -R^T @ t
    """
    R = Rotation.from_quat(quat_xyzw)
    camera_center = -R.inv().apply(tvec)
    return camera_center


def build_trajectory(images: dict) -> tuple:
    """
    Build ordered trajectory from parsed COLMAP images.
    Returns (frame_indices, positions, rotations) sorted by frame index.
    """
    frame_indices = sorted(images.keys())
    positions = []
    rotations = []

    for idx in frame_indices:
        pos = colmap_pose_to_world_position(images[idx]["quat"], images[idx]["tvec"])
        positions.append(pos)
        rotations.append(Rotation.from_quat(images[idx]["quat"]))

    return np.array(frame_indices), np.array(positions), rotations


def interpolate_trajectory(
    colmap_frame_indices: np.ndarray,
    positions: np.ndarray,
    rotations: list,
    total_frames: int,
    subsample_step: int,
) -> tuple:
    """
    Interpolate the sparse COLMAP trajectory to the full frame set.

    Args:
        colmap_frame_indices: Frame indices of the 228 COLMAP images (e.g., 0, 1, ..., 227)
        positions: (N, 3) camera positions in world frame
        rotations: list of Rotation objects
        total_frames: Total number of robot images (~1200)
        subsample_step: Every Nth frame was used for COLMAP (7)

    Returns:
        full_frame_indices: array of frame indices in the original robot sequence
        full_positions: (M, 3) interpolated positions
    """
    # The COLMAP frame names are frame_XXXXXX.jpg. The frame index from the filename
    # already encodes which of the 228 subsampled images it is (0..227).
    # These correspond to robot frames: colmap_frame_idx * subsample_step
    robot_frame_indices = colmap_frame_indices * subsample_step

    # Ensure strictly increasing (sort and deduplicate)
    sort_order = np.argsort(robot_frame_indices)
    robot_frame_indices = robot_frame_indices[sort_order]
    positions = positions[sort_order]

    # Remove duplicates
    _, unique_idx = np.unique(robot_frame_indices, return_index=True)
    robot_frame_indices = robot_frame_indices[unique_idx]
    positions = positions[unique_idx]

    # Cubic spline interpolation of positions (only interpolate, never extrapolate)
    cs_x = CubicSpline(robot_frame_indices, positions[:, 0])
    cs_y = CubicSpline(robot_frame_indices, positions[:, 1])
    cs_z = CubicSpline(robot_frame_indices, positions[:, 2])

    # Only interpolate within the range covered by COLMAP keyframes
    min_frame = int(robot_frame_indices[0])
    max_frame = min(int(robot_frame_indices[-1]), total_frames - 1)
    full_frame_indices = np.arange(min_frame, max_frame + 1)

    full_positions = np.column_stack([
        cs_x(full_frame_indices),
        cs_y(full_frame_indices),
        cs_z(full_frame_indices),
    ])

    return full_frame_indices, full_positions


def compute_cumulative_arc_length(positions: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length along a trajectory."""
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.zeros(len(positions))
    cumulative[1:] = np.cumsum(segment_lengths)
    return cumulative


def resample_by_distance(
    frame_indices: np.ndarray,
    positions: np.ndarray,
    num_samples: int = None,
    target_spacing: float = None,
) -> tuple:
    """
    Resample trajectory at equal arc-length intervals.

    Args:
        frame_indices: array of frame indices
        positions: (N, 3) positions
        num_samples: desired number of evenly-spaced samples (if provided)
        target_spacing: desired distance between consecutive samples (if provided)
            If neither is given, defaults to num_samples = len(frame_indices) // 5

    Returns:
        selected_frame_indices: frame indices of selected frames
        selected_positions: (M, 3) positions at selected frames
        arc_lengths_at_samples: arc length values at each sample
    """
    cumulative_arc = compute_cumulative_arc_length(positions)
    total_length = cumulative_arc[-1]

    if target_spacing is not None:
        num_samples = int(total_length / target_spacing) + 1
    elif num_samples is None:
        num_samples = max(len(frame_indices) // 5, 2)

    # Evenly spaced arc-length values
    target_arc_lengths = np.linspace(0, total_length, num_samples)

    # Find the frame index closest to each target arc length
    selected_indices = []
    selected_positions_list = []
    for s in target_arc_lengths:
        idx = np.argmin(np.abs(cumulative_arc - s))
        selected_indices.append(idx)
        selected_positions_list.append(positions[idx])

    selected_indices = np.array(selected_indices)
    selected_frame_indices = frame_indices[selected_indices]
    selected_positions_arr = np.array(selected_positions_list)

    return selected_frame_indices, selected_positions_arr, target_arc_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Resample a COLMAP trajectory at equal arc-length intervals."
    )
    parser.add_argument(
        "--images_txt",
        type=str,
        default="/workspace/colmap/cheezit/sparse/0_txt/images.txt",
        help="Path to COLMAP images.txt",
    )
    parser.add_argument(
        "--total_frames",
        type=int,
        default=1200,
        help="Total number of robot images in the full sequence",
    )
    parser.add_argument(
        "--subsample_step",
        type=int,
        default=7,
        help="Every Nth frame was used for COLMAP",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of evenly-spaced output samples (mutually exclusive with --target_spacing)",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        default=None,
        help="Target distance between consecutive output frames (in world units)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for selected frame indices (one per line). If not set, prints to stdout.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a 3D plot of the original and resampled trajectory",
    )
    parser.add_argument(
        "--full_images_dir",
        type=str,
        default="/workspace/nerf_comparsions/nerf_paper/cheezit_images/images_full",
        help="Directory containing the full set of robot images (1200+)",
    )
    parser.add_argument(
        "--save_images",
        type=str,
        default=None,
        help="Output directory to copy/symlink the selected frame images into",
    )
    args = parser.parse_args()

    # --- Step 1: Parse COLMAP data ---
    print(f"Parsing COLMAP images from: {args.images_txt}")
    images = parse_colmap_images_txt(args.images_txt)
    print(f"  Found {len(images)} COLMAP images")

    # --- Step 2: Build trajectory ---
    colmap_frame_indices, positions, rotations = build_trajectory(images)
    print(f"  COLMAP frame range: {colmap_frame_indices[0]} - {colmap_frame_indices[-1]}")

    # Compute distances between consecutive COLMAP frames
    colmap_arc = compute_cumulative_arc_length(positions)
    print(f"  Total trajectory length (COLMAP sparse): {colmap_arc[-1]:.4f} world units")

    # --- Step 3: Interpolate to full frame set ---
    print(f"\nInterpolating to full {args.total_frames}-frame robot sequence (step={args.subsample_step})...")
    full_frame_indices, full_positions = interpolate_trajectory(
        colmap_frame_indices, positions, rotations,
        args.total_frames, args.subsample_step,
    )
    full_arc = compute_cumulative_arc_length(full_positions)
    print(f"  Interpolated frame range: {full_frame_indices[0]} - {full_frame_indices[-1]}")
    print(f"  Total trajectory length (interpolated): {full_arc[-1]:.4f} world units")

    # --- Step 4: Resample by equal distance ---
    print(f"\nResampling by equal arc-length...")
    selected_frames, selected_positions, sample_arcs = resample_by_distance(
        full_frame_indices, full_positions,
        num_samples=args.num_samples,
        target_spacing=args.target_spacing,
    )

    spacing = np.diff(sample_arcs)
    print(f"  Number of resampled frames: {len(selected_frames)}")
    print(f"  Spacing between frames: {spacing[0]:.4f} world units (uniform)")
    print(f"  Selected robot frame indices (first 20): {selected_frames[:20].tolist()}")

    # Verify uniform spacing in position space
    actual_dists = np.linalg.norm(np.diff(selected_positions, axis=0), axis=1)
    print(f"  Actual distance stats: mean={actual_dists.mean():.4f}, "
          f"std={actual_dists.std():.4f}, min={actual_dists.min():.4f}, max={actual_dists.max():.4f}")

    # --- Step 5: Output ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Resampled trajectory: {len(selected_frames)} frames at equal distance\n")
            f.write(f"# Spacing: {spacing[0]:.6f} world units\n")
            f.write(f"# Format: robot_frame_index  x  y  z\n")
            for frame_idx, pos in zip(selected_frames, selected_positions):
                f.write(f"{frame_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        print(f"\nSaved to: {args.output}")
    else:
        print(f"\n--- Resampled Frame Indices (robot frames) ---")
        for frame_idx, pos in zip(selected_frames, selected_positions):
            print(f"  frame {frame_idx:5d}  pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")

    # --- Step 6: Save selected images ---
    if args.save_images:
        save_dir = Path(args.save_images)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_img_dir = Path(args.full_images_dir)

        if not full_img_dir.exists():
            print(f"\nWARNING: Full images directory not found: {full_img_dir}")
        else:
            print(f"\nCopying {len(selected_frames)} selected images to: {save_dir}")
            copied = 0
            for i, frame_idx in enumerate(selected_frames):
                src = full_img_dir / f"frame_{frame_idx:06d}.jpg"
                if not src.exists():
                    # Try .png
                    src = full_img_dir / f"frame_{frame_idx:06d}.png"
                if src.exists():
                    # Rename with sequential index so they sort correctly
                    dst = save_dir / f"{i:04d}_frame_{frame_idx:06d}{src.suffix}"
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    print(f"  Missing: frame_{frame_idx:06d}.*")
            print(f"  Copied {copied}/{len(selected_frames)} images")

    # --- Optional visualization ---
    if args.visualize:
        # --- Interactive 3D with Plotly ---
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Full interpolated trajectory (thin blue line)
            fig.add_trace(go.Scatter3d(
                x=full_positions[::3, 0], y=full_positions[::3, 1], z=full_positions[::3, 2],
                mode="lines",
                line=dict(color="royalblue", width=2),
                name=f"Interpolated trajectory ({len(full_positions)} pts)",
                opacity=0.4,
            ))

            # COLMAP keyframes (small blue dots)
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name=f"COLMAP keyframes ({len(positions)})",
                text=[f"COLMAP frame {i}" for i in colmap_frame_indices],
            ))

            # Resampled frames (red crosses)
            fig.add_trace(go.Scatter3d(
                x=selected_positions[:, 0], y=selected_positions[:, 1], z=selected_positions[:, 2],
                mode="markers",
                marker=dict(size=5, color="red", symbol="x"),
                name=f"Resampled ({len(selected_positions)})",
                text=[f"Robot frame {f}\nPos: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                      for f, p in zip(selected_frames, selected_positions)],
            ))

            # Lines connecting consecutive resampled points (to see spacing)
            fig.add_trace(go.Scatter3d(
                x=selected_positions[:, 0], y=selected_positions[:, 1], z=selected_positions[:, 2],
                mode="lines",
                line=dict(color="red", width=1, dash="dot"),
                name="Resampled path",
                opacity=0.5,
                showlegend=False,
            ))

            fig.update_layout(
                title="Camera Trajectory — Equal-Distance Resampling",
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                    aspectmode="data",
                ),
                width=1000, height=700,
                legend=dict(x=0.01, y=0.99),
            )

            html_path = str(Path(args.images_txt).parent.parent / "trajectory_3d.html")
            fig.write_html(html_path)
            print(f"\nInteractive 3D plot saved to: {html_path}")
            print("  Open in browser to rotate/zoom/pan.")

        except ImportError:
            print("plotly not available. Install with: python3 -m pip install plotly")

        # --- Static matplotlib fallback ---
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(121, projection="3d")
            ax1.plot(full_positions[:, 0], full_positions[:, 1], full_positions[:, 2],
                     "b-", alpha=0.3, label="Full interpolated traj")
            ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c="blue", s=5, alpha=0.5, label=f"COLMAP ({len(positions)})")
            ax1.scatter(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2],
                        c="red", s=20, marker="x", label=f"Resampled ({len(selected_positions)})")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.legend()
            ax1.set_title("3D Trajectory")

            ax2 = fig.add_subplot(122)
            ax2.plot(full_frame_indices, full_arc, "b-", alpha=0.5, label="Arc length vs frame")
            resampled_arc_indices = []
            for sf in selected_frames:
                idx_in_full = np.searchsorted(full_frame_indices, sf)
                idx_in_full = min(idx_in_full, len(full_arc) - 1)
                resampled_arc_indices.append(full_arc[idx_in_full])
            ax2.scatter(selected_frames, resampled_arc_indices,
                        c="red", marker="x", s=40, label="Resampled frames")
            ax2.set_xlabel("Robot frame index")
            ax2.set_ylabel("Cumulative arc length")
            ax2.legend()
            ax2.set_title("Arc Length Parameterization")

            plt.tight_layout()
            png_path = str(Path(args.images_txt).parent.parent / "trajectory_resampling.png")
            plt.savefig(png_path, dpi=150)
            print(f"Static plot saved to: {png_path}")
        except ImportError:
            print("matplotlib not available, skipping static plot")


if __name__ == "__main__":
    main()
