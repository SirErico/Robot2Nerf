#!/usr/bin/env python3

import os
import glob
import struct
import shutil

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# PointField datatype sizes
_PFTYPE_TO_NPTYPE = {
    PointField.INT8:    np.int8,
    PointField.UINT8:   np.uint8,
    PointField.INT16:   np.int16,
    PointField.UINT16:  np.uint16,
    PointField.INT32:   np.int32,
    PointField.UINT32:  np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def pointcloud2_to_xyz_rgb_fast(msg: PointCloud2):
    """
    Fast vectorized conversion of PointCloud2 to numpy xyz and rgb arrays.
    """
    field_map = {f.name: f for f in msg.fields}
    for name in ('x', 'y', 'z'):
        if name not in field_map:
            raise ValueError(f"PointCloud2 missing required field '{name}'")

    n_points = msg.width * msg.height
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_points, point_step)

    # Extract xyz
    xyz = np.empty((n_points, 3), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        f = field_map[name]
        offset = f.offset
        xyz[:, i] = np.frombuffer(
            data[:, offset:offset + 4].tobytes(), dtype=np.float32
        )

    # Extract rgb if available
    rgb = None
    if 'rgb' in field_map or 'rgba' in field_map:
        rgb_field = field_map.get('rgb', field_map.get('rgba'))
        offset = rgb_field.offset
        rgba_bytes = data[:, offset:offset + 4]
        rgb = np.empty((n_points, 3), dtype=np.uint8)
        rgb[:, 0] = rgba_bytes[:, 2]  # R
        rgb[:, 1] = rgba_bytes[:, 1]  # G
        rgb[:, 2] = rgba_bytes[:, 0]  # B

    return xyz, rgb


def write_ply(filepath, xyz, rgb=None):
    """Write a point cloud as a binary PLY file."""
    n = xyz.shape[0]
    has_color = rgb is not None

    with open(filepath, 'wb') as f:
        # Header
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        if has_color:
            header += "property uchar red\n"
            header += "property uchar green\n"
            header += "property uchar blue\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        # Data
        if has_color:
            for i in range(n):
                f.write(struct.pack('<fff', xyz[i, 0], xyz[i, 1], xyz[i, 2]))
                f.write(struct.pack('<BBB', rgb[i, 0], rgb[i, 1], rgb[i, 2]))
        else:
            for i in range(n):
                f.write(struct.pack('<fff', xyz[i, 0], xyz[i, 1], xyz[i, 2]))


def write_ply_fast(filepath, xyz, rgb=None):
    """Write a point cloud as a binary PLY file (vectorized)."""
    # Filter out NaN / Inf points
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    if rgb is not None:
        rgb = rgb[valid]

    n = xyz.shape[0]
    has_color = rgb is not None

    with open(filepath, 'wb') as f:
        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        if has_color:
            header += "property uchar red\n"
            header += "property uchar green\n"
            header += "property uchar blue\n"
        header += "end_header\n"
        f.write(header.encode('ascii'))

        if has_color:
            # Interleave xyz (float32) and rgb (uint8) per point
            record_dt = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
            ])
            records = np.empty(n, dtype=record_dt)
            records['x'] = xyz[:, 0]
            records['y'] = xyz[:, 1]
            records['z'] = xyz[:, 2]
            records['r'] = rgb[:, 0]
            records['g'] = rgb[:, 1]
            records['b'] = rgb[:, 2]
            f.write(records.tobytes())
        else:
            f.write(xyz.astype('<f4').tobytes())


class PointCloudCollector(Node):
    """
    ROS2 Node that subscribes to a PointCloud2 topic (e.g. from a depth camera)
    and saves individual point cloud frames as PLY files.
    """

    def __init__(self):
        super().__init__('pointcloud_collector')

        # Declare parameters
        self.declare_parameter('pointcloud_topic', '/camera/depth_registered/points')
        self.declare_parameter('output_dir', 'pointcloud_data')
        self.declare_parameter('collection_rate', 10)
        self.declare_parameter('merge', False)
        self.declare_parameter('rotate_180', False)

        # Get parameters
        self.pc_topic = self.get_parameter('pointcloud_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.collection_rate = self.get_parameter('collection_rate').value
        self.merge = self.get_parameter('merge').value
        self.rotate_180 = self.get_parameter('rotate_180').value

        # Counters
        self.msg_count = 0
        self.frame_count = 0

        # For merging all frames into one cloud
        self.all_xyz = []
        self.all_rgb = []

        # Create / clear output directory
        self.ply_dir = os.path.join(self.output_dir, "pointclouds")
        if os.path.exists(self.ply_dir):
            try:
                for entry in os.listdir(self.ply_dir):
                    path = os.path.join(self.ply_dir, entry)
                    if os.path.isfile(path) or os.path.islink(path):
                        os.unlink(path)
                    else:
                        shutil.rmtree(path)
            except Exception as e:
                self.get_logger().warn(f"Failed to clear output directory: {e}")
        os.makedirs(self.ply_dir, exist_ok=True)

        # Subscribe to PointCloud2 topic
        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self.pointcloud_callback, 10
        )

        self.get_logger().info(f"PointCloud Collector initialized")
        self.get_logger().info(f"Topic: {self.pc_topic}")
        self.get_logger().info(f"Saving every {self.collection_rate}th frame to: {self.ply_dir}")
        if self.merge:
            self.get_logger().info("Merge mode ON — will also save a combined cloud on shutdown")

    def pointcloud_callback(self, msg: PointCloud2):
        """Process and save every nth PointCloud2 message as a PLY file."""
        if self.msg_count % self.collection_rate == 0:
            try:
                xyz, rgb = pointcloud2_to_xyz_rgb_fast(msg)

                # Rotate 180° around Z axis (negate x and y)
                if self.rotate_180:
                    xyz[:, 0] = -xyz[:, 0]
                    xyz[:, 1] = -xyz[:, 1]

                # Save individual frame
                filename = f"cloud_{self.frame_count:06d}.ply"
                filepath = os.path.join(self.ply_dir, filename)
                write_ply_fast(filepath, xyz, rgb)

                n_valid = np.isfinite(xyz).all(axis=1).sum()
                self.get_logger().info(
                    f"Saved frame {self.frame_count}: {filename} "
                    f"({n_valid} valid points)"
                )

                # Accumulate for merging
                if self.merge:
                    valid = np.isfinite(xyz).all(axis=1)
                    self.all_xyz.append(xyz[valid])
                    if rgb is not None:
                        self.all_rgb.append(rgb[valid])

                self.frame_count += 1

            except Exception as e:
                self.get_logger().error(f"Failed to process point cloud: {str(e)}")

        self.msg_count += 1

    def save_merged(self):
        """Save all accumulated frames as a single merged PLY file using ICP registration."""
        if not self.all_xyz:
            self.get_logger().warn("No data to merge")
            return

        if not HAS_OPEN3D:
            self.get_logger().warn(
                "Open3D not installed — falling back to naive concatenation (no alignment). "
                "Install with: pip install open3d"
            )
            merged_xyz = np.concatenate(self.all_xyz, axis=0)
            merged_rgb = np.concatenate(self.all_rgb, axis=0) if self.all_rgb else None
            merged_path = os.path.join(self.output_dir, "merged_cloud.ply")
            write_ply_fast(merged_path, merged_xyz, merged_rgb)
            self.get_logger().info(
                f"Saved merged cloud (unaligned): {merged_path} ({merged_xyz.shape[0]} points)"
            )
            return

        self.get_logger().info(
            f"Merging {len(self.all_xyz)} frames with ICP registration..."
        )
        merged_path = os.path.join(self.output_dir, "merged_cloud.ply")
        merge_pointclouds_icp(
            ply_dir=self.ply_dir,
            output_path=merged_path,
            voxel_size=0.005,
            icp_threshold=0.02,
            nb_neighbors=20,
            std_ratio=2.0,
            logger=self.get_logger(),
        )


# ---------------------------------------------------------------------------
# ICP-based point cloud registration & merging (requires Open3D)
# ---------------------------------------------------------------------------

def merge_pointclouds_icp(
    ply_dir: str,
    output_path: str,
    voxel_size: float = 0.005,
    icp_threshold: float = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    logger=None,
):
    """
    Load all PLY files from *ply_dir*, align them with pairwise ICP,
    and save a single dense merged cloud to *output_path*.

    Parameters
    ----------
    ply_dir : str
        Directory containing individual cloud_XXXXXX.ply files.
    output_path : str
        Where to write the merged PLY.
    voxel_size : float
        Voxel size (m) for down-sampling before ICP. Smaller = denser but slower.
    icp_threshold : float
        Max correspondence distance for ICP (m).
    nb_neighbors : int
        Statistical outlier removal — number of neighbours to analyse.
    std_ratio : float
        Statistical outlier removal — standard-deviation multiplier.
    logger
        Optional ROS2 logger; uses print() as fallback.
    """
    if not HAS_OPEN3D:
        raise RuntimeError("Open3D is required for ICP merging. pip install open3d")

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    if not ply_files:
        log("No PLY files found — nothing to merge.")
        return

    log(f"Found {len(ply_files)} PLY files to merge")

    # Load first cloud as the reference
    merged = o3d.io.read_point_cloud(ply_files[0])
    merged = merged.voxel_down_sample(voxel_size)

    cumulative_transform = np.eye(4)

    for i in range(1, len(ply_files)):
        source = o3d.io.read_point_cloud(ply_files[i])
        source = source.voxel_down_sample(voxel_size)

        if len(source.points) == 0:
            log(f"  [{i}/{len(ply_files)-1}] Skipped (empty cloud)")
            continue

        # Estimate normals (needed for point-to-plane ICP)
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
        )
        merged.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
        )

        # Run ICP (point-to-plane)
        reg = o3d.pipelines.registration.registration_icp(
            source,
            merged,
            icp_threshold,
            cumulative_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )

        cumulative_transform = reg.transformation

        # Transform source into merged frame and combine
        source.transform(reg.transformation)
        merged = merged + source

        # Periodically downsample to keep memory in check
        if i % 10 == 0:
            merged = merged.voxel_down_sample(voxel_size)

        fitness = reg.fitness
        rmse = reg.inlier_rmse
        log(f"  [{i}/{len(ply_files)-1}] fitness={fitness:.4f}  RMSE={rmse:.6f}  "
            f"points={len(merged.points)}")

    # Final voxel downsample
    log("Final voxel downsample...")
    merged = merged.voxel_down_sample(voxel_size)

    # Statistical outlier removal
    log("Removing outliers...")
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    log(f"Merged cloud: {len(merged.points)} points")
    o3d.io.write_point_cloud(output_path, merged, write_ascii=False)
    log(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# Standalone merge entry point  (ros2 run mlinpl merge_pointclouds)
# ---------------------------------------------------------------------------

def merge_main():
    """
    Standalone CLI to merge previously saved PLY frames using ICP.

    Usage:
        ros2 run mlinpl merge_pointclouds
        # or directly:
        python3 -m mlinpl.pointcloud_maker --merge \
            --ply-dir pointcloud_data/pointclouds \
            --output  pointcloud_data/merged_cloud.ply
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge individual PLY point clouds using ICP registration"
    )
    parser.add_argument(
        "--ply-dir", default="pointcloud_data/pointclouds",
        help="Directory containing cloud_XXXXXX.ply files (default: pointcloud_data/pointclouds)",
    )
    parser.add_argument(
        "--output", default="pointcloud_data/merged_cloud.ply",
        help="Output merged PLY path (default: pointcloud_data/merged_cloud.ply)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.005,
        help="Voxel size in meters for downsampling (default: 0.005)",
    )
    parser.add_argument(
        "--icp-threshold", type=float, default=0.02,
        help="ICP max correspondence distance in meters (default: 0.02)",
    )
    parser.add_argument(
        "--nb-neighbors", type=int, default=20,
        help="Statistical outlier nb_neighbors (default: 20)",
    )
    parser.add_argument(
        "--std-ratio", type=float, default=2.0,
        help="Statistical outlier std_ratio (default: 2.0)",
    )
    args = parser.parse_args()

    merge_pointclouds_icp(
        ply_dir=args.ply_dir,
        output_path=args.output,
        voxel_size=args.voxel_size,
        icp_threshold=args.icp_threshold,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )


def main(args=None):
    rclpy.init(args=args)
    collector = PointCloudCollector()

    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info(
            f"Collection stopped. Saved {collector.frame_count} frames to {collector.ply_dir}"
        )
        if collector.merge:
            collector.save_merged()
    finally:
        collector.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

