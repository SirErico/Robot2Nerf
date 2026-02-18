#!/usr/bin/env python3

import os
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class NerfDataCollector(Node):
    """
    ROS2 Node that extracts images from a rosbag and saves them to a folder.
    """

    def __init__(self):
        super().__init__('nerf_data_collector')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_dir', 'nerf_data')
        self.declare_parameter('collection_rate', 10)

        # Get parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.collection_rate = self.get_parameter('collection_rate').value

        # Initialize
        self.bridge = CvBridge()
        self.frame_count = 0
        self.i = 0

        # Create output directory
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # Subscriber
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        self.get_logger().info(f"NeRF Data Collector initialized")
        self.get_logger().info(f"Saving every {self.collection_rate}th frame to: {self.images_dir}")

    def image_callback(self, msg):
        """Save every nth image to disk."""
        if self.i % self.collection_rate == 0:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                image_filename = f"frame_{self.frame_count:06d}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                cv2.imwrite(image_path, cv_img)
                self.frame_count += 1
                self.get_logger().info(f"Saved frame {self.frame_count}: {image_filename}")
            except Exception as e:
                self.get_logger().warn(f"Failed to save frame: {str(e)}")

        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    try:
        collector = NerfDataCollector()
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info(f"Collection stopped. Saved {collector.frame_count} frames to {collector.images_dir}")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()

