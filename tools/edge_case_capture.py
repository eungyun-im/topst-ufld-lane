#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32


class EdgeCaseCapture(Node):
    def __init__(self, args):
        super().__init__('edge_case_capture')
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.latest_raw = None
        self.latest_overlay = None
        self.latest_bev = None
        self.latest_offset = None
        self.latest_heading = None
        self.latest_state = None
        self.latest_ufld_debug = None

        self.prev_valid = None
        self.false_started_at = None
        self.last_false_sample_at = 0.0
        self.saved_count = 0

        self.create_subscription(
            CompressedImage, args.raw_topic, self.on_raw_image, qos_profile_sensor_data
        )
        self.create_subscription(
            CompressedImage, args.overlay_topic, self.on_overlay_image, qos_profile_sensor_data
        )
        self.create_subscription(
            CompressedImage, args.bev_topic, self.on_bev_image, qos_profile_sensor_data
        )
        self.create_subscription(Bool, args.valid_topic, self.on_valid, 10)
        self.create_subscription(Float32, args.offset_topic, self.on_offset, 10)
        self.create_subscription(Float32, args.heading_topic, self.on_heading, 10)
        self.create_subscription(Int32, args.mission_topic, self.on_mission_state, 10)
        self.create_subscription(Float32MultiArray, args.ufld_debug_topic, self.on_ufld_debug, 10)

        self.get_logger().info(
            'edge capture started: '
            f'raw={args.raw_topic}, valid={args.valid_topic}, out={self.output_dir}'
        )

    @staticmethod
    def decode_compressed(msg):
        arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def on_raw_image(self, msg):
        frame = self.decode_compressed(msg)
        if frame is not None:
            self.latest_raw = frame

    def on_overlay_image(self, msg):
        frame = self.decode_compressed(msg)
        if frame is not None:
            self.latest_overlay = frame

    def on_bev_image(self, msg):
        frame = self.decode_compressed(msg)
        if frame is not None:
            self.latest_bev = frame

    def on_offset(self, msg):
        self.latest_offset = float(msg.data)

    def on_heading(self, msg):
        self.latest_heading = float(msg.data)

    def on_mission_state(self, msg):
        self.latest_state = int(msg.data)

    def on_ufld_debug(self, msg):
        self.latest_ufld_debug = [float(v) for v in msg.data]

    def on_valid(self, msg):
        valid = bool(msg.data)
        now = time.monotonic()

        if self.prev_valid is None:
            self.prev_valid = valid
            if not valid:
                self.false_started_at = now
                self.save_snapshot('initial_false', now, valid)
            return

        if self.prev_valid and not valid:
            self.false_started_at = now
            self.last_false_sample_at = now
            self.save_snapshot('falling_edge', now, valid)
        elif not self.prev_valid and not valid:
            false_age = 0.0 if self.false_started_at is None else now - self.false_started_at
            can_sample = now - self.last_false_sample_at >= self.args.false_sample_sec
            within_window = false_age <= self.args.max_false_sec
            if can_sample and within_window:
                self.last_false_sample_at = now
                self.save_snapshot('false_sample', now, valid, false_age=false_age)
        elif not self.prev_valid and valid:
            false_age = 0.0 if self.false_started_at is None else now - self.false_started_at
            self.save_snapshot('recovery_edge', now, valid, false_age=false_age)
            self.false_started_at = None

        self.prev_valid = valid

    def save_snapshot(self, reason, now_monotonic, valid, false_age=None):
        if self.latest_raw is None:
            self.get_logger().warning(f'skip {reason}: no raw frame yet')
            return

        wall_ns = time.time_ns()
        stem = f'{wall_ns}_{reason}'
        raw_path = self.output_dir / f'{stem}_raw.jpg'
        meta_path = self.output_dir / f'{stem}.json'

        cv2.imwrite(str(raw_path), self.latest_raw)
        overlay_path = None
        bev_path = None
        if self.args.save_debug_images and self.latest_overlay is not None:
            overlay_path = self.output_dir / f'{stem}_overlay.jpg'
            cv2.imwrite(str(overlay_path), self.latest_overlay)
        if self.args.save_debug_images and self.latest_bev is not None:
            bev_path = self.output_dir / f'{stem}_bev.jpg'
            cv2.imwrite(str(bev_path), self.latest_bev)

        meta = {
            'reason': reason,
            'valid': bool(valid),
            'false_age_sec': false_age,
            'wall_time_ns': wall_ns,
            'monotonic_sec': now_monotonic,
            'raw_image': raw_path.name,
            'overlay_image': None if overlay_path is None else overlay_path.name,
            'bev_image': None if bev_path is None else bev_path.name,
            'offset': self.latest_offset,
            'heading': self.latest_heading,
            'mission_state': self.latest_state,
            'ufld_debug': self.latest_ufld_debug,
            'topics': {
                'raw': self.args.raw_topic,
                'overlay': self.args.overlay_topic,
                'bev': self.args.bev_topic,
                'valid': self.args.valid_topic,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

        self.saved_count += 1
        self.get_logger().info(
            f'saved #{self.saved_count} {reason}: {raw_path.name}'
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Capture UFLD invalid/recovery edge-case frames.')
    parser.add_argument('--output-dir', default='~/ufld_edge_cases')
    parser.add_argument('--raw-topic', default='/camera/image/compressed')
    parser.add_argument('--overlay-topic', default='/ufld/image/compressed')
    parser.add_argument('--bev-topic', default='/line/image/bev')
    parser.add_argument('--valid-topic', default='/line_valid')
    parser.add_argument('--offset-topic', default='/line_offset')
    parser.add_argument('--heading-topic', default='/line_heading')
    parser.add_argument('--mission-topic', default='/mission_state')
    parser.add_argument('--ufld-debug-topic', default='/ufld_debug')
    parser.add_argument('--false-sample-sec', type=float, default=2.0)
    parser.add_argument('--max-false-sec', type=float, default=6.0)
    parser.add_argument('--save-debug-images', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = EdgeCaseCapture(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
