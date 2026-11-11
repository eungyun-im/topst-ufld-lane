#!/usr/bin/env python3

import time

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


class YoloOrtNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        cv2.setNumThreads(2)

        self.image_topic = self.declare_parameter('image_topic', '/image_jpeg/compressed').value
        self.green_topic = self.declare_parameter('traffic_light_green_topic', '/traffic_light_green').value
        self.red_topic = self.declare_parameter('traffic_light_red_topic', '/traffic_light_red').value
        self.aruco_topic = self.declare_parameter('aruco_flag_topic', '/aruco_flag').value

        self.model_path = self.declare_parameter('model_path', '').value
        self.conf_threshold = float(self.declare_parameter('conf_threshold', 0.5).value)
        self.nms_threshold = float(self.declare_parameter('nms_threshold', 0.4).value)
        self.input_width = int(self.declare_parameter('input_width', 320).value)
        self.input_height = int(self.declare_parameter('input_height', 320).value)
        self.class_green = int(self.declare_parameter('class_green', 0).value)
        self.class_red = int(self.declare_parameter('class_red', 1).value)

        self.aruco_dict_id = int(self.declare_parameter('aruco_dict', 8).value)
        self.target_aruco_id = int(self.declare_parameter('target_aruco_id', 3).value)
        self.min_marker_size = float(self.declare_parameter('min_marker_size', 30.0).value)

        self.aruco_corner_refine = bool(self.declare_parameter('aruco_corner_refine', True).value)
        self.aruco_win_min = int(self.declare_parameter('aruco_win_min', 3).value)
        self.aruco_win_max = int(self.declare_parameter('aruco_win_max', 23).value)
        self.aruco_win_step = int(self.declare_parameter('aruco_win_step', 4).value)
        self.aruco_thresh_constant = float(self.declare_parameter('aruco_thresh_constant', 7.0).value)
        self.aruco_min_perimeter = float(self.declare_parameter('aruco_min_perimeter', 0.02).value)
        self.aruco_max_perimeter = float(self.declare_parameter('aruco_max_perimeter', 4.0).value)
        self.aruco_poly_accuracy = float(self.declare_parameter('aruco_poly_accuracy', 0.05).value)
        self.aruco_error_correction = float(self.declare_parameter('aruco_error_correction', 0.6).value)
        self.aruco_equalize = bool(self.declare_parameter('aruco_equalize', True).value)

        self.show_debug = bool(self.declare_parameter('show_debug', False).value)
        self.publish_debug_images = bool(self.declare_parameter('publish_debug_images', True).value)
        self.debug_topic_prefix = self.declare_parameter('debug_topic_prefix', '/yolo_debug').value
        self.debug_jpeg_quality = int(self.declare_parameter('debug_jpeg_quality', 70).value)
        self.debug_image_skip = max(1, int(self.declare_parameter('debug_image_skip', 1).value))
        self.debug_jpeg_quality = max(1, min(self.debug_jpeg_quality, 100))

        self.traffic_roi_x = int(self.declare_parameter('traffic_roi_x', 0).value)
        self.traffic_roi_y = int(self.declare_parameter('traffic_roi_y', 0).value)
        self.traffic_roi_width = int(self.declare_parameter('traffic_roi_width', 0).value)
        self.traffic_roi_height = int(self.declare_parameter('traffic_roi_height', 0).value)

        self.aruco_roi_x = int(self.declare_parameter('aruco_roi_x', 0).value)
        self.aruco_roi_y = int(self.declare_parameter('aruco_roi_y', 0).value)
        self.aruco_roi_width = int(self.declare_parameter('aruco_roi_width', 0).value)
        self.aruco_roi_height = int(self.declare_parameter('aruco_roi_height', 0).value)

        self.frame = None
        self.frame_seq = 0
        self.processed_seq = 0
        self.aruco_visible = False
        self.debug_image_counter = 0
        self.last_log_time = 0.0

        self.session = None
        self.input_name = ''
        self.output_names = []
        self.load_model()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_cb, image_qos)

        self.green_pub = self.create_publisher(Bool, self.green_topic, 10)
        self.red_pub = self.create_publisher(Bool, self.red_topic, 10)
        self.aruco_pub = self.create_publisher(Bool, self.aruco_topic, 10)

        self.debug_vis_pub = None
        if self.publish_debug_images:
            self.debug_vis_pub = self.create_publisher(
                CompressedImage, self.debug_topic_prefix + '/detection', image_qos)

        self.dictionary = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        self.detector_params = self.build_detector_params()
        # OpenCV >= 4.7 removed the free-function aruco API; use ArucoDetector there.
        self.aruco_detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        if self.show_debug:
            cv2.namedWindow('YOLO + ArUco', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLO + ArUco', 640, 480)

        self.timer = self.create_timer(0.01, self.run)
        model_status = self.model_path if self.session is not None else 'NOT LOADED'
        self.get_logger().info(
            f'[yolo_pkg] Started with ONNX Runtime. target_aruco_id={self.target_aruco_id} model={model_status}')

    def build_detector_params(self):
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            params = cv2.aruco.DetectorParameters_create()
        else:
            params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = self.aruco_win_min
        params.adaptiveThreshWinSizeMax = self.aruco_win_max
        params.adaptiveThreshWinSizeStep = self.aruco_win_step
        params.adaptiveThreshConstant = self.aruco_thresh_constant
        params.minMarkerPerimeterRate = self.aruco_min_perimeter
        params.maxMarkerPerimeterRate = self.aruco_max_perimeter
        params.polygonalApproxAccuracyRate = self.aruco_poly_accuracy
        params.errorCorrectionRate = self.aruco_error_correction
        if self.aruco_corner_refine:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return params

    def detect_aruco(self, image):
        if self.aruco_equalize:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.equalizeHist(gray)
        if self.aruco_detector is not None:
            return self.aruco_detector.detectMarkers(image)
        return cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.detector_params)

    def load_model(self):
        if not self.model_path:
            self.get_logger().warning('[yolo_pkg] No model path set. YOLO detection disabled.')
            return

        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=['CPUExecutionProvider'],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.get_logger().info(f'[yolo_pkg] YOLO model loaded by ONNX Runtime: {self.model_path}')
        except Exception as exc:
            self.session = None
            self.get_logger().error(f'[yolo_pkg] Failed to load model with ONNX Runtime: {exc}')

    def image_cb(self, msg):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            self.frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            self.frame_seq += 1
        except Exception as exc:
            self.get_logger().warning(f'[yolo_pkg] Image decode error: {exc}')

    def publish_debug_image(self, image):
        if self.debug_vis_pub is None or image is None or image.size == 0:
            return

        ok, encoded = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.debug_jpeg_quality])
        if not ok:
            self.get_logger().warning('[yolo_pkg] Failed to encode debug image as JPEG')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'yolo_debug'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        self.debug_vis_pub.publish(msg)

    @staticmethod
    def marker_size(corners):
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 4:
            return 0.0
        return float(sum(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)) / 4.0)

    def run_yolo(self, frame):
        if self.session is None or frame is None or frame.size == 0:
            return []

        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

        try:
            outputs = self.session.run(self.output_names, {self.input_name: blob})
        except Exception as exc:
            self.get_logger().warning(f'[yolo_pkg] ONNX Runtime inference error: {exc}')
            return []

        if not outputs:
            return []

        output = np.asarray(outputs[0])
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2 or output.shape[-1] < 6:
            self.get_logger().warning(f'[yolo_pkg] Unsupported YOLO output shape: {output.shape}')
            return []

        if output.shape[1] == 6:
            detections = self.parse_end2end_output(output, frame.shape[1], frame.shape[0])
        else:
            detections = self.parse_raw_output(output, frame.shape[1], frame.shape[0])
        return detections

    def parse_end2end_output(self, output, frame_w, frame_h):
        boxes = []
        confidences = []
        class_ids = []

        x_scale = frame_w / float(self.input_width)
        y_scale = frame_h / float(self.input_height)

        for row in output:
            conf = float(row[4])
            if conf < self.conf_threshold:
                continue

            class_id = int(round(float(row[5])))
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1 *= frame_w
                x2 *= frame_w
                y1 *= frame_h
                y2 *= frame_h
            else:
                x1 *= x_scale
                x2 *= x_scale
                y1 *= y_scale
                y2 *= y_scale

            x = int(round(min(x1, x2)))
            y = int(round(min(y1, y2)))
            w = int(round(abs(x2 - x1)))
            h = int(round(abs(y2 - y1)))
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, w, h])
            confidences.append(conf)
            class_ids.append(class_id)

        return self.apply_nms(boxes, confidences, class_ids)

    def parse_raw_output(self, output, frame_w, frame_h):
        if output.shape[0] < output.shape[1] and output.shape[0] <= 16:
            output = output.T

        boxes = []
        confidences = []
        class_ids = []
        x_scale = frame_w / float(self.input_width)
        y_scale = frame_h / float(self.input_height)

        for row in output:
            scores = row[4:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < self.conf_threshold:
                continue

            cx = float(row[0]) * x_scale
            cy = float(row[1]) * y_scale
            w = float(row[2]) * x_scale
            h = float(row[3]) * y_scale
            boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
            confidences.append(conf)
            class_ids.append(class_id)

        return self.apply_nms(boxes, confidences, class_ids)

    def apply_nms(self, boxes, confidences, class_ids):
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        detections = []
        for index in np.array(indices).flatten():
            x, y, w, h = boxes[int(index)]
            detections.append({
                'class_id': class_ids[int(index)],
                'confidence': confidences[int(index)],
                'box': [x, y, w, h],
            })
        return detections

    def make_roi(self, frame, x, y, width, height):
        use_roi = width > 0 and height > 0
        if not use_roi:
            return frame, None
        rx = max(0, x)
        ry = max(0, y)
        rw = min(width, frame.shape[1] - rx)
        rh = min(height, frame.shape[0] - ry)
        if rw <= 0 or rh <= 0:
            return frame, None
        rect = (rx, ry, rw, rh)
        return frame[ry:ry + rh, rx:rx + rw], rect

    def run(self):
        if self.frame is None or self.frame_seq == self.processed_seq:
            return

        frame = self.frame
        self.processed_seq = self.frame_seq
        vis = frame.copy()
        green_detected = False
        red_detected = False

        if self.session is not None:
            traffic_frame, traffic_roi = self.make_roi(
                frame,
                self.traffic_roi_x,
                self.traffic_roi_y,
                self.traffic_roi_width,
                self.traffic_roi_height,
            )
            if traffic_roi is not None:
                x, y, w, h = traffic_roi
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 165, 0), 2)

            detections = self.run_yolo(traffic_frame)
            for det in detections:
                class_id = det['class_id']
                x, y, w, h = det['box']
                if traffic_roi is not None:
                    x += traffic_roi[0]
                    y += traffic_roi[1]

                if class_id == self.class_green:
                    color = (0, 255, 0)
                    label = 'GREEN'
                    green_detected = True
                elif class_id == self.class_red:
                    color = (0, 0, 255)
                    label = 'RED'
                    red_detected = True
                else:
                    continue

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    vis,
                    f'{label} {int(det["confidence"] * 100)}%',
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            self.green_pub.publish(Bool(data=green_detected))
            self.red_pub.publish(Bool(data=red_detected))

        aruco_frame, aruco_roi = self.make_roi(
            frame,
            self.aruco_roi_x,
            self.aruco_roi_y,
            self.aruco_roi_width,
            self.aruco_roi_height,
        )
        if aruco_roi is not None:
            x, y, w, h = aruco_roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        corners, ids, _ = self.detect_aruco(aruco_frame)
        aruco_now = False
        if ids is not None and len(ids) > 0:
            draw_corners = corners
            if aruco_roi is not None:
                draw_corners = [corner.copy() for corner in corners]
                for corner in draw_corners:
                    corner[:, :, 0] += aruco_roi[0]
                    corner[:, :, 1] += aruco_roi[1]
            cv2.aruco.drawDetectedMarkers(vis, draw_corners, ids)
            for index, marker_id in enumerate(ids.flatten()):
                if int(marker_id) == self.target_aruco_id and self.marker_size(corners[index]) >= self.min_marker_size:
                    aruco_now = True
                    cv2.putText(
                        vis,
                        f'ARUCO id={marker_id}',
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    break

        if aruco_now != self.aruco_visible:
            self.aruco_visible = aruco_now
            self.aruco_pub.publish(Bool(data=self.aruco_visible))
            self.get_logger().warning(
                f'[yolo_pkg] ArUco id={self.target_aruco_id} -> {str(self.aruco_visible).lower()}')

        now = time.monotonic()
        if now - self.last_log_time >= 0.5:
            self.last_log_time = now
            self.get_logger().info(
                '[yolo_pkg] green=%s red=%s aruco=%s' % (
                    'ON' if green_detected else 'off',
                    'ON' if red_detected else 'off',
                    'ON' if self.aruco_visible else 'off',
                )
            )

        if self.publish_debug_images:
            self.debug_image_counter += 1
            if self.debug_image_counter >= self.debug_image_skip:
                self.debug_image_counter = 0
                self.publish_debug_image(vis)

        if self.show_debug:
            cv2.imshow('YOLO + ArUco', vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloOrtNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
