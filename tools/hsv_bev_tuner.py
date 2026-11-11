#!/usr/bin/env python3
"""
hsv_bev_tuner.py  —  로컬 PC에서 실행하는 HSV + BEV 실시간 튜너

사전 조건 (로컬 PC):
  - ROS2 Humble 설치
  - python3-opencv 설치  (sudo apt install python3-opencv)
  - D-Racer와 동일 WiFi 네트워크 연결

실행:
  source /opt/ros/humble/setup.bash
  python3 hsv_bev_tuner.py

키보드 단축키:
  a / Enter  : 현재 트랙바 값을 /line_node 파라미터로 전송
  s          : 현재 값을 line_params_saved.yaml 로 저장
  p          : 현재 값을 line.launch XML 형식으로 터미널 출력
  r          : 트랙바를 launch 파일 기본값으로 초기화
  q / Esc    : 종료

창 구성:
  [Camera + BEV ROI]  : 원본 카메라 + BEV src 폴리곤 + BEV src 트랙바
  [Lane Mask (HSV)]   : HSV 임계값 적용 결과 + HSV 트랙바
  [BEV Result]        : BEV 변환 결과 + BEV dst 트랙바
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Bool, Float32, Float32MultiArray
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False
    print('[WARN] rclpy를 찾을 수 없습니다. 로컬 데모 모드로 실행합니다.')
    print('       source /opt/ros/humble/setup.bash 후 다시 실행하세요.')

# ── 이미지 해상도 (D-Racer 카메라 320x160) ───────────────────────────
IMG_W, IMG_H = 320, 160

# ── launch 파일 기본값 ────────────────────────────────────────────────
DEFAULTS = {
    # BEV src 좌표 (원본 이미지 픽셀)
    'bev_src_bl_x': 0,   'bev_src_bl_y': 118,
    'bev_src_br_x': 320, 'bev_src_br_y': 118,
    'bev_src_tl_x': 90,  'bev_src_tl_y': 80,
    'bev_src_tr_x': 230, 'bev_src_tr_y': 80,
    # BEV dst 좌표 (변환 후 이미지 픽셀)
    'bev_dst_bl_x': 75,  'bev_dst_bl_y': 120,
    'bev_dst_br_x': 245, 'bev_dst_br_y': 120,
    'bev_dst_tl_x': 75,  'bev_dst_tl_y': 0,
    'bev_dst_tr_x': 245, 'bev_dst_tr_y': 0,
    # HSV 흰색 차선
    'white_h_min': 0,   'white_h_max': 180,
    'white_s_min': 0,   'white_s_max': 55,
    'white_v_min': 150, 'white_v_max': 255,
    # HSV 노란색 차선
    'yellow_h_min': 15,  'yellow_h_max': 30,
    'yellow_s_min': 70,  'yellow_s_max': 255,
    'yellow_v_min': 120, 'yellow_v_max': 255,
}

# ── 전역 상태 ─────────────────────────────────────────────────────────
params = dict(DEFAULTS)
latest_frame = None
latest_offset = 0.0
latest_yellow = False
frame_lock = threading.Lock()
status_msg = 'Waiting for camera...'

WIN_CAM  = 'Camera + BEV ROI'
WIN_MASK = 'Lane Mask (HSV)'
WIN_BEV  = 'BEV Result'

TRACKBAR_DEFS = {
    # (window, name, key, max_val)
    WIN_CAM: [
        ('src_bl_x', 'bev_src_bl_x', IMG_W),
        ('src_bl_y', 'bev_src_bl_y', IMG_H),
        ('src_br_x', 'bev_src_br_x', IMG_W),
        ('src_br_y', 'bev_src_br_y', IMG_H),
        ('src_tl_x', 'bev_src_tl_x', IMG_W),
        ('src_tl_y', 'bev_src_tl_y', IMG_H),
        ('src_tr_x', 'bev_src_tr_x', IMG_W),
        ('src_tr_y', 'bev_src_tr_y', IMG_H),
    ],
    WIN_BEV: [
        ('dst_bl_x', 'bev_dst_bl_x', IMG_W),
        ('dst_bl_y', 'bev_dst_bl_y', IMG_H),
        ('dst_br_x', 'bev_dst_br_x', IMG_W),
        ('dst_br_y', 'bev_dst_br_y', IMG_H),
        ('dst_tl_x', 'bev_dst_tl_x', IMG_W),
        ('dst_tl_y', 'bev_dst_tl_y', IMG_H),
        ('dst_tr_x', 'bev_dst_tr_x', IMG_W),
        ('dst_tr_y', 'bev_dst_tr_y', IMG_H),
    ],
    WIN_MASK: [
        ('W_H_min', 'white_h_min',  180),
        ('W_H_max', 'white_h_max',  180),
        ('W_S_min', 'white_s_min',  255),
        ('W_S_max', 'white_s_max',  255),
        ('W_V_min', 'white_v_min',  255),
        ('W_V_max', 'white_v_max',  255),
        ('Y_H_min', 'yellow_h_min', 180),
        ('Y_H_max', 'yellow_h_max', 180),
        ('Y_S_min', 'yellow_s_min', 255),
        ('Y_S_max', 'yellow_s_max', 255),
        ('Y_V_min', 'yellow_v_min', 255),
        ('Y_V_max', 'yellow_v_max', 255),
    ],
}


def nothing(_): pass


# ── ROS2 노드 ──────────────────────────────────────────────────────────
class TunerNode(Node):
    def __init__(self):
        super().__init__('hsv_bev_tuner')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CompressedImage, '/camera/image/compressed', self._image_cb, qos)
        self.create_subscription(Float32, '/line_offset', self._offset_cb, 10)
        self.create_subscription(Bool, '/yellow_detected', self._yellow_cb, 10)
        # /line_tuning 퍼블리셔 (Float32MultiArray 28개)
        # [0-5]:  white HSV   (h_min, h_max, s_min, s_max, v_min, v_max)
        # [6-11]: yellow HSV  (h_min, h_max, s_min, s_max, v_min, v_max)
        # [12-19]: bev_src    (bl_x, bl_y, br_x, br_y, tl_x, tl_y, tr_x, tr_y)
        # [20-27]: bev_dst    (bl_x, bl_y, br_x, br_y, tl_x, tl_y, tr_x, tr_y)
        self.tuning_pub = self.create_publisher(Float32MultiArray, '/line_tuning', 10)

    def _image_cb(self, msg):
        global latest_frame
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with frame_lock:
                latest_frame = frame

    def _offset_cb(self, msg):
        global latest_offset
        latest_offset = msg.data

    def _yellow_cb(self, msg):
        global latest_yellow
        latest_yellow = msg.data

    def publish_tuning(self):
        msg = Float32MultiArray()
        msg.data = [
            float(params['white_h_min']),  float(params['white_h_max']),
            float(params['white_s_min']),  float(params['white_s_max']),
            float(params['white_v_min']),  float(params['white_v_max']),
            float(params['yellow_h_min']), float(params['yellow_h_max']),
            float(params['yellow_s_min']), float(params['yellow_s_max']),
            float(params['yellow_v_min']), float(params['yellow_v_max']),
            float(params['bev_src_bl_x']), float(params['bev_src_bl_y']),
            float(params['bev_src_br_x']), float(params['bev_src_br_y']),
            float(params['bev_src_tl_x']), float(params['bev_src_tl_y']),
            float(params['bev_src_tr_x']), float(params['bev_src_tr_y']),
            float(params['bev_dst_bl_x']), float(params['bev_dst_bl_y']),
            float(params['bev_dst_br_x']), float(params['bev_dst_br_y']),
            float(params['bev_dst_tl_x']), float(params['bev_dst_tl_y']),
            float(params['bev_dst_tr_x']), float(params['bev_dst_tr_y']),
        ]
        self.tuning_pub.publish(msg)


def _ros_spin(node):
    rclpy.spin(node)


def apply_all_params(node):
    global status_msg
    node.publish_tuning()
    status_msg = f'[{datetime.now().strftime("%H:%M:%S")}] /line_tuning 전송 완료'
    print(status_msg)


# ── 저장 / 출력 헬퍼 ──────────────────────────────────────────────────
def save_yaml():
    global status_msg
    out_path = Path(__file__).parent / 'line_params_saved.yaml'
    lines = ['# line_node 저장된 파라미터\n']
    for k, v in params.items():
        lines.append(f'{k}: {v}\n')
    out_path.write_text(''.join(lines))
    status_msg = f'[{datetime.now().strftime("%H:%M:%S")}] 저장: {out_path}'
    print(status_msg)


def print_launch_xml():
    print('\n--- line.launch 파라미터 (복사해서 사용) ---')
    print('    <!-- BEV src -->')
    for name in ['bev_src_bl_x', 'bev_src_bl_y', 'bev_src_br_x', 'bev_src_br_y',
                 'bev_src_tl_x', 'bev_src_tl_y', 'bev_src_tr_x', 'bev_src_tr_y']:
        print(f'    <param name="{name}" type="float" value="{float(params[name]):.1f}"/>')
    print('    <!-- BEV dst -->')
    for name in ['bev_dst_bl_x', 'bev_dst_bl_y', 'bev_dst_br_x', 'bev_dst_br_y',
                 'bev_dst_tl_x', 'bev_dst_tl_y', 'bev_dst_tr_x', 'bev_dst_tr_y']:
        print(f'    <param name="{name}" type="float" value="{float(params[name]):.1f}"/>')
    print('    <!-- HSV 흰색 -->')
    for name in ['white_h_min', 'white_h_max', 'white_s_min', 'white_s_max',
                 'white_v_min', 'white_v_max']:
        print(f'    <param name="{name}" type="int" value="{int(params[name])}"/>')
    print('    <!-- HSV 노란색 -->')
    for name in ['yellow_h_min', 'yellow_h_max', 'yellow_s_min', 'yellow_s_max',
                 'yellow_v_min', 'yellow_v_max']:
        print(f'    <param name="{name}" type="int" value="{int(params[name])}"/>')
    print('-------------------------------------------\n')


# ── OpenCV 창 & 트랙바 초기화 ────────────────────────────────────────
def create_windows():
    for win in (WIN_CAM, WIN_MASK, WIN_BEV):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, IMG_W * 2, IMG_H * 2 + 30 * len(TRACKBAR_DEFS[win]))

    for win, tb_list in TRACKBAR_DEFS.items():
        for (tb_name, key, max_val) in tb_list:
            cv2.createTrackbar(tb_name, win, int(params[key]), max_val, nothing)


def read_trackbars():
    for win, tb_list in TRACKBAR_DEFS.items():
        for (tb_name, key, _) in tb_list:
            params[key] = cv2.getTrackbarPos(tb_name, win)


def reset_trackbars():
    params.update(DEFAULTS)
    for win, tb_list in TRACKBAR_DEFS.items():
        for (tb_name, key, _) in tb_list:
            cv2.setTrackbarPos(tb_name, win, int(DEFAULTS[key]))


# ── 이미지 처리 (로컬 재현) ───────────────────────────────────────────
def _bev_matrix():
    src = np.float32([
        [params['bev_src_bl_x'], params['bev_src_bl_y']],
        [params['bev_src_br_x'], params['bev_src_br_y']],
        [params['bev_src_tr_x'], params['bev_src_tr_y']],
        [params['bev_src_tl_x'], params['bev_src_tl_y']],
    ])
    dst = np.float32([
        [params['bev_dst_bl_x'], params['bev_dst_bl_y']],
        [params['bev_dst_br_x'], params['bev_dst_br_y']],
        [params['bev_dst_tr_x'], params['bev_dst_tr_y']],
        [params['bev_dst_tl_x'], params['bev_dst_tl_y']],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def process(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        (params['white_h_min'], params['white_s_min'], params['white_v_min']),
        (params['white_h_max'], params['white_s_max'], params['white_v_max']),
    )
    yellow_mask = cv2.inRange(
        hsv,
        (params['yellow_h_min'], params['yellow_s_min'], params['yellow_v_min']),
        (params['yellow_h_max'], params['yellow_s_max'], params['yellow_v_max']),
    )
    combined = cv2.bitwise_or(white_mask, yellow_mask)

    # ROI 폴리곤 마스크
    roi_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    poly = np.array([
        [params['bev_src_bl_x'], params['bev_src_bl_y']],
        [params['bev_src_br_x'], params['bev_src_br_y']],
        [params['bev_src_tr_x'], params['bev_src_tr_y']],
        [params['bev_src_tl_x'], params['bev_src_tl_y']],
    ], dtype=np.int32)
    cv2.fillPoly(roi_mask, [poly], 255)
    masked = cv2.bitwise_and(combined, roi_mask)

    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    M = _bev_matrix()
    bev = cv2.warpPerspective(binary, M, (IMG_W, IMG_H))

    # ── Camera 창: 폴리곤 + 각 코너 레이블 ─────────────
    cam_vis = frame.copy()
    cv2.polylines(cam_vis, [poly], True, (0, 255, 0), 1)
    corners = {
        'BL': (params['bev_src_bl_x'], params['bev_src_bl_y']),
        'BR': (params['bev_src_br_x'], params['bev_src_br_y']),
        'TL': (params['bev_src_tl_x'], params['bev_src_tl_y']),
        'TR': (params['bev_src_tr_x'], params['bev_src_tr_y']),
    }
    for label, (cx, cy) in corners.items():
        cv2.circle(cam_vis, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(cam_vis, label, (cx + 2, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    offset_color = (0, 255, 255)
    cv2.putText(cam_vis, f'offset: {latest_offset:.1f}px', (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, offset_color, 1)
    yellow_color = (0, 255, 0) if latest_yellow else (80, 80, 80)
    cv2.putText(cam_vis, f'yellow: {"ON" if latest_yellow else "off"}', (4, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, yellow_color, 1)

    # ── Mask 창: 흰색 / 노란색 채널 구분 표시 ────────
    mask_bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    mask_bgr[:, :, 2] = cv2.bitwise_and(combined, roi_mask)   # 빨간: 통합
    mask_bgr[:, :, 0] = cv2.bitwise_and(white_mask, roi_mask) # 파란: 흰색만
    mask_bgr[:, :, 1] = cv2.bitwise_and(yellow_mask, roi_mask)# 초록: 노란색만

    # ── BEV 창: dst 코너 레이블 ───────────────────────
    bev_bgr = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    dst_corners = {
        'BL': (params['bev_dst_bl_x'], params['bev_dst_bl_y']),
        'BR': (params['bev_dst_br_x'], params['bev_dst_br_y']),
        'TL': (params['bev_dst_tl_x'], params['bev_dst_tl_y']),
        'TR': (params['bev_dst_tr_x'], params['bev_dst_tr_y']),
    }
    for label, (cx, cy) in dst_corners.items():
        cv2.circle(bev_bgr, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(bev_bgr, label, (cx + 2, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    return cam_vis, mask_bgr, bev_bgr


# ── 메인 루프 ─────────────────────────────────────────────────────────
def main():
    global status_msg

    ros_ok = _HAS_ROS
    node = None

    if ros_ok:
        rclpy.init()
        node = TunerNode()
        ros_thread = threading.Thread(target=_ros_spin, args=(node,), daemon=True)
        ros_thread.start()

    create_windows()

    print('=' * 52)
    print('  D-Racer HSV + BEV 튜너')
    print('  a / Enter : /line_node에 파라미터 전송')
    print('  s         : YAML 파일로 저장')
    print('  p         : launch XML 형식으로 출력')
    print('  r         : 기본값으로 초기화')
    print('  q / Esc   : 종료')
    print('=' * 52)

    placeholder = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Waiting for camera...', (5, IMG_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    while True:
        read_trackbars()

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is not None:
            try:
                # 카메라 이미지가 예상 크기와 다르면 리사이즈
                if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
                    frame = cv2.resize(frame, (IMG_W, IMG_H))
                cam_vis, mask_bgr, bev_bgr = process(frame)
                cv2.imshow(WIN_CAM,  cam_vis)
                cv2.imshow(WIN_MASK, mask_bgr)
                cv2.imshow(WIN_BEV,  bev_bgr)
            except cv2.error as e:
                print(f'[OpenCV 오류] {e}')
                cv2.imshow(WIN_CAM,  placeholder)
                cv2.imshow(WIN_MASK, placeholder)
                cv2.imshow(WIN_BEV,  placeholder)
        else:
            cv2.imshow(WIN_CAM,  placeholder)
            cv2.imshow(WIN_MASK, placeholder)
            cv2.imshow(WIN_BEV,  placeholder)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):  # q 또는 Esc
            break
        elif key in (ord('a'), 13):  # a 또는 Enter
            if ros_ok:
                apply_all_params(node)
            else:
                print('[WARN] ROS2 없음 — 파라미터 전송 불가')
        elif key == ord('s'):
            save_yaml()
        elif key == ord('p'):
            print_launch_xml()
        elif key == ord('r'):
            reset_trackbars()
            status_msg = '기본값으로 초기화'
            print(f'[Reset] {status_msg}')

    cv2.destroyAllWindows()
    if node:
        node.destroy_node()
    if ros_ok:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
