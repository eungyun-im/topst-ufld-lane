#!/usr/bin/env python3
import argparse
import json
import os
import site
import sys
import time
from collections import deque

# The car and the PC sit on different subnets; multicast discovery does not
# cross them. Apply the unicast peer profile automatically when unset.
if 'FASTRTPS_DEFAULT_PROFILES_FILE' not in os.environ:
    for _profile in ('~/fastrtps_kuuve.xml', '~/fastrtps_topst.xml'):
        _p = os.path.expanduser(_profile)
        if os.path.isfile(_p):
            os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] = _p
            print(f'[ufld_view] using DDS profile {_p}')
            break

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Float32MultiArray, Int32

try:
    from control_msgs.msg import Control
except Exception:
    Control = None

try:
    from joystick_msgs.msg import Joystick
except Exception:
    Joystick = None

try:
    from battery_msgs.msg import Battery
except Exception:
    Battery = None



for user_site in site.getusersitepackages(), site.getuserbase():
    sys.path = [p for p in sys.path if not p.startswith(user_site)]

import cv2


TUNING_WINDOW_TRACKBARS = [
    'Y %', 'LaneW px', 'OffEMA x100', 'HdgEMA x100', 'CurvEMA x100',
    'OffBand x100', 'MaxStep px',
    'XWidth px', 'XYTol px', 'XTicks', 'XClear',
]
# Main window: the knobs actually touched while tuning on track (p -> d -> strScale -> i).
CONTROL_WINDOW_TRACKBARS = [
    'P x100', 'I x1000', 'D x100', 'HGain', 'StrScale x100',
]
# Advanced window: set once, then leave alone.
CONTROL_ADV_WINDOW_TRACKBARS = [
    'CurvDead x1e4', 'CurvFull x1e4', 'DFiltA x100',
    'MaxSteer x100', 'HClamp px', 'SteerStep x100',
    'IErrLim px', 'ITermLim px', 'ILeak x1000',
]
SPEED_WINDOW_TRACKBARS = [
    'MaxSpd x100', 'MinSpd x100', 'Risk x100', 'DecelA x100', 'AccelA x100',
]
BEV_WINDOW_TRACKBARS = [
    'BEV ON',
    'SrcBLx', 'SrcBLy', 'SrcBRx', 'SrcBRy', 'SrcTLx', 'SrcTLy', 'SrcTRx', 'SrcTRy',
    'DstBLx', 'DstBLy', 'DstBRx', 'DstBRy', 'DstTLx', 'DstTLy', 'DstTRx', 'DstTRy',
]

DEFAULT_TUNING = [
    0, 180, 0, 55, 150, 255,
    15, 30, 70, 255, 120, 255,
    0, 118, 320, 118, 90, 80, 230, 80,
    75, 120, 245, 120, 75, 0, 245, 0,
    0.74, 180.0, 0.25,
    0.22, 30.0,
    1.0, 1.20, 2.00, 0.04, 0.12,
    1.0,
    0.25,
    # [40] yellow X-cross width px, [41] X-cross y tolerance px,
    # [42] X-cross stable ticks, [43] X-cross clear ticks
    # (matches ufld_line_node tuningCb).
    60.0, 12.0, 2.0, 6.0,
]


class UFLDView(Node):
    def __init__(self, args):
        super().__init__('ufld_view')
        self.args = args
        self.last_offset = None
        self.frame_count = 0
        self.fps_t0 = time.perf_counter()
        self.rx_fps = 0.0
        self.paused = False
        self.has_received_image = False
        self.last_tuning = None
        self.last_control_tuning = None
        self.last_heading = None
        self.last_control = None
        self.last_mission_state = None
        self.last_joystick_mode = None
        self.last_ufld_debug = None
        self.last_control_debug = None
        self.last_bev_frame = None
        self.last_bev_t = 0.0
        self.last_offset_t = 0.0
        self.last_heading_t = 0.0
        self.last_control_t = 0.0
        self.last_mission_state_t = 0.0
        self.last_joystick_t = 0.0
        self.last_ufld_debug_t = 0.0
        self.last_control_debug_t = 0.0
        self.last_debug_print_t = 0.0
        self.last_battery = None
        self.last_battery_t = 0.0
        self.last_lane_mode = None
        self.last_lane_mode_t = 0.0

        # X-cross labeling capture: raw camera frames are buffered so the
        # approach to a crossing (pre-trigger) is saved along with the event.
        self.capture_buf = deque()
        self.recording = False
        self.was_capturing = False
        self.auto_record_until = 0.0
        self.capture_seq = 0
        self.capture_dir = None

        self.create_subscription(
            CompressedImage,
            args.capture_topic,
            self.on_capture_frame,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CompressedImage,
            args.image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(Float32, args.offset_topic, self.on_offset, 10)
        self.create_subscription(Float32, args.heading_topic, self.on_heading, 10)
        self.create_subscription(Int32, args.mission_topic, self.on_mission_state, 10)
        self.create_subscription(Int32, args.lane_mode_topic, self.on_lane_mode, 10)
        self.create_subscription(Float32MultiArray, args.ufld_debug_topic, self.on_ufld_debug, 10)
        self.create_subscription(Float32MultiArray, args.control_debug_topic, self.on_control_debug, 10)
        self.create_subscription(
            CompressedImage,
            args.bev_topic,
            self.on_bev_image,
            qos_profile_sensor_data,
        )
        if Control is not None:
            self.create_subscription(Control, args.control_topic, self.on_control, 10)
        if Joystick is not None:
            self.create_subscription(Joystick, args.joystick_topic, self.on_joystick, 10)
        if Battery is not None:
            self.create_subscription(Battery, args.battery_topic, self.on_battery, 10)

        self.tuning_pub = self.create_publisher(Float32MultiArray, args.tuning_topic, 10)
        self.control_tuning_pub = self.create_publisher(Float32MultiArray, args.control_tuning_topic, 10)
        self.create_timer(0.03, self.on_gui_timer)

        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(args.tuning_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(args.control_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(args.control_adv_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(args.speed_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(args.bev_window, cv2.WINDOW_NORMAL)

        # Lane/perception tuning: how the lane center + EMA filters are computed.
        cv2.createTrackbar('Y %', args.tuning_window, 74, 98, lambda _: None)
        cv2.createTrackbar('LaneW px', args.tuning_window, 180, 320, lambda _: None)
        cv2.createTrackbar('OffEMA x100', args.tuning_window, 25, 100, lambda _: None)
        cv2.createTrackbar('HdgEMA x100', args.tuning_window, 12, 100, lambda _: None)
        cv2.createTrackbar('CurvEMA x100', args.tuning_window, 25, 100, lambda _: None)
        cv2.createTrackbar('OffBand x100', args.tuning_window, 22, 50, lambda _: None)
        cv2.createTrackbar('MaxStep px', args.tuning_window, 30, 320, lambda _: None)
        # Yellow X-cross detector (BEV separation between yellow L/R lanes).
        cv2.createTrackbar('XWidth px', args.tuning_window, 60, 320, lambda _: None)
        cv2.createTrackbar('XYTol px', args.tuning_window, 12, 80, lambda _: None)
        cv2.createTrackbar('XTicks', args.tuning_window, 2, 30, lambda _: None)
        cv2.createTrackbar('XClear', args.tuning_window, 6, 60, lambda _: None)

        # Steering control: constant PID gains + curvature-based straight-section
        # P reduction (the old error-amplitude dynamic gain schedule was removed
        # from the controller; it acted as a limit-cycle generator on straights).
        # Main window: on-track tuning knobs.
        cv2.createTrackbar('P x100', args.control_window, 120, 600, lambda _: None)
        cv2.createTrackbar('I x1000', args.control_window, 20, 100, lambda _: None)
        cv2.createTrackbar('D x100', args.control_window, 60, 600, lambda _: None)
        cv2.createTrackbar('HGain', args.control_window, 20, 200, lambda _: None)
        cv2.createTrackbar('StrScale x100', args.control_window, 70, 100, lambda _: None)

        # Advanced window: set-once parameters.
        cv2.createTrackbar('CurvDead x1e4', args.control_adv_window, 20, 200, lambda _: None)
        cv2.createTrackbar('CurvFull x1e4', args.control_adv_window, 80, 400, lambda _: None)
        cv2.createTrackbar('DFiltA x100', args.control_adv_window, 35, 100, lambda _: None)
        cv2.createTrackbar('MaxSteer x100', args.control_adv_window, 45, 100, lambda _: None)
        cv2.createTrackbar('HClamp px', args.control_adv_window, 80, 320, lambda _: None)
        cv2.createTrackbar('SteerStep x100', args.control_adv_window, 5, 100, lambda _: None)
        cv2.createTrackbar('IErrLim px', args.control_adv_window, 40, 160, lambda _: None)
        cv2.createTrackbar('ITermLim px', args.control_adv_window, 30, 80, lambda _: None)
        cv2.createTrackbar('ILeak x1000', args.control_adv_window, 995, 1000, lambda _: None)

        # Speed/throttle control: risk-based speed reduction and its smoothing.
        cv2.createTrackbar('MaxSpd x100', args.speed_window, 35, 100, lambda _: None)
        cv2.createTrackbar('MinSpd x100', args.speed_window, 10, 100, lambda _: None)
        cv2.createTrackbar('Risk x100', args.speed_window, 100, 200, lambda _: None)
        cv2.createTrackbar('DecelA x100', args.speed_window, 45, 100, lambda _: None)
        cv2.createTrackbar('AccelA x100', args.speed_window, 8, 100, lambda _: None)

        # BEV geometry: each of the 8 corners (src trapezoid + dst rectangle) moves independently.
        cv2.createTrackbar('BEV ON', args.bev_window, 1, 1, lambda _: None)
        cv2.createTrackbar('SrcBLx', args.bev_window, 0, 320, lambda _: None)
        cv2.createTrackbar('SrcBLy', args.bev_window, 118, 160, lambda _: None)
        cv2.createTrackbar('SrcBRx', args.bev_window, 320, 320, lambda _: None)
        cv2.createTrackbar('SrcBRy', args.bev_window, 118, 160, lambda _: None)
        cv2.createTrackbar('SrcTLx', args.bev_window, 90, 320, lambda _: None)
        cv2.createTrackbar('SrcTLy', args.bev_window, 80, 160, lambda _: None)
        cv2.createTrackbar('SrcTRx', args.bev_window, 230, 320, lambda _: None)
        cv2.createTrackbar('SrcTRy', args.bev_window, 80, 160, lambda _: None)
        cv2.createTrackbar('DstBLx', args.bev_window, 75, 320, lambda _: None)
        cv2.createTrackbar('DstBLy', args.bev_window, 120, 160, lambda _: None)
        cv2.createTrackbar('DstBRx', args.bev_window, 245, 320, lambda _: None)
        cv2.createTrackbar('DstBRy', args.bev_window, 120, 160, lambda _: None)
        cv2.createTrackbar('DstTLx', args.bev_window, 75, 320, lambda _: None)
        cv2.createTrackbar('DstTLy', args.bev_window, 0, 160, lambda _: None)
        cv2.createTrackbar('DstTRx', args.bev_window, 245, 320, lambda _: None)
        cv2.createTrackbar('DstTRy', args.bev_window, 0, 160, lambda _: None)

        cv2.setTrackbarMin('P x100', args.control_window, 0)
        cv2.setTrackbarMin('I x1000', args.control_window, 0)
        cv2.setTrackbarMin('D x100', args.control_window, 0)
        cv2.setTrackbarMin('HGain', args.control_window, 0)
        cv2.setTrackbarMin('StrScale x100', args.control_window, 20)
        cv2.setTrackbarMin('DFiltA x100', args.control_adv_window, 1)
        cv2.setTrackbarMin('CurvFull x1e4', args.control_adv_window, 1)
        cv2.setTrackbarMin('ILeak x1000', args.control_adv_window, 900)

        cv2.setTrackbarMin('MaxSpd x100', args.speed_window, 0)
        cv2.setTrackbarMin('MinSpd x100', args.speed_window, 0)
        cv2.setTrackbarMin('Risk x100', args.speed_window, 0)
        cv2.setTrackbarMin('DecelA x100', args.speed_window, 1)
        cv2.setTrackbarMin('AccelA x100', args.speed_window, 1)

        cv2.setTrackbarMin('Y %', args.tuning_window, 10)
        cv2.setTrackbarMin('LaneW px', args.tuning_window, 20)
        cv2.setTrackbarMin('OffEMA x100', args.tuning_window, 1)
        cv2.setTrackbarMin('HdgEMA x100', args.tuning_window, 1)
        cv2.setTrackbarMin('CurvEMA x100', args.tuning_window, 1)
        cv2.setTrackbarMin('XWidth px', args.tuning_window, 5)
        cv2.setTrackbarMin('XYTol px', args.tuning_window, 1)
        cv2.setTrackbarMin('XTicks', args.tuning_window, 1)
        cv2.setTrackbarMin('XClear', args.tuning_window, 1)
        cv2.setTrackbarMin('BEV ON', args.bev_window, 0)
        for name in ('SrcBLx', 'SrcBLy', 'SrcBRx', 'SrcBRy', 'SrcTLx', 'SrcTLy', 'SrcTRx', 'SrcTRy',
                     'DstBLx', 'DstBLy', 'DstBRx', 'DstBRy', 'DstTLx', 'DstTLy', 'DstTRx', 'DstTRy'):
            cv2.setTrackbarMin(name, args.bev_window, 0)

        # Move and resize windows to be side-by-side: video | lane | steering | speed | BEV.
        # Steering is split into a short main window (on-track knobs) with the
        # advanced (set-once) window stacked directly below it in the same column.
        cv2.resizeWindow(args.window, 640, 680)
        cv2.moveWindow(args.window, 50, 100)

        cv2.resizeWindow(args.tuning_window, 380, 460)
        cv2.moveWindow(args.tuning_window, 710, 100)

        cv2.resizeWindow(args.control_window, 380, 230)
        cv2.moveWindow(args.control_window, 1100, 100)

        cv2.resizeWindow(args.control_adv_window, 380, 380)
        cv2.moveWindow(args.control_adv_window, 1100, 380)

        cv2.resizeWindow(args.speed_window, 380, 230)
        cv2.moveWindow(args.speed_window, 1490, 100)

        cv2.resizeWindow(args.bev_window, 380, 700)
        cv2.moveWindow(args.bev_window, 1880, 100)

        self.load_state()

        self.publish_tuning(force=True)
        self.publish_control_tuning(force=True)
        self.get_logger().info(
            f'viewing TOPST UFLD image={args.image_topic}, offset={args.offset_topic}, '
            f'bev={args.bev_topic}, tuning={args.tuning_topic}, control_tuning={args.control_tuning_topic}'
        )
        self.get_logger().info(
            f'debug print={args.debug_print}, ufld_debug={args.ufld_debug_topic}, '
            f'control_debug={args.control_debug_topic}, joystick={args.joystick_topic}, '
            f'bev_window={args.bev_window}'
        )

    def on_offset(self, msg):
        self.last_offset = float(msg.data)
        self.last_offset_t = time.perf_counter()

    def on_heading(self, msg):
        self.last_heading = float(msg.data)
        self.last_heading_t = time.perf_counter()

    def on_mission_state(self, msg):
        self.last_mission_state = int(msg.data)
        self.last_mission_state_t = time.perf_counter()

    def on_lane_mode(self, msg):
        self.last_lane_mode = int(msg.data)
        self.last_lane_mode_t = time.perf_counter()

    def on_control(self, msg):
        self.last_control = (float(msg.steering), float(msg.throttle))
        self.last_control_t = time.perf_counter()

    def on_joystick(self, msg):
        self.last_joystick_mode = bool(msg.use_joystick_control)
        self.last_joystick_t = time.perf_counter()

    def on_battery(self, msg):
        self.last_battery = float(msg.battery_status)
        self.last_battery_t = time.perf_counter()


    def on_ufld_debug(self, msg):
        self.last_ufld_debug = [float(v) for v in msg.data]
        self.last_ufld_debug_t = time.perf_counter()
        # X-CROSS active extends the auto-capture window; the pre-buffer in
        # on_capture_frame covers the approach before this fires.
        if (self.args.auto_capture and len(self.last_ufld_debug) >= 30
                and self.last_ufld_debug[28] >= 0.5):
            self.auto_record_until = self.last_ufld_debug_t + self.args.capture_post_sec

    def capture_active(self, now):
        return self.recording or now <= self.auto_record_until

    def ensure_capture_dir(self):
        if self.capture_dir is None:
            base = os.path.expanduser(self.args.capture_dir)
            self.capture_dir = os.path.join(base, time.strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self.capture_dir, exist_ok=True)
            self.get_logger().info(f'saving X-cross captures to {self.capture_dir}')
        return self.capture_dir

    def save_capture(self, t, data):
        path = self.ensure_capture_dir()
        name = f'{self.capture_seq:06d}.jpg'
        try:
            with open(os.path.join(path, name), 'wb') as f:
                f.write(data)
        except OSError as exc:
            self.get_logger().warn(f'capture save failed: {exc}')
            return
        self.capture_seq += 1
        dbg = self.last_ufld_debug or []

        def d(i):
            return f'{dbg[i]:.1f}' if len(dbg) > i else ''

        try:
            log_path = os.path.join(path, 'labels.csv')
            write_header = not os.path.isfile(log_path)
            with open(log_path, 'a') as f:
                if write_header:
                    f.write('file,t,xgap_px,xstreak,xcross,lane_mode,l1_pts,l2_pts\n')
                f.write(f'{name},{t:.3f},{d(26)},{d(27)},{d(28)},{d(25)},{d(17)},{d(18)}\n')
        except OSError:
            pass

    def on_capture_frame(self, msg):
        now = time.perf_counter()
        data = bytes(msg.data)
        active = self.capture_active(now)
        if active:
            if not self.was_capturing:
                for t, buffered in self.capture_buf:
                    self.save_capture(t, buffered)
                self.capture_buf.clear()
            self.save_capture(now, data)
        else:
            self.capture_buf.append((now, data))
            while self.capture_buf and now - self.capture_buf[0][0] > self.args.capture_pre_sec:
                self.capture_buf.popleft()
        self.was_capturing = active

    def on_control_debug(self, msg):
        self.last_control_debug = [float(v) for v in msg.data]
        self.last_control_debug_t = time.perf_counter()

    def on_bev_image(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.last_bev_frame = frame
        self.last_bev_t = time.perf_counter()

    def on_gui_timer(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            rclpy.shutdown()
        elif key == ord(' '):
            self.paused = not self.paused
        elif key == ord('r'):
            self.recording = not self.recording
            self.get_logger().info(f'manual capture {"ON" if self.recording else "OFF"}')
        elif key == ord('c'):
            if self.capture_buf:
                t, data = self.capture_buf[-1]
                self.save_capture(t, data)
        self.publish_tuning()
        self.publish_control_tuning()
        self.print_debug_line()

        if not self.has_received_image:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "WAITING FOR TOPST CAMERA FEED...", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            placeholder = self.draw_hud(placeholder)
            cv2.imshow(self.args.window, self.stack_with_bev(placeholder))

    def current_bev_points(self):
        def pos(name):
            return float(cv2.getTrackbarPos(name, self.args.bev_window))

        # Order matches the /line_tuning protocol: bl, br, tl, tr for src, then dst.
        return [
            pos('SrcBLx'), pos('SrcBLy'),
            pos('SrcBRx'), pos('SrcBRy'),
            pos('SrcTLx'), pos('SrcTLy'),
            pos('SrcTRx'), pos('SrcTRy'),
            pos('DstBLx'), pos('DstBLy'),
            pos('DstBRx'), pos('DstBRy'),
            pos('DstTLx'), pos('DstTLy'),
            pos('DstTRx'), pos('DstTRy'),
        ]

    def current_tuning(self):
        tuning = list(DEFAULT_TUNING)
        tuning[12:28] = self.current_bev_points()
        tuning[28] = cv2.getTrackbarPos('Y %', self.args.tuning_window) / 100.0
        tuning[29] = float(cv2.getTrackbarPos('LaneW px', self.args.tuning_window))
        tuning[30] = cv2.getTrackbarPos('OffEMA x100', self.args.tuning_window) / 100.0
        tuning[31] = cv2.getTrackbarPos('OffBand x100', self.args.tuning_window) / 100.0
        tuning[32] = float(cv2.getTrackbarPos('MaxStep px', self.args.tuning_window))
        # Keep index 33 to 36 as their default values in DEFAULT_TUNING
        # Except DynLD (index 33) which we explicitly set to 0.0 to disable dynamic LD.
        tuning[33] = 0.0
        tuning[37] = cv2.getTrackbarPos('HdgEMA x100', self.args.tuning_window) / 100.0
        tuning[38] = float(cv2.getTrackbarPos('BEV ON', self.args.bev_window))
        tuning[39] = cv2.getTrackbarPos('CurvEMA x100', self.args.tuning_window) / 100.0
        tuning[40] = float(cv2.getTrackbarPos('XWidth px', self.args.tuning_window))
        tuning[41] = float(cv2.getTrackbarPos('XYTol px', self.args.tuning_window))
        tuning[42] = float(cv2.getTrackbarPos('XTicks', self.args.tuning_window))
        tuning[43] = float(cv2.getTrackbarPos('XClear', self.args.tuning_window))
        return tuning

    def current_control_tuning(self):
        # /control_tuning protocol v2 (matches control_pkg control_node):
        # [0] reserved (was FSM state), [1] P, [2] D, [3] heading gain,
        # [4] D filter alpha, [5] max steering rad, [6] max speed, [7] min speed,
        # [8] throttle risk scale, [9] heading clamp px, [10] steering step,
        # [11] I gain, [12] straight P scale, [13] straight curv deadband,
        # [14] decel alpha, [15] accel alpha, [16] straight curv full,
        # [17] integral error limit px, [18] integral term limit px,
        # [19] integral leak, [20] protocol version marker.
        p_gain = cv2.getTrackbarPos('P x100', self.args.control_window) / 100.0
        i_gain = cv2.getTrackbarPos('I x1000', self.args.control_window) / 1000.0
        d_gain = cv2.getTrackbarPos('D x100', self.args.control_window) / 100.0
        heading_gain = float(cv2.getTrackbarPos('HGain', self.args.control_window))
        straight_scale = cv2.getTrackbarPos('StrScale x100', self.args.control_window) / 100.0

        d_alpha = cv2.getTrackbarPos('DFiltA x100', self.args.control_adv_window) / 100.0
        max_steer = cv2.getTrackbarPos('MaxSteer x100', self.args.control_adv_window) / 100.0
        h_clamp = float(cv2.getTrackbarPos('HClamp px', self.args.control_adv_window))
        steer_step = cv2.getTrackbarPos('SteerStep x100', self.args.control_adv_window) / 100.0
        curv_dead = cv2.getTrackbarPos('CurvDead x1e4', self.args.control_adv_window) / 10000.0
        curv_full = cv2.getTrackbarPos('CurvFull x1e4', self.args.control_adv_window) / 10000.0
        i_err_lim = float(cv2.getTrackbarPos('IErrLim px', self.args.control_adv_window))
        i_term_lim = float(cv2.getTrackbarPos('ITermLim px', self.args.control_adv_window))
        i_leak = cv2.getTrackbarPos('ILeak x1000', self.args.control_adv_window) / 1000.0

        max_speed = cv2.getTrackbarPos('MaxSpd x100', self.args.speed_window) / 100.0
        min_speed = cv2.getTrackbarPos('MinSpd x100', self.args.speed_window) / 100.0
        risk_scale = cv2.getTrackbarPos('Risk x100', self.args.speed_window) / 100.0
        decel_alpha = cv2.getTrackbarPos('DecelA x100', self.args.speed_window) / 100.0
        accel_alpha = cv2.getTrackbarPos('AccelA x100', self.args.speed_window) / 100.0

        return [0.0, p_gain, d_gain, heading_gain, d_alpha, max_steer,
                max_speed, min_speed, risk_scale, h_clamp, steer_step,
                i_gain, straight_scale, curv_dead, decel_alpha, accel_alpha,
                curv_full, i_err_lim, i_term_lim, i_leak, 2.0]

    def publish_tuning(self, force=False):
        tuning = self.current_tuning()
        if not force and self.last_tuning == tuning:
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in tuning]
        self.tuning_pub.publish(msg)
        self.last_tuning = tuning
        self.save_state()

    def publish_control_tuning(self, force=False):
        tuning = self.current_control_tuning()
        if not force and self.last_control_tuning == tuning:
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in tuning]
        self.control_tuning_pub.publish(msg)
        self.last_control_tuning = tuning
        self.save_state()

    def _trackbar_groups(self):
        return [
            ('tuning_window', self.args.tuning_window, TUNING_WINDOW_TRACKBARS),
            ('control_window', self.args.control_window, CONTROL_WINDOW_TRACKBARS),
            ('control_adv_window', self.args.control_adv_window, CONTROL_ADV_WINDOW_TRACKBARS),
            ('speed_window', self.args.speed_window, SPEED_WINDOW_TRACKBARS),
            ('bev_window', self.args.bev_window, BEV_WINDOW_TRACKBARS),
        ]

    def save_state(self):
        state = {
            key: {name: cv2.getTrackbarPos(name, window) for name in names}
            for key, window, names in self._trackbar_groups()
        }
        path = os.path.expanduser(self.args.state_file)
        try:
            tmp_path = path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)
        except OSError as exc:
            self.get_logger().warn(f'Failed to save tuning state to {path}: {exc}')

    def load_state(self):
        path = os.path.expanduser(self.args.state_file)
        if not os.path.isfile(path):
            return
        try:
            with open(path, 'r') as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Failed to load tuning state from {path}: {exc}')
            return

        for key, window, names in self._trackbar_groups():
            for name, value in state.get(key, {}).items():
                if name not in names:
                    continue
                try:
                    cv2.setTrackbarPos(name, window, int(value))
                except cv2.error:
                    pass
        self.get_logger().info(f'Loaded tuning state from {path}')

    def is_auto_mode(self):
        return self.last_joystick_mode is not True

    def fmt(self, value, precision=2):
        if value is None:
            return 'nan'
        return f'{value:.{precision}f}'

    def is_fresh(self, stamp, now):
        return stamp > 0.0 and (now - stamp) <= self.args.stale_sec

    def print_debug_line(self):
        if not self.args.debug_print or not self.is_auto_mode():
            return

        now = time.perf_counter()
        if now - self.last_debug_print_t < self.args.debug_period:
            return
        self.last_debug_print_t = now

        tuning = self.current_tuning()
        ctrl_tuning = self.current_control_tuning()
        state = self.last_mission_state
        steering = throttle = None
        if self.last_control is not None and self.is_fresh(self.last_control_t, now):
            steering, throttle = self.last_control

        ufld_fresh = self.is_fresh(self.last_ufld_debug_t, now)
        ctrl_fresh = self.is_fresh(self.last_control_debug_t, now)
        ufld = self.last_ufld_debug if ufld_fresh else None
        ctrl = self.last_control_debug if ctrl_fresh else None

        if ufld and len(ufld) >= 23:
            ufld_part = (
                f'UFLD ok={int(ufld[0])} raw={ufld[1]:.1f} clamp={ufld[2]:.1f} '
                f'filt={ufld[3]:.1f} h_raw={ufld[4]:.3f} h_filt={ufld[5]:.3f} '
                f'dyn_y={ufld[6]:.2f} dyn={ufld[7]:.2f} center={ufld[8]:.1f} '
                f'dstep={ufld[9]:.1f} sclamp={int(ufld[10])} '
                f'lanes=[{int(ufld[16])},{int(ufld[17])},{int(ufld[18])},{int(ufld[19])}] '
                f'infer={ufld[20]:.1f}ms succ={ufld[22]:.1f}%'
            )
            if len(ufld) >= 30:
                gap_str = f'{ufld[26]:.0f}' if ufld[26] >= 0 else '--'
                ufld_part += (
                    f' lane_mode={int(ufld[25])} xgap={gap_str} '
                    f'xstk={int(ufld[27])} xcross={int(ufld[28])}'
                )
        else:
            stale = 'STALE ' if self.last_ufld_debug is not None and not ufld_fresh else ''
            ufld_part = (
                f'UFLD {stale}offset={self.fmt(self.last_offset)} heading={self.fmt(self.last_heading, 3)} '
                f'Y={tuning[28]:.2f} laneW={tuning[29]:.0f} alpha={tuning[30]:.2f} '
                f'band={tuning[31]:.2f} maxStep={tuning[32]:.0f}'
            )

        if ctrl and len(ctrl) >= 20:
            h_extra = ''
            if len(ctrl) >= 22:
                h_extra = f'hraw={ctrl[20]:.1f} hlim={ctrl[21]:.0f} '
            if len(ctrl) >= 23:
                h_extra += f'sstep={ctrl[22]:.3f} '
            if len(ctrl) >= 24:
                h_extra += f'line_ok={int(ctrl[23])} '
            if len(ctrl) >= 29:
                h_extra += (
                    f'thrT={ctrl[24]:.3f} risk={ctrl[25]:.2f} '
                    f'ro={ctrl[26]:.2f} rh={ctrl[27]:.2f} rd={ctrl[28]:.2f} '
                )
            if len(ctrl) >= 34:
                h_extra += f'iterm={ctrl[31]:.1f} integ={ctrl[32]:.0f} strS={ctrl[33]:.2f} '
            ctrl_part = (
                f'CTRL state={int(ctrl[0])} off={ctrl[1]:.1f} head={ctrl[2]:.3f} '
                f'hg={ctrl[3]:.1f} {h_extra}hterm={ctrl[4]:.1f} eff={ctrl[5]:.1f} '
                f'kp={ctrl[7]:.2f} kd={ctrl[8]:.2f} p={ctrl[9]:.1f} '
                f'draw={ctrl[10]:.1f} dfilt={ctrl[11]:.1f} dterm={ctrl[12]:.1f} '
                f'pid={ctrl[13]:.1f} steer={ctrl[14]:.4f} dsteer={ctrl[15]:.4f} '
                f'thr={ctrl[16]:.3f} freshD={int(ctrl[17])}'
            )
        else:
            heading = self.last_heading if self.is_fresh(self.last_heading_t, now) and self.last_heading is not None else 0.0
            offset = self.last_offset if self.is_fresh(self.last_offset_t, now) and self.last_offset is not None else 0.0
            heading_gain = ctrl_tuning[3]
            raw_h_term = heading_gain * heading
            h_limit = ctrl_tuning[9]
            h_term = max(-h_limit, min(h_limit, raw_h_term))
            eff_error = offset + h_term
            stale = 'STALE ' if self.last_control_debug is not None and not ctrl_fresh else ''
            ctrl_part = (
                f'CTRL {stale}'
                f'p={ctrl_tuning[1]:.2f} i={ctrl_tuning[11]:.3f} d={ctrl_tuning[2]:.2f} '
                f'hg={heading_gain:.1f} strS={ctrl_tuning[12]:.2f} sstep={ctrl_tuning[10]:.3f} '
                f'hraw={raw_h_term:.1f} hlim={h_limit:.0f} '
                f'hterm={h_term:.1f} eff={eff_error:.1f} '
                f'steer={self.fmt(steering, 4)} thr={self.fmt(throttle, 3)}'
            )

        print(f'DBG mode=auto {ufld_part} | {ctrl_part}', flush=True)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        if w < 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
            h, w = frame.shape[:2]

        # Draw a semi-transparent HUD background at the bottom
        hud_h = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (15, 15, 15), -1)
        # Apply alpha blending
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw borders
        cv2.line(frame, (0, h - hud_h), (w, h - hud_h), (50, 50, 50), 1)

        # Retrieve debug info
        now = time.perf_counter()
        ctrl_fresh = self.is_fresh(self.last_control_debug_t, now)
        ctrl = self.last_control_debug if ctrl_fresh else None

        # 1. Draw MODE info
        is_auto = self.is_auto_mode()
        mode_str = "AUTO" if is_auto else "JOYSTICK"
        mode_color = (0, 255, 0) if is_auto else (0, 165, 255)
        cv2.putText(frame, mode_str, (15, h - hud_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # 2. Draw Speed / Throttle bar
        throttle = 0.0
        steering = 0.0
        if self.last_control is not None and self.is_fresh(self.last_control_t, now):
            steering, throttle = self.last_control
        elif ctrl is not None and len(ctrl) >= 17:
            steering = ctrl[14]
            throttle = ctrl[16]

        cv2.putText(frame, f"THR: {throttle:.2f}", (15, h - hud_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw throttle progress bar
        bar_w = 120
        cv2.rectangle(frame, (15, h - hud_h + 70), (15 + bar_w, h - hud_h + 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, h - hud_h + 70), (15 + int(bar_w * max(0.0, min(1.0, throttle))), h - hud_h + 80), (0, 255, 0), -1)

        # 3. Draw Steering bar
        cv2.putText(frame, f"STR: {steering:.2f}", (180, h - hud_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw steering center-aligned bar
        cv2.rectangle(frame, (180, h - hud_h + 40), (180 + bar_w, h - hud_h + 50), (50, 50, 50), -1)
        center_x = 180 + bar_w // 2
        steer_offset = int((bar_w // 2) * max(-1.0, min(1.0, steering)))
        if steer_offset >= 0:
            cv2.rectangle(frame, (center_x, h - hud_h + 40), (center_x + steer_offset, h - hud_h + 50), (0, 255, 255), -1)
        else:
            cv2.rectangle(frame, (center_x + steer_offset, h - hud_h + 40), (center_x, h - hud_h + 50), (0, 255, 255), -1)
        cv2.line(frame, (center_x, h - hud_h + 38), (center_x, h - hud_h + 52), (255, 255, 255), 1)

        # 4. Draw Active Gains (KP / KD)
        kp = 0.0
        kd = 0.0
        if ctrl is not None and len(ctrl) >= 9:
            kp = ctrl[7]
            kd = ctrl[8]
        else:
            kp = cv2.getTrackbarPos('P x100', self.args.control_window) / 100.0
            kd = cv2.getTrackbarPos('D x100', self.args.control_window) / 100.0

        cv2.putText(frame, f"Active KP: {kp:.2f}", (340, h - hud_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(frame, f"Active KD: {kd:.2f}", (340, h - hud_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # 6. Draw Battery Status
        batt_str = "BATT: N/A"
        batt_color = (150, 150, 150)
        if self.last_battery is not None and self.is_fresh(self.last_battery_t, now):
            batt_percent = self.last_battery
            batt_str = f"BATT: {batt_percent:.1f}%"
            if batt_percent > 50.0:
                batt_color = (0, 255, 0)
            elif batt_percent > 20.0:
                batt_color = (0, 165, 255)
            else:
                batt_color = (0, 0, 255)
        cv2.putText(frame, batt_str, (340, h - hud_h + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, batt_color, 1)

        # 7. Mission lane mode (/mission_lane_mode; falls back to /ufld_debug[25])
        ufld_dbg = self.last_ufld_debug if self.is_fresh(self.last_ufld_debug_t, now) else None
        lane_mode = None
        if self.last_lane_mode is not None and self.is_fresh(self.last_lane_mode_t, now):
            lane_mode = self.last_lane_mode
        elif ufld_dbg is not None and len(ufld_dbg) >= 30 and ufld_dbg[25] >= 0:
            lane_mode = int(ufld_dbg[25])
        lane_names = {0: 'WHITE', 1: 'YELLOW', 2: 'YELLOW-RIGHT', 3: 'YELLOW-LEFT'}
        if lane_mode is None:
            lane_str, lane_color = 'LANE: N/A', (150, 150, 150)
        else:
            lane_str = f'LANE: {lane_names.get(lane_mode, lane_mode)}'
            lane_color = (255, 255, 255) if lane_mode == 0 else (0, 255, 255)
        cv2.putText(frame, lane_str, (15, h - hud_h + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_color, 2 if lane_mode in (2, 3) else 1)

        # 8. Yellow X-cross detector state: min BEV gap between yellow L/R,
        # streak count vs stable ticks, applied width threshold.
        if ufld_dbg is not None and len(ufld_dbg) >= 30:
            gap = ufld_dbg[26]
            streak = int(ufld_dbg[27])
            cross_active = ufld_dbg[28] >= 0.5
            gap_thr = ufld_dbg[29]
            gap_str = f'{gap:.0f}px' if gap >= 0 else '--'
            x_color = (0, 0, 255) if cross_active else ((0, 255, 255) if streak > 0 else (150, 150, 150))
            cv2.putText(frame, f'XGAP: {gap_str} thr={gap_thr:.0f} stk={streak}',
                        (180, h - hud_h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, x_color, 1)
            if cross_active:
                cv2.putText(frame, 'X-CROSS', (180, h - hud_h + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


        # 5. Draw Tracking Error
        offset = 0.0
        if self.last_offset is not None:
            offset = self.last_offset
        cv2.putText(frame, f"Offset: {offset:.1f}px", (480, h - hud_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw error level
        err_color = (0, 255, 0)
        if abs(offset) > 40:
            err_color = (0, 165, 255)
        if abs(offset) > 80:
            err_color = (0, 0, 255)
        cv2.putText(frame, "STATUS: OK" if abs(offset) < 60 else "WARN", (480, h - hud_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, err_color, 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.rx_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # 9. Capture status (top-right): red while saving, gray tally after.
        if self.capture_active(now):
            tag = 'REC(man)' if self.recording else 'REC(x)'
            cv2.putText(frame, f'{tag} {self.capture_seq}', (w - 140, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif self.capture_seq > 0:
            cv2.putText(frame, f'saved {self.capture_seq}', (w - 120, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        return frame

    def stack_with_bev(self, frame):
        target_w = frame.shape[1]
        bev_h = 200
        now = time.perf_counter()
        bev = None
        if self.last_bev_frame is not None and self.is_fresh(self.last_bev_t, now):
            bev = self.last_bev_frame.copy()

        if bev is None:
            bev = np.zeros((bev_h, target_w, 3), dtype=np.uint8)
            cv2.putText(bev, f"WAITING FOR BEV: {self.args.bev_topic}", (20, bev_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            h, w = bev.shape[:2]
            scale = target_w / max(1, w)
            bev = cv2.resize(bev, (target_w, max(1, int(h * scale))))
            if bev.shape[0] > bev_h:
                bev = cv2.resize(bev, (target_w, bev_h))
            elif bev.shape[0] < bev_h:
                pad = np.zeros((bev_h - bev.shape[0], target_w, 3), dtype=np.uint8)
                bev = np.vstack([bev, pad])

        cv2.rectangle(bev, (0, 0), (target_w, 24), (15, 15, 15), -1)
        cv2.putText(bev, "BEV", (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(bev, (target_w // 2, 0), (target_w // 2, bev_h), (255, 255, 0), 1)
        return np.vstack([frame, bev])

    def on_image(self, msg):
        if self.paused:
            return

        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        self.has_received_image = True
        self.frame_count += 1
        now = time.perf_counter()
        if now - self.fps_t0 >= 1.0:
            self.rx_fps = self.frame_count / (now - self.fps_t0)
            self.frame_count = 0
            self.fps_t0 = now

        frame = self.draw_hud(frame)
        cv2.imshow(self.args.window, self.stack_with_bev(frame))


def parse_args():
    parser = argparse.ArgumentParser(description='View UFLD inference image published by TOPST line_node.')
    parser.add_argument('--image-topic', default='/ufld/image/compressed')
    parser.add_argument('--offset-topic', default='/line_offset')
    parser.add_argument('--heading-topic', default='/line_heading')
    parser.add_argument('--mission-topic', default='/mission_state')
    parser.add_argument('--lane-mode-topic', default='/mission_lane_mode')
    parser.add_argument('--control-topic', default='/control')
    parser.add_argument('--joystick-topic', default='joystick')
    parser.add_argument('--ufld-debug-topic', default='/ufld_debug')
    parser.add_argument('--control-debug-topic', default='/control_debug')
    parser.add_argument('--bev-topic', default='/line/image/bev')
    parser.add_argument('--tuning-topic', default='/line_tuning')
    parser.add_argument('--control-tuning-topic', default='/control_tuning')
    parser.add_argument('--battery-topic', default='/battery_status')
    # X-cross labeling capture: raw (un-annotated) camera frames.
    parser.add_argument('--capture-topic', default='/camera/image/compressed')
    parser.add_argument('--capture-dir', default='~/Downloads/xcross_captures')
    parser.add_argument('--capture-pre-sec', type=float, default=2.0)
    parser.add_argument('--capture-post-sec', type=float, default=1.0)
    parser.add_argument('--auto-capture', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--debug-print', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--debug-period', type=float, default=0.5)
    parser.add_argument('--stale-sec', type=float, default=1.5)
    parser.add_argument('--window', default='TOPST UFLD View')
    parser.add_argument('--tuning-window', default='UFLD Direct Offset Tuning')
    parser.add_argument('--bev-window', default='UFLD BEV Tuning')
    parser.add_argument('--control-window', default='UFLD Steering Tuning')
    parser.add_argument('--control-adv-window', default='UFLD Steering Advanced')
    parser.add_argument('--speed-window', default='UFLD Speed Tuning')
    # v2: fresh state file so slider positions saved for the old dynamic-gain
    # controller don't override the new controller's start values.
    parser.add_argument('--state-file', default='~/.ufld_view_tuning_v2.json')
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = UFLDView(args)
    try:
        rclpy.spin(node)
    finally:
        node.save_state()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
