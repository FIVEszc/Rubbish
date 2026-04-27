import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from RobotArmController import RobotArmController
from basic_control import grasp, open_grasp


try:
    robot_controller = RobotArmController("192.168.1.18", 8080, 3)
    print("Robot arm connected.")
except Exception as exc:
    print("Robot arm connection failed:", exc)
    raise SystemExit(1)


# Keep the same home pose and motion flow as bottle.py.
home = [
    0.013,
    -0.239,
    0.127,
    np.pi,
    np.pi / 64,
    5 * np.pi / 18,
]

robot_controller.movel(home)
time.sleep(1)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

selected_pixel = None
selected_box_index = -1


def mouse_callback(event, x, y, flags, param):
    global selected_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)


def print_robot_snapshot(label):
    state_ret, state = robot_controller.get_current_arm_state()
    joint_ret, joints = robot_controller.get_current_joint_degree()
    print(f"[{label}] current_state_ret={state_ret}, joint_ret={joint_ret}")
    if state_ret == 0:
        print(f"[{label}] current_pose={state['pose']}")
        print(f"[{label}] current_err={state['err']}")
    if joint_ret == 0:
        print(f"[{label}] current_joints_deg={joints}")


def move_with_diagnostics(pose, step_name):
    print(f"[{step_name}] target_pose={pose}")
    result = robot_controller.movel(pose)
    if result != 0:
        print_robot_snapshot(f"{step_name}_failed")
        raise RuntimeError(f"{step_name} failed with code {result}")
    return result


cv2.namedWindow("Bottle Demo V2")
cv2.setMouseCallback("Bottle Demo V2", mouse_callback)

model = YOLO("yolo11n.pt")

try:
    while True:
        robot_controller.movel(home)

        selected_pixel = None
        selected_box_index = -1

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            color_image = np.asanyarray(color_frame.get_data())
            results = model(color_image, classes=[39])

            centers = []
            boxes_list = []

            for r in results:
                boxes = r.boxes
                names = model.names

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{names[cls]} {conf:.2f}"

                    centers.append((cx, cy))
                    boxes_list.append((x1, y1, x2, y2, label))

            if selected_pixel and centers:
                min_dist = float("inf")
                for i, (cx, cy) in enumerate(centers):
                    dist = math.hypot(cx - selected_pixel[0], cy - selected_pixel[1])
                    if dist < min_dist:
                        min_dist = dist
                        selected_box_index = i

            for i, (x1, y1, x2, y2, label) in enumerate(boxes_list):
                cx, cy = centers[i]
                if i == selected_box_index:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.circle(color_image, (cx, cy), 5, color, -1)

            cv2.putText(
                color_image,
                "Left click target point | Enter to grasp",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Bottle Demo V2", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key in (10, 13) and selected_pixel:
                break

        pixel = selected_pixel
        if selected_box_index >= 0 and selected_box_index < len(centers):
            pixel = centers[selected_box_index]

        depth = depth_frame.get_distance(pixel[0], pixel[1])
        x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
        print(f"Selected pixel: {pixel}, depth: {depth:.3f} m, xyz=({x:.3f}, {y:.3f}, {z:.3f})")
        print_robot_snapshot("before_pick")

        open_grasp()

        target_pose = home.copy()
        target_pose[5] += 2 * np.pi / 3
        move_with_diagnostics(target_pose, "rotate_before_pick")

        target_pose[0] -= x * 1.1
        move_with_diagnostics(target_pose, "move_x_to_pick")

        target_pose[1] += y
        target_pose[1] -= 0.02
        move_with_diagnostics(target_pose, "move_y_to_pick")

        target_pose[4] += 11 * np.pi / 32
        move_with_diagnostics(target_pose, "rotate_ry_to_pick")

        target_pose[2] -= z * 1.46
        target_pose[2] += 0.141
        print(
            f"Original flow target pose: "
            f"x={target_pose[0]:.3f}, y={target_pose[1]:.3f}, z={target_pose[2]:.3f}, "
            f"rx={target_pose[3]:.3f}, ry={target_pose[4]:.3f}, rz={target_pose[5]:.3f}"
        )
        move_with_diagnostics(target_pose, "move_z_to_pick")

        grasp()
        time.sleep(0.5)

        target_pose[2] = 0.414
        move_with_diagnostics(target_pose, "lift_after_grasp")

        huanchong = [
            0.323167,
            -0.132920,
            0.461525,
            -11 * np.pi / 16,
            5 * np.pi / 14,
            -7 * np.pi / 12,
        ]
        move_with_diagnostics(huanchong, "move_to_buffer")
        time.sleep(0.2)

        cangku = [
            0.269854,
            0.202165,
            0.409083,
            -7 * np.pi / 8,
            7 * np.pi / 20,
            -5 * np.pi / 18,
        ]
        move_with_diagnostics(cangku, "move_to_drop")
        time.sleep(0.2)

        open_grasp()
        time.sleep(0.2)
        move_with_diagnostics(huanchong, "leave_drop_zone")
        move_with_diagnostics(home, "return_home")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
