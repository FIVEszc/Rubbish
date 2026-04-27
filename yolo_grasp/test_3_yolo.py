import time
import cv2
from ultralytics import YOLO
import numpy as np
import math
from RobotArmController import RobotArmController
from basic_control import grasp, open_grasp

# 加载YOLO模型并指定使用GPU
try:
    robot_controller = RobotArmController("192.168.1.18", 8080, 3)
    print("登录成功")
except Exception as e:
    print("登录失败:", e)
    exit()



home = [0.386, 0.009, 0.081, np.pi / 2, -np.pi / 4, np.pi / 2]

robot_controller.movel(home)

time.sleep(1)

# 相机检测到的目标点 (x, y, z)，单位为米，构造成齐次坐标
import pyrealsense2 as rs

# 初始化管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 添加对齐器：将 depth 对齐到 color
align_to = rs.stream.color
align = rs.align(align_to)

selected_pixel = None
selected_box_index = -1  # 被选中的框的索引


def mouse_callback(event, x, y, flags, param):
    global selected_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)


# 注册鼠标事件
cv2.namedWindow("YOLO Detection")
cv2.setMouseCallback("YOLO Detection", mouse_callback)

model = YOLO("yolo11n.pt")

while True:
    # 每次操作结束后返回home位置
    robot_controller.movel(home)

    selected_pixel = None
    selected_box_index = -1  # 重置选中框索引

    while True:
        # 获取对齐帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 对齐处理
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 获取彩色帧
        # color_image = np.asanyarray(color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转
        results = model(color_image, classes=[47, 49])

        # 边界框中心点列表
        centers = []
        boxes_list = []

        for r in results:
            boxes = r.boxes
            names = model.names

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{names[cls]} {conf:.2f}"

                centers.append((cx, cy))
                boxes_list.append((x1, y1, x2, y2, label))

        # 如果有点击，找最近的 bbox
        if selected_pixel and centers:
            min_dist = float("inf")
            for i, (cx, cy) in enumerate(centers):
                dist = math.hypot(cx - selected_pixel[0], cy - selected_pixel[1])
                if dist < min_dist:
                    min_dist = dist
                    selected_box_index = i

        # 绘制所有框
        for i, (x1, y1, x2, y2, label) in enumerate(boxes_list):
            cx, cy = centers[i]
            if i == selected_box_index:
                color = (0, 0, 255)  # 红色：选中
            else:
                color = (0, 255, 0)  # 绿色：未选中

            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(color_image, (cx, cy), 5, color, -1)
            # cv2.putText(color_image, label, (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("YOLO Detection", color_image)
        key = cv2.waitKey(1)
        if key == 27 and selected_pixel:  # ESC 退出
            break

    pixel = selected_pixel
    depth = depth_frame.get_distance(pixel[0], pixel[1])
    x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
    print("选中点空间坐标（米）:", x, y, z)
    # x, y, z = x * 1000, y * 1000, z * 1000  # 转为毫米
    # print(f"x,y,z={x},{y},{z}")
    # exit()
    open_grasp()
    # T_base_ee = get_current_end_effector_pose()
    T_base_ee = home.copy()  # 复制一份home作为初始末端位姿
    T_base_ee[1] -= x
    T_base_ee[1] += 0.055  # 越大越左
    T_base_ee[2] -= y
    T_base_ee[2] += 0.032  # 越大越上
    T_base_ee[4] -= np.pi / 6
    robot_controller.movel(T_base_ee)
    # T_base_ee = get_current_end_effector_pose()
    T_base_ee[0] += z
    T_base_ee[0] -= 0.108  # 越大越里
    # T_base_ee[4] -= np.pi / 12
    robot_controller.movel(T_base_ee)
    T_base_ee[2] += 0.02  # 越大越上
    robot_controller.movel(T_base_ee)

    print("T_base_ee", T_base_ee)

    # set_width(robot, 70)
    grasp()
    time.sleep(0.5)

    # T_base_ee = get_current_end_effector_pose()
    # T_base_ee[1] = -450
    # robot_controller.movel(T_base_ee)

    robot_controller.movel(home)
    home1 = home.copy()
    home1[2] -= 0.09
    home1[4] += np.pi / 2
    robot_controller.movel(home1)
    time.sleep(1)
    open_grasp()
    time.sleep(1)
    robot_controller.movel(home)
    # set_width(robot, 100)
    time.sleep(1)
