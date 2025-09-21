#!/usr/bin/env python3
"""
HybrIK + RealSense Real-time Demo
"""
import sys
import os

HYBRIK_PATH = os.environ.get('HYBRIK_PATH', '/home/vb/workspace/HybrIK')
sys.path.append(HYBRIK_PATH)

import cv2
import numpy as np
import torch
import time
import pyrealsense2 as rs
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

print("HybrIK + RealSense Demo")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

print("\nLoading HybrIK...")

cfg_file = os.path.join(HYBRIK_PATH, 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml')
CKPT = os.path.join(HYBRIK_PATH, 'pretrained_models/hybrik_hrnet.pth')
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]

dummy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPLCam(
    dummy_set,
    scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR,
    sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False,
    add_dpg=False,
    loss_type=cfg.LOSS['TYPE']
)

det_transform = T.Compose([T.ToTensor()])
det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)

save_dict = torch.load(CKPT, map_location=device)
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

det_model.to(device)
hybrik_model.to(device)
det_model.eval()
hybrik_model.eval()
print("HybrIK loaded")

h2o_keypoint_indices = [7, 8, 16, 17, 18, 19, 20, 21]

print("Initializing RealSense...")

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("No RealSense device found!")
    exit(1)

rs_device = devices[0]
serial = rs_device.get_info(rs.camera_info.serial_number)
print(f"Found: {rs_device.get_info(rs.camera_info.name)} (Serial: {serial})")

import subprocess
result = subprocess.run(['lsusb', '-t'], capture_output=True, text=True)
if '5000M' in result.stdout:
    print("USB 3.0 detected")
else:
    print("WARNING: USB 2.0 detected - Lower performance")

pipeline = rs.pipeline()

resolutions = [
    (640, 480, 30),
    (640, 480, 15),
    (640, 480, 6),
    (424, 240, 30),
    (424, 240, 15),
    (424, 240, 6)
]

width, height, fps = 640, 480, 30
for w, h, f in resolutions:
    try:
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)

        profile = pipeline.start(config)

        for _ in range(5):
            test_frames = pipeline.wait_for_frames(timeout_ms=2000)
            if test_frames.get_color_frame():
                print(f"Working configuration: {w}x{h} @ {f}fps")
                width, height, fps = w, h, f
                break
        break
    except Exception as e:
        print(f"  {w}x{h}@{f}fps failed: {e}")
        try:
            pipeline.stop()
        except:
            pass
        continue

align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print(f"RealSense ready: {width}x{height} @ {fps}fps")

def process_hybrik(image, det_model, hybrik_model, transformation, device):

    det_input = det_transform(image).to(device)
    det_output = det_model([det_input])[0]

    tight_bbox = get_one_box(det_output)
    if tight_bbox is None:
        return None

    pose_input, bbox, img_center = transformation.test_transform(image, tight_bbox)
    pose_input = pose_input.to(device)[None, :, :, :]

    pose_output = hybrik_model(
        pose_input,
        flip_test=True,
        bboxes=torch.from_numpy(np.array(bbox)).to(device).unsqueeze(0).float(),
        img_center=torch.from_numpy(img_center).to(device).unsqueeze(0).float()
    )

    uv_coords = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2].detach().cpu().numpy()

    input_size_w = cfg.MODEL.IMAGE_SIZE[0]
    input_size_h = cfg.MODEL.IMAGE_SIZE[1]

    coords_in_patch_x = (uv_coords[:, 0] + 0.5) * input_size_w
    coords_in_patch_y = (uv_coords[:, 1] + 0.5) * input_size_h

    scale_x = bbox[2] / input_size_w
    scale_y = bbox[3] / input_size_h

    joints_2d_pixel = np.zeros_like(uv_coords)
    joints_2d_pixel[:, 0] = coords_in_patch_x * scale_x + bbox[0]
    joints_2d_pixel[:, 1] = coords_in_patch_y * scale_y + bbox[1]

    if hasattr(pose_output, 'pred_xyz_jts_29'):
        joints_3d = pose_output.pred_xyz_jts_29.reshape(29, 3).detach().cpu().numpy()
    else:
        joints_3d = pose_output.pred_uvd_jts.reshape(29, 3).detach().cpu().numpy()

    return {
        'joints_2d': joints_2d_pixel,
        'joints_3d': joints_3d,
        'bbox': bbox,
        'pose_output': pose_output
    }

print("Starting demo. Press 'q' to quit")

frame_count = 0
fps_history = []

try:
    while True:
        start_time = time.time()

        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
        except:
            continue

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        result = process_hybrik(image_rgb, det_model, hybrik_model, transformation, device)

        if result is not None:
            joints_2d = result['joints_2d']
            joints_3d = result['joints_3d']
            bbox = result['bbox']

            h2o_joints = joints_3d[h2o_keypoint_indices]

            vis_image = color_image.copy()

            connections = [
                (16, 18), (18, 20),
                (17, 19), (19, 21),
                (16, 17),
                (0, 1), (1, 2),
                (0, 3), (3, 6),
                (0, 4), (4, 7),
                (7, 8)
            ]

            for start, end in connections:
                if start < len(joints_2d) and end < len(joints_2d):
                    pt1 = (int(joints_2d[start][0]), int(joints_2d[start][1]))
                    pt2 = (int(joints_2d[end][0]), int(joints_2d[end][1]))
                    if 0 <= pt1[0] < width and 0 <= pt1[1] < height:
                        if 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                            cv2.line(vis_image, pt1, pt2, (255, 255, 0), 2)

            joint_labels = {
                0: "Pelvis", 7: "L_Ankle", 8: "R_Ankle",
                16: "L_Shoulder", 17: "R_Shoulder",
                18: "L_Elbow", 19: "R_Elbow",
                20: "L_Wrist", 21: "R_Wrist",
                15: "Head", 12: "Neck"
            }

            for i, (x, y) in enumerate(joints_2d):
                if 0 <= x < width and 0 <= y < height:
                    is_h2o = i in h2o_keypoint_indices
                    color = (0, 0, 255) if is_h2o else (0, 255, 0)
                    size = 7 if is_h2o else 4
                    cv2.circle(vis_image, (int(x), int(y)), size, color, -1)

                    if i in joint_labels:
                        label = joint_labels[i]
                        cv2.putText(vis_image, label, (int(x)+8, int(y)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                   (255, 255, 255) if is_h2o else (200, 200, 200), 1)

                    if is_h2o and depth_image is not None:
                        depth_value = depth_image[int(y), int(x)]
                        if depth_value > 0:
                            cv2.circle(vis_image, (int(x), int(y)), 10, (255, 0, 0), 2)
                            depth_m = depth_value * depth_scale
                            cv2.putText(vis_image, f"{depth_m:.2f}m",
                                       (int(x)+12, int(y)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            cv2.rectangle(vis_image,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                         (255, 255, 255), 1)

            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history) if fps_history else 0

            info = [
                f"RealSense @ {width}x{height}",
                f"FPS: {avg_fps:.1f} ({device.type.upper()})",
                f"H2O Keypoints: {len(h2o_keypoint_indices)}",
                f"Depth: Available"
            ]

            y_pos = 30
            for text in info:
                cv2.putText(vis_image, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            for i, (x, y) in enumerate(joints_2d):
                if 0 <= x < width and 0 <= y < height:
                    is_h2o = i in h2o_keypoint_indices
                    if is_h2o:
                        cv2.circle(depth_colormap, (int(x), int(y)), 5, (255, 255, 255), -1)

            combined = np.hstack((vis_image, depth_colormap))
            cv2.imshow("HybrIK + RealSense", combined)

        else:
            cv2.putText(color_image, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("HybrIK + RealSense", combined)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: {avg_fps:.1f} FPS (Person {'detected' if result else 'not detected'})")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if fps_history:
        print(f"Performance:")
        print(f"  Average FPS: {np.mean(fps_history):.1f}")
        print(f"  Device: {device.type.upper()}")

    print("Demo stopped")