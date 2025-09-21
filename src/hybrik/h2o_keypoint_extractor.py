#!/usr/bin/env python3
"""
H2O Joint Extractor for G1 Teleoperation
Extracts H2O keypoints from HybrIK pose estimation
"""
import sys
import os

HYBRIK_PATH = os.environ.get('HYBRIK_PATH', '/home/vb/workspace/HybrIK')
sys.path.append(HYBRIK_PATH)

import torch
import numpy as np
import cv2
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class H2OJointExtractor:
    def __init__(self, config_file=None, checkpoint_file=None):
        """Initialize the H2O joint extractor"""
        
        if config_file is None:
            config_file = os.path.join(HYBRIK_PATH, 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml')
        if checkpoint_file is None:
            checkpoint_file = os.path.join(HYBRIK_PATH, 'pretrained_models/hybrik_hrnet.pth')
        
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        
        # H2O keypoint indices: [7, 8, 16, 17, 18, 19, 20, 21]
        self.h2o_keypoint_indices = [7, 8, 16, 17, 18, 19, 20, 21]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_models()
        
    def init_models(self):
        """Initialize HybrIK and detection models"""
        print("Initializing H2O Joint Extractor...")
        
        self.cfg = update_config(self.config_file)
        self.hybrik_model = builder.build_sppe(self.cfg.MODEL)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_file, map_location=self.device)
        if 'model' in checkpoint:
            self.hybrik_model.load_state_dict(checkpoint['model'], strict=False)
        else:
            self.hybrik_model.load_state_dict(checkpoint, strict=False)
        
        self.detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.hybrik_model.to(self.device).eval()
        self.detection_model.to(self.device).eval()
        
        bbox_3d_shape = getattr(self.cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        
        dummy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })
        
        self.transformation = SimpleTransform3DSMPLCam(
            dummy_set,
            scale_factor=self.cfg.DATASET.SCALE_FACTOR,
            color_factor=self.cfg.DATASET.COLOR_FACTOR,
            occlusion=self.cfg.DATASET.OCCLUSION,
            input_size=self.cfg.MODEL.IMAGE_SIZE,
            output_size=self.cfg.MODEL.HEATMAP_SIZE,
            depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=self.cfg.DATASET.ROT_FACTOR,
            sigma=self.cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=self.cfg.LOSS['TYPE']
        )
        
        print("H2O Joint Extractor initialized!")
    
    def extract_full_output(self, image):
        """
        Extract complete HybrIK output including 2D/3D joints and SMPL parameters
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            dict with keys:
                - joints_3d: (29, 3) 3D joint positions
                - joints_2d: (29, 2) 2D joint positions in pixels
                - h2o_3d: (8, 3) H2O keypoints 3D
                - h2o_2d: (8, 2) H2O keypoints 2D
                - bbox: Detection bounding box
                - pred_phi: SMPL pose parameters
                - pred_shape: SMPL shape parameters
                - pred_vertices: SMPL mesh vertices
                - transl: Camera translation
        """
        original_height, original_width = image.shape[:2]
        downsampled = cv2.resize(image, (original_width//2, original_height//2), interpolation=cv2.INTER_LINEAR)
        if len(downsampled.shape) == 3 and downsampled.shape[2] == 3:
            image_rgb = cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = downsampled
        
        det_transform = T.ToTensor()
        det_input = det_transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            det_output = self.detection_model(det_input)[0]
        
        bbox = get_one_box(det_output)
        if bbox is None:
            return None
        
        scale_factor = 2
        bbox_original = [bbox[0]*scale_factor, bbox[1]*scale_factor, 
                        bbox[2]*scale_factor, bbox[3]*scale_factor]
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb_full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb_full = image
        
        pose_input, bbox_out, img_center = self.transformation.test_transform(image_rgb_full, bbox_original)
        pose_input = pose_input.to(self.device).unsqueeze(0).float()
        
        with torch.no_grad():
            pose_output = self.hybrik_model(
                pose_input,
                flip_test=False,
                bboxes=torch.from_numpy(np.array(bbox_out)).to(self.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(self.device).unsqueeze(0).float()
            )
        
        result = {}
        joints_3d = pose_output.pred_xyz_jts_29.reshape(29, 3).detach().cpu().numpy()
        result['joints_3d'] = joints_3d
        
        pred_uvd = pose_output.pred_uvd_jts.reshape(29, 3).detach().cpu().numpy()
        coords_x = (pred_uvd[:, 0] + 0.5) * self.cfg.MODEL.IMAGE_SIZE[0]
        coords_y = (pred_uvd[:, 1] + 0.5) * self.cfg.MODEL.IMAGE_SIZE[1]
        
        scale_x = bbox_out[2] / self.cfg.MODEL.IMAGE_SIZE[0]
        scale_y = bbox_out[3] / self.cfg.MODEL.IMAGE_SIZE[1]
        joints_2d = np.zeros((29, 2))
        joints_2d[:, 0] = coords_x * scale_x + bbox_out[0]
        joints_2d[:, 1] = coords_y * scale_y + bbox_out[1]
        result['joints_2d'] = joints_2d
        
        result['h2o_3d'] = joints_3d[self.h2o_keypoint_indices]
        result['h2o_2d'] = joints_2d[self.h2o_keypoint_indices]
        
        if hasattr(pose_output, 'pred_phi'):
            result['pred_phi'] = pose_output.pred_phi.detach().cpu().numpy()
        if hasattr(pose_output, 'pred_shape'):
            result['pred_shape'] = pose_output.pred_shape.detach().cpu().numpy()
        if hasattr(pose_output, 'pred_vertices'):
            result['pred_vertices'] = pose_output.pred_vertices.detach().cpu().numpy()
        if hasattr(pose_output, 'transl'):
            result['transl'] = pose_output.transl.detach().cpu().numpy()
        
        result['bbox'] = bbox_original
        
        return result
    
    def extract_h2o_keypoints(self, image):
        """
        Extract H2O keypoints only
        
        Returns:
            tuple: (h2o_3d, h2o_2d) or (None, None) if no person detected
        """
        full_output = self.extract_full_output(image)
        if full_output is None:
            return None, None
        return full_output['h2o_3d'], full_output['h2o_2d']
    
    def prepare_h2o_state(self, full_output):
        """
        Prepare H2O state vector for motion matching or RL policy
        
        Args:
            full_output: Output from extract_full_output()
            
        Returns:
            h2o_state: Normalized state vector for H2O pipeline
        """
        if full_output is None:
            return None
        
        h2o_3d = full_output['h2o_3d']
        
        # Center using average of shoulders
        center = (h2o_3d[2] + h2o_3d[3]) / 2.0  # shoulders
        h2o_centered = h2o_3d - center
        
        # Normalize using shoulder width
        shoulder_width = np.linalg.norm(h2o_3d[3] - h2o_3d[2])
        if shoulder_width > 0:
            h2o_normalized = h2o_centered / shoulder_width
        else:
            h2o_normalized = h2o_centered
        
        # Flatten to state vector
        h2o_state = h2o_normalized.flatten()
        
        return h2o_state

def extract_joint_positions(image, extractor=None):
    """Extract joint positions from image"""
    if extractor is None:
        extractor = H2OJointExtractor()
    
    full_output = extractor.extract_full_output(image)
    if full_output is None:
        return None
    
    # Return in legacy format
    return {
        'success': True,
        'all_joints_3d': full_output['joints_3d'],
        'all_joints_2d_pixel': full_output['joints_2d'],
        'h2o_keypoints_3d': full_output['h2o_3d'],
        'h2o_keypoints_2d': full_output['h2o_2d'],
        'vertices': full_output.get('pred_vertices'),
        'camera': full_output.get('transl'),
        'smpl_params': {
            'shape': full_output.get('pred_shape', []),
            'pose': full_output.get('pred_phi', []),
            'transl': full_output.get('transl', [])
        }
    }

if __name__ == "__main__":
    print("Testing H2O Joint Extractor...")
    
    # Test with a sample image
    extractor = H2OJointExtractor()
    
    # Create test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test extraction
    result = extractor.extract_full_output(test_img)
    if result is None:
        print("No person detected in test image (expected)")
    else:
        print(f"Extracted {len(result['joints_3d'])} 3D joints")
        print(f"H2O keypoints shape: {result['h2o_3d'].shape}")
    
    print("Extractor working!")