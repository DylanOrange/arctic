import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer


class HandHMR(nn.Module):
    def __init__(self, feat_dim, is_rhand, n_iter):
        super().__init__()
        self.is_rhand = is_rhand

        hand_specs = {"pose_6d": 6 * 16, "cam_t/wp": 3, "shape": 10}
        self.hmr_layer = HMRLayer(feat_dim, 1024, hand_specs)

        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.hand_specs = hand_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, features):
        #initialize pose, shape and camera paramter 
        batch_size = features.shape[0]
        dev = features.device
        init_pose = (
            rot_conv.matrix_to_rotation_6d(
                rot_conv.axis_angle_to_matrix(torch.zeros(16, 3))
            )
            .reshape(1, -1)
            .repeat(batch_size, 1)
        )#one 6d rotation representation, but the DOF is still 3
        init_shape = torch.zeros(1, 10).repeat(batch_size, 1)

        #initialize camera paramter using a cam_init head
        init_transl = self.cam_init(features)

        out = {}
        out["pose_6d"] = init_pose#64,16*6
        out["shape"] = init_shape#64,10
        out["cam_t/wp"] = init_transl#64,3
        out = xdict(out).to(dev)
        return out

    def forward(self, features, use_pool=True):
        batch_size = features.shape[0]

        #to get a single image feature vector
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        init_vdict = self.init_vector_dict(feat)
        init_cam_t = init_vdict["cam_t/wp"].clone()

        #refinement
        pred_vdict = self.hmr_layer(feat, init_vdict, self.n_iter)

        pred_rotmat = rot_conv.rotation_6d_to_matrix(
            pred_vdict["pose_6d"].reshape(-1, 6)
        ).view(batch_size, 16, 3, 3)

        pred_vdict["pose"] = pred_rotmat#64,16,3,3
        pred_vdict["cam_t.wp.init"] = init_cam_t
        pred_vdict = pred_vdict.replace_keys("/", ".")
        return pred_vdict
