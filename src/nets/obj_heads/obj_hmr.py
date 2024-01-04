import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer


class ObjectHMR(nn.Module):
    def __init__(self, feat_dim, n_iter):
        super().__init__()

        obj_specs = {"rot": 3, "cam_t/wp": 3, "radian": 1}
        self.hmr_layer = HMRLayer(feat_dim+512, 1024, obj_specs)

        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim+512+3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.obj_specs = obj_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, features, noise):
        batch_size = features.shape[0]
        dev = features.device
        # init_rot = torch.zeros(batch_size, 3)
        # init_angle = torch.zeros(batch_size, 1)

        init_rot = noise[:,:3]
        init_angle = noise[:,3].unsqueeze(-1)

        init_transl = self.cam_init(torch.concat([features, noise[:,-3:]], dim=-1))

        out = {}
        out["rot"] = init_rot
        out["radian"] = init_angle
        out["cam_t/wp"] = init_transl
        out = xdict(out).to(dev)
        return out

    def forward(self, features, noise, time, use_pool=True):
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        feat = torch.concat([feat, time], dim=1)
        init_vdict = self.init_vector_dict(feat, noise)
        init_cam_t = init_vdict["cam_t/wp"].clone()
        pred_vdict = self.hmr_layer(feat, init_vdict, self.n_iter)
        pred_vdict["cam_t.wp.init"] = init_cam_t
        pred_vdict = pred_vdict.replace_keys("/", ".")
        return pred_vdict
