import torch
import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.obj_heads.obj_hmr import ObjectHMR


class InitRegressor(nn.Module):
    def __init__(self, backbone, pose_regressor):
        super(InitRegressor, self).__init__()

        feat_dim = get_backbone_info(backbone)["n_output_channels"]

        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)
        self.head_o = ObjectHMR(feat_dim, n_iter=3)
        self.regressor = pose_regressor
    
    def forward(self, features, meta_info):

        hmr_output_r = self.head_r(features)#64,2048,7,7 -> 64,2048
        hmr_output_l = self.head_l(features)
        hmr_output_o = self.head_o(features)

        #initial regressor
        mano_output_r, mano_output_l, arti_output = self.regressor(hmr_output_r, hmr_output_l, hmr_output_o, meta_info)

        #add init camera parameters
        # root_r_init = hmr_output_r["cam_t.wp.init"]
        # root_l_init = hmr_output_l["cam_t.wp.init"]
        # root_o_init = hmr_output_o["cam_t.wp.init"]
        # mano_output_r["cam_t.wp.init.r"] = root_r_init
        # mano_output_l["cam_t.wp.init.l"] = root_l_init
        # arti_output["cam_t.wp.init"] = root_o_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")

        output = xdict()
        output["mano_output_r"] = mano_output_r
        output["mano_output_l"] = mano_output_l
        output["arti_output"] = arti_output

        # output.merge(mano_output_r)
        # output.merge(mano_output_l)
        # output.merge(arti_output)

        output["hmr_output_r"] = hmr_output_r
        output["hmr_output_l"] = hmr_output_l
        output["hmr_output_o"] = hmr_output_o

        return output
