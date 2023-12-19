import torch
import torch.nn as nn

from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead


class Regressor(nn.Module):
    def __init__(self, focal_length, img_res):
        super(Regressor, self).__init__()

        self.mano_r = MANOHead(is_rhand=True, focal_length=focal_length, img_res=img_res)
        self.mano_l = MANOHead(is_rhand=False, focal_length=focal_length, img_res=img_res)
        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)

    def forward(self, hmr_output_r, hmr_output_l, hmr_output_o, meta_info):

        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_o = hmr_output_o["cam_t.wp"]

        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=root_l,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_o["rot"],
            angle=hmr_output_o["radian"],
            query_names=query_names,
            cam=root_o,
            K=K,
        )

        return mano_output_r, mano_output_l, arti_output