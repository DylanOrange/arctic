import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.pointnet import PointNetfeat


class Upsampler(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.upsampling = torch.nn.Linear(in_dim, out_dim)

    def forward(self, pred_vertices_sub):
        temp_transpose = pred_vertices_sub.transpose(1, 2)
        pred_vertices = self.upsampling(temp_transpose)
        pred_vertices = pred_vertices.transpose(1, 2)
        return pred_vertices


class RegressHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
        )

    def forward(self, x):
        dist = self.network(x).permute(0, 2, 1)[:, :, 0]
        return dist


class FieldSF(nn.Module):
    def __init__(self, backbone, focal_length, img_res):
        super().__init__()
        # if backbone == "resnet18":
        #     from src.nets.backbone.resnet import resnet18 as resnet
        # elif backbone == "resnet50":
        #     from src.nets.backbone.resnet import resnet50 as resnet
        # else:
        #     assert False
        # self.backbone = resnet(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)

        img_down_dim = 512
        img_mid_dim = 512
        pt_out_dim = 512
        self.down = nn.Sequential(
            nn.Linear(feat_dim, img_mid_dim),
            nn.ReLU(),
            nn.Linear(img_mid_dim, img_down_dim),
            nn.ReLU(),
        )  # downsize image features

        pt_shallow_dim = 512
        pt_mid_dim = 512
        self.point_backbone = PointNetfeat(
            input_dim=3 + img_down_dim,
            shallow_dim=pt_shallow_dim,
            mid_dim=pt_mid_dim,
            out_dim=pt_out_dim,
        )

        # self.point_backbone_h = PointNetfeat(
        #     input_dim=1,
        #     shallow_dim=128,
        #     mid_dim=128,
        #     out_dim=128,
        # )

        # self.point_backbone_o = PointNetfeat(
        #     input_dim=2,
        #     shallow_dim=128,
        #     mid_dim=128,
        #     out_dim=128,
        # )

        # self.residual_point_backbone = PointNetfeat(
        #     input_dim=3 + img_down_dim + 256,
        #     shallow_dim=pt_shallow_dim,
        #     mid_dim=pt_mid_dim,
        #     out_dim=pt_out_dim,
        # )

        pts_dim = pt_shallow_dim + pt_out_dim
        self.dist_head_or = RegressHead(pts_dim)
        self.dist_head_ol = RegressHead(pts_dim)
        self.dist_head_ro = RegressHead(pts_dim)
        self.dist_head_lo = RegressHead(pts_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.dist_residual_or = RegressHead(pts_dim)
        # self.dist_residual_ol = RegressHead(pts_dim)
        # self.dist_residual_ro = RegressHead(pts_dim)
        # self.dist_residual_lo = RegressHead(pts_dim)

        self.num_v_sub = 195  # mano subsampled
        self.num_v_o_sub = 300 * 2  # object subsampled
        self.num_v_o = 4000  # object
        self.upsampling_r = Upsampler(self.num_v_sub, 778)
        self.upsampling_l = Upsampler(self.num_v_sub, 778)
        self.upsampling_o = Upsampler(self.num_v_o_sub, self.num_v_o)

    def _decode(self, pts_all_feat):
        pts_all_feat = self.point_backbone(pts_all_feat)[0]#64,1024,990
        pts_r_feat, pts_l_feat, pts_o_feat = torch.split(
            pts_all_feat,
            [self.num_mano_pts, self.num_mano_pts, self.num_object_pts],
            dim=2,
        )

        dist_ro = self.dist_head_ro(pts_r_feat)#64,1024,195->64,195
        dist_lo = self.dist_head_lo(pts_l_feat)
        dist_or = self.dist_head_or(pts_o_feat)
        dist_ol = self.dist_head_ol(pts_o_feat)

        return dist_ro, dist_lo, dist_or, dist_ol

    # def residual(self, pts_all_feat):
    #     pts_all_feat = self.residual_point_backbone(pts_all_feat)[0]#64,1024,990
    #     pts_r_feat, pts_l_feat, pts_o_feat = torch.split(
    #         pts_all_feat,
    #         [self.num_mano_pts, self.num_mano_pts, self.num_object_pts],
    #         dim=2,
    #     )

    #     dist_ro = self.dist_residual_or(pts_r_feat)#64,1024,195->64,195
    #     dist_lo = self.dist_residual_ol(pts_l_feat)
    #     dist_or = self.dist_residual_ro(pts_o_feat)
    #     dist_ol = self.dist_residual_lo(pts_o_feat)

    #     return dist_ro, dist_lo, dist_or, dist_ol
    
    def forward(self, img_feat, meta_info):
        # images = inputs["img"]#64,3,224,224

        #change to keypoint estimation
        points_r = meta_info["v0.r"].permute(0, 2, 1)[:, :, :21]#64,3,21, exclude keyjoints
        points_l = meta_info["v0.l"].permute(0, 2, 1)[:, :, :21]#64,3,21
        points_o = meta_info["v0.o.kp"].permute(0, 2, 1)#64,3,32
        points_all = torch.cat((points_r, points_l, points_o), dim=2)#64,3,74

        # #get single image feature vector
        # img_feat = self.backbone(images)#64,2048,7,7
        img_feat = self.avgpool(img_feat).view(img_feat.shape[0], -1)#64,2048
        pred_vec = img_feat.clone()

        #downsize image feature vector
        img_feat = self.down(img_feat)#64,512

        self.num_mano_pts = points_r.shape[2]
        self.num_object_pts = points_o.shape[2]

        #repeat image feature vector
        img_feat_all = img_feat[:, :, None].repeat(
            1, 1, self.num_mano_pts * 2 + self.num_object_pts
        )#64,512,990

        #concat points coordinates with image vector
        pts_all_feat = torch.cat((points_all, img_feat_all), dim=1)#64,515,990
        dist_ro, dist_lo, dist_or, dist_ol = self._decode(pts_all_feat)

        # #refinement
        # dist_o = torch.stack([field["dist.or.kp"], field["dist.ol.kp"]], dim=1)

        # dist_feat_r = self.point_backbone_h(field["dist.ro.kp"][:,None,:])[0]#64,256,195
        # dist_feat_l = self.point_backbone_h(field["dist.lo.kp"][:,None,:])[0]
        # dist_feat_o = self.point_backbone_o(dist_o)[0]

        # dist_feat = torch.cat([dist_feat_r, dist_feat_l, dist_feat_o], dim=2)#64,256,990
        # refine_all_feat = torch.cat([points_all, img_feat_all, dist_feat], dim=1)
        # dist_ro_residual, dist_lo_residual, dist_or_residual, dist_ol_residual = self.residual(refine_all_feat)
        # #
        # dist_ro += dist_ro_residual
        # dist_lo += dist_lo_residual
        # dist_or += dist_or_residual
        # dist_ol += dist_ol_residual

        # #upsampling
        # dist_ro = self.upsampling_r((dist_ro)[:, :, None])[:, :, 0]#64,778
        # dist_lo = self.upsampling_l((dist_lo)[:, :, None])[:, :, 0]#64,778
        # dist_or = self.upsampling_o((dist_or)[:, :, None])[:, :, 0]#64,4000
        # dist_ol = self.upsampling_o((dist_ol)[:, :, None])[:, :, 0]#64,4000

        out = xdict()
        out["dist.ro.kp"] = dist_ro#778
        out["dist.lo.kp"] = dist_lo#778
        out["dist.or.kp"] = dist_or#4000
        out["dist.ol.kp"] = dist_ol#4000
        out["feat_vec"] = pred_vec
        return out
