import torch
import torch.nn as nn
import pytorch3d.transforms.rotation_conversions as rot_conv
import torch.nn.functional as F

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer
from src.nets.backbone.residual import Residual
from src.nets.transformer import STE
from src.nets.imagejointfeaturefusion import ImageJointfeaturefusion
from src.nets.pointnet_utils import normalize_point_cloud_torch
from src.nets.pointnet import PointNetfeat, PointNet2, ReconDecoder, PointNetObject
from pytorch3d.ops import knn_points

class Refinement(nn.Module):
    def __init__(self, feat_dim, hand_joint_num, object_joint_num, hand_specs, obj_specs):
        super(Refinement, self).__init__()

        self.hand_joint_num = hand_joint_num
        self.object_joint_num = object_joint_num

        # self.fuse_img_joint_features_l = ImageJointfeaturefusion(256,128)
        # self.fuse_img_joint_features_r = ImageJointfeaturefusion(256,128)
        # self.fuse_img_joint_features_o = ImageJointfeaturefusion(256,128)

        # self.interaction_rl = STE(num_joints = (self.hand_joint_num * 2), in_chans=128, out_dim=64, depth=4)
        self.interaction_r = STE(num_joints = (self.hand_joint_num), in_chans=512, out_dim=256, depth=4)
        self.interaction_l = STE(num_joints = (self.hand_joint_num), in_chans=512, out_dim=256, depth=4)
        self.interaction_o = STE(num_joints = self.object_joint_num, in_chans=512, out_dim=256, depth=4)

        # self.refine_hmr_layer_l = HMRLayer(2048+512, 1024, hand_specs)
        # self.refine_hmr_layer_r = HMRLayer(2048+512, 1024, hand_specs)
        # self.refine_hmr_layer_o = HMRLayer(2048+512, 1024, obj_specs)
        self.refine_hmr_layer_l = HMRLayer(256*(self.hand_joint_num), 1024, hand_specs)
        self.refine_hmr_layer_r = HMRLayer(256*(self.hand_joint_num), 1024, hand_specs)
        # self.refine_hmr_layer_o = HMRLayer(2048+1024, 1024, obj_specs)
        self.refine_hmr_layer_o = HMRLayer(256*(self.object_joint_num), 1024, obj_specs)

        # self.point_backbone_r = PointNet2(in_channel=3+256)
        # self.point_backbone_l = PointNet2(in_channel=3+256)
        # self.point_backbone_or = PointNetObject(in_channel=3+256)
        # self.point_backbone_ol = PointNetObject(in_channel=3+256)
    
        self.point_backbone_h = PointNet2(in_channel=0)
        # # self.point_backbone_l = PointNet2(in_channel=3)
        self.point_backbone_o = PointNetObject(in_channel=0)
        # # self.point_backbone_ol = PointNetObject(in_channel=3)

        # self.point_backbone_o = PointNetfeat(input_dim = 3, shallow_dim=128, mid_dim=128, out_dim=128)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

    def prepare_hand(self,dict):

        out = {}
        out["pose_6d"] = dict["pose_6d"]#64,16*6
        out["shape"] = dict["shape"]#64,10
        out["cam_t/wp"] = dict["cam_t.wp"]#64,3
        out = xdict(out).to(dict["cam_t.wp"].device)

        return out
    
    def prepare_object(self,dict):

        out = {}
        out["rot"] = dict["rot"]#64,16*6
        out["radian"] = dict["radian"]#64,10
        out["cam_t/wp"] = dict["cam_t.wp"]#64,3
        out = xdict(out).to(dict["cam_t.wp"].device)

        return out
    
    def forward(self, img_feat, img_feat_full, field_feat_r, field_feat_l, field_feat_or, field_feat_ol,\
                mano_output_r, mano_output_l, arti_output, \
                hmr_output_r, hmr_output_l, hmr_output_o):
        
        # root_r = mano_output_r["mano.cam_t.r"][:,None,:]
        # root_l = mano_output_l["mano.cam_t.l"][:,None,:]

        # vertex_r_3d = mano_output_r["mano.v3d.cam.r"] - root_r
        # vertex_r_3d, center_r, scale_r = normalize_point_cloud_torch(vertex_r_3d)
        # vertex_l_3d = mano_output_l["mano.v3d.cam.l"] - root_l        
        # vertex_l_3d, center_l, scale_l = normalize_point_cloud_torch(vertex_l_3d)

        # vertex_o_3d = arti_output["object.v.cam"]
        # vertex_o_3d, center_o, scale_o = normalize_point_cloud_torch(vertex_o_3d)

        kp_r_3d, center_r, scale_r = normalize_point_cloud_torch(mano_output_r["mano.j3d.cam.r"])
        kp_l_3d, center_l, scale_l = normalize_point_cloud_torch(mano_output_l["mano.j3d.cam.l"])
        kp_o_3d, center_o, scale_o = normalize_point_cloud_torch(arti_output["object.kp3d.cam"])

        B = kp_r_3d.shape[0]
        # ro_cloest_kp_idx = find_knn(mano_output_r["mano.j3d.cam.r"], arti_output["object.kp3d.cam"], None).expand(B, self.hand_joint_num, 3)
        # lo_cloest_kp_idx = find_knn(mano_output_l["mano.j3d.cam.l"], arti_output["object.kp3d.cam"], None).expand(B, self.hand_joint_num, 3)
        # or_cloest_kp_idx = find_knn(arti_output["object.kp3d.cam"], mano_output_r["mano.j3d.cam.r"], None).expand(B, self.object_joint_num, 3)
        # ol_cloest_kp_idx = find_knn(arti_output["object.kp3d.cam"], mano_output_l["mano.j3d.cam.l"], None).expand(B, self.object_joint_num, 3)

        # ro_cloest_kp = arti_output["object.kp3d.cam"].gather(dim=1, index = ro_cloest_kp_idx)
        # lo_cloest_kp = arti_output["object.kp3d.cam"].gather(dim=1, index = lo_cloest_kp_idx)
        # or_cloest_kp = mano_output_r["mano.j3d.cam.r"].gather(dim=1, index = or_cloest_kp_idx)
        # ol_cloest_kp = mano_output_l["mano.j3d.cam.l"].gather(dim=1, index = ol_cloest_kp_idx)

        # field_feat_r = ro_cloest_kp-mano_output_r["mano.j3d.cam.r"]
        # field_feat_l = lo_cloest_kp-mano_output_l["mano.j3d.cam.l"]
        # field_feat_or = or_cloest_kp-arti_output["object.kp3d.cam"]
        # field_feat_ol = ol_cloest_kp-arti_output["object.kp3d.cam"]

        field_feat_r = kp_r_3d.permute(0,2,1)
        field_feat_l = kp_l_3d.permute(0,2,1)
        field_feat_o = kp_o_3d.permute(0,2,1)
        # field_feat_r = torch.concat([kp_r_3d, field_feat_r/scale_r], dim=2).permute(0,2,1)#16,6,778
        # field_feat_l = torch.concat([kp_l_3d, field_feat_l/scale_l], dim=2).permute(0,2,1)#16,6,778
        # field_feat_o = torch.concat([kp_o_3d, field_feat_or/scale_o,  field_feat_ol/scale_o], dim=2).permute(0,2,1)#16,6,778
        # field_feat_or = torch.concat([kp_o_3d, field_feat_or/scale_o], dim=2).permute(0,2,1)#16,6,778
        # field_feat_ol = torch.concat([kp_o_3d, field_feat_ol/scale_o], dim=2).permute(0,2,1)#16,6,778

        # field_feat_or = field_feat_or[:,:vertex_o_3d.shape[1]]
        # field_feat_ol = field_feat_ol[:,:vertex_o_3d.shape[1]]

        img_feat_r = F.grid_sample(img_feat, mano_output_r["mano.j2d.norm.r"].unsqueeze(1).detach()).squeeze(-2)#64,256,21
        img_feat_l = F.grid_sample(img_feat, mano_output_l["mano.j2d.norm.l"].unsqueeze(1).detach()).squeeze(-2)#64,256,21
        img_feat_o = F.grid_sample(img_feat, arti_output["object.kp2d.norm"].unsqueeze(1).detach()).squeeze(-2)#64,256,21

        # field_feat_r = torch.concat([vertex_r_3d, field_feat_r/scale_r, img_feat_r.permute(0,2,1)], dim=2).permute(0,2,1)#16,6,778
        # field_feat_l = torch.concat([vertex_l_3d, field_feat_l/scale_l, img_feat_l.permute(0,2,1)], dim=2).permute(0,2,1)#16,6,778
        # field_feat_or = torch.concat([vertex_o_3d, field_feat_or/scale_o, img_feat_o.permute(0,2,1)], dim=2).permute(0,2,1)#16,6,778
        # field_feat_ol = torch.concat([vertex_o_3d, field_feat_ol/scale_o, img_feat_o.permute(0,2,1)], dim=2).permute(0,2,1)#16,6,778

        # field_feat_r = vertex_r_3d.permute(0,2,1)#16,6,778
        # field_feat_l = vertex_l_3d.permute(0,2,1)#16,6,778

        spatial_field_feat_r = self.point_backbone_h(field_feat_r)#16,384,21 16,1088,21
        spatial_field_feat_l = self.point_backbone_h(field_feat_l)#16,384,21
        spatial_field_feat_o = self.point_backbone_o(field_feat_o)#48,512
        # spatial_field_feat_ol = self.point_backbone_o(field_feat_ol)#48,512
    
        # spatial_field_feat_r = self.point_backbone_h(field_feat_r)#16,384,21 16,1088,21
        # spatial_field_feat_l = self.point_backbone_h(field_feat_l)#16,384,21
        # spatial_field_feat_or = self.point_backbone_o(field_feat_or)#48,512
        # spatial_field_feat_ol = self.point_backbone_o(field_feat_ol)#48,512

        # spatial_field_feat_o = torch.amax(torch.stack([spatial_field_feat_or, spatial_field_feat_ol], dim=-1), dim=-1)
        
        # img_feat_full = self.avgpool(img_feat_full)
        # img_feat_full = img_feat_full.view(img_feat_full.size(0), -1)

        # img_feat_r = self.fuse_img_joint_features_r(img_feat, field_feat_r, mano_output_r["j2d.norm.r"], mano_output_r["j3d.cam.r"])#64,21,128
        # img_feat_l = self.fuse_img_joint_features_l(img_feat, field_feat_l, mano_output_l["j2d.norm.l"], mano_output_l["j3d.cam.l"])#64,21,128
        # img_feat_o = self.fuse_img_joint_features_o(img_feat, field_feat_o, arti_output["kp2d.norm"], arti_output["kp3d.cam"])#64,32,128

        # field_feat_o = self.point_backbone_o(arti_output["kp3d.cam"].permute(0, 2, 1))[0]#16,384,32
        # img_feat_o = F.grid_sample(img_feat, arti_output["kp2d.norm"].unsqueeze(1).detach()).squeeze(-2)#64,256,21

        img_feat_r = torch.concat([img_feat_r, spatial_field_feat_r], dim=1)
        img_feat_r = self.interaction_r(img_feat_r.permute(0,2,1))

        img_feat_l = torch.concat([img_feat_l, spatial_field_feat_l], dim=1)
        img_feat_l = self.interaction_l(img_feat_l.permute(0,2,1))

        img_feat_o = torch.concat([img_feat_o, spatial_field_feat_o], dim=1)
        img_feat_o = self.interaction_o(img_feat_o.permute(0,2,1))

        _,C,_ = img_feat_o.shape#64,21,128

        refine_dict_r = self.prepare_hand(hmr_output_r)
        refine_dict_l = self.prepare_hand(hmr_output_l)
        refine_dict_o = self.prepare_object(hmr_output_o)

        # refine_mano_output_r = self.refine_hmr_layer_r(torch.concat([img_feat_full, spatial_field_feat_r],dim=1), refine_dict_r, n_iter = 3)
        # refine_mano_output_l = self.refine_hmr_layer_l(torch.concat([img_feat_full, spatial_field_feat_l],dim=1), refine_dict_l, n_iter = 3)

        refine_mano_output_r = self.refine_hmr_layer_r(img_feat_r.reshape(B,C*(self.hand_joint_num)), refine_dict_r, n_iter = 3)
        refine_mano_output_l = self.refine_hmr_layer_l(img_feat_l.reshape(B,C*(self.hand_joint_num)), refine_dict_l, n_iter = 3)
        refine_mano_output_o = self.refine_hmr_layer_o(img_feat_o.reshape(B,C*(self.object_joint_num)), refine_dict_o, n_iter = 3)

        refine_mano_output_r["pose"] = rot_conv.rotation_6d_to_matrix(
            refine_mano_output_r["pose_6d"].reshape(-1, 6)).view(B, 16, 3, 3)
        refine_mano_output_r = refine_mano_output_r.replace_keys("/", ".")

        refine_mano_output_l["pose"] = rot_conv.rotation_6d_to_matrix(
            refine_mano_output_l["pose_6d"].reshape(-1, 6)).view(B, 16, 3, 3)
        refine_mano_output_l = refine_mano_output_l.replace_keys("/", ".")

        refine_mano_output_o = refine_mano_output_o.replace_keys("/", ".")

        return refine_mano_output_r, refine_mano_output_l, refine_mano_output_o
    
def find_knn(a, b, c):
    _, knn_idx, _ = knn_points(
        a, b, None, c, K=1, return_nn=True
    )
    # knn_dists = knn_dists.sqrt()[:, :, 0]

    # knn_dists = torch.clamp(knn_dists, dist_min, dist_max)
    return knn_idx