import torch

import src.utils.interfield as inter
from src.nets.pointnet_utils import normalize_point_cloud_torch
from pytorch3d.ops import knn_points

def prepare_mano_template(batch_size, mano_layer, mesh_sampler, is_right):
    root_idx = 0

    # Generate T-pose template mesh
    template_pose = torch.zeros((1, 48))
    template_pose = template_pose.cuda()
    template_betas = torch.zeros((1, 10)).cuda()
    out = mano_layer(
        betas=template_betas,
        hand_pose=template_pose[:, 3:],
        global_orient=template_pose[:, :3],
        transl=None,
    )
    template_3d_joints = out.joints
    template_vertices = out.vertices
    template_vertices_sub = mesh_sampler.downsample(template_vertices, is_right)

    _, idx, _ = knn_points(template_3d_joints, template_vertices, None, None, K=1, return_nn=True)
    
    # normalize
    template_root = template_3d_joints[:, root_idx, :]
    template_3d_joints = template_3d_joints - template_root[:, None, :]
    template_vertices = template_vertices - template_root[:, None, :]
    template_vertices_sub = template_vertices_sub - template_root[:, None, :]

    # concatinate template joints and template vertices, and then duplicate to batch size
    ref_vertices = torch.cat([template_3d_joints, template_vertices_sub], dim=1)
    ref_vertices = ref_vertices.expand(batch_size, -1, -1)

    ref_vertices_full = torch.cat([template_3d_joints, template_vertices], dim=1)
    ref_vertices_full = ref_vertices_full.expand(batch_size, -1, -1)
    return ref_vertices, ref_vertices_full, idx


def prepare_templates(
    batch_size,
    mano_r,
    mano_l,
    mesh_sampler,
    arti_head,
    query_names,
):
    #joints(21)+sub_vertics(195), joints(21)+vertics(778)
    v0_r, v0_r_full, idx_r = prepare_mano_template(
        batch_size, mano_r, mesh_sampler, is_right=True
    )
    v0_l, v0_l_full, idx_l = prepare_mano_template(
        batch_size, mano_l, mesh_sampler, is_right=False
    )
    #sub_vertics(600) + vertics(4000)
    (v0_o, pidx, v0_full, mask, v0_o_kp, idx_o) = prepare_object_template(
        batch_size,
        arti_head.object_tensors,
        query_names,
    )
    CAM_R, CAM_L, CAM_O = list(range(100))[-3:]
    cams = (
        torch.FloatTensor([CAM_R, CAM_L, CAM_O]).view(1, 3, 1).repeat(batch_size, 1, 3)
        / 100
    )
    cams = cams.to(v0_r.device)
    return (
        v0_r,
        v0_l,
        v0_o,
        v0_o_kp,
        pidx,
        v0_r_full,
        v0_l_full,
        v0_full,
        mask,
        cams,
        idx_r,
        idx_l,
        idx_o
    )


def prepare_object_template(batch_size, object_tensors, query_names):
    template_angles = torch.zeros((batch_size, 1)).cuda()
    template_rot = torch.zeros((batch_size, 3)).cuda()
    out = object_tensors.forward(
        angles=template_angles,
        global_orient=template_rot,
        transl=None,
        query_names=query_names,
    )
    ref_vertices = out["v_sub"]
    parts_idx = out["parts_ids"]

    mask = out["mask"]

    ref_mean = ref_vertices.mean(dim=1)[:, None, :]
    ref_vertices -= ref_mean

    v_template = out["v"]
    #sub and ori in the same coordiante system
    v_template -= ref_mean

    ref_kp3d = out["kp3d"]
    ref_kp3d -= ref_mean

    _, idx_o, _ = knn_points(ref_kp3d, v_template, None, out["v_len"], K=1, return_nn=True)

    return (ref_vertices, parts_idx, v_template, mask, ref_kp3d, idx_o)

def prepare_nearest_vertex(targets, meta_info):

    _, idx_r, _ = knn_points(
        targets["mano.j3d.cam.r"], targets["mano.v3d.cam.r"], None, None, K=1, return_nn=True
    )

    _, idx_l, _ = knn_points(
        targets["mano.j3d.cam.l"], targets["mano.v3d.cam.l"], None, None, K=1, return_nn=True
    )

    _, idx_o, _ = knn_points(
        targets["object.kp3d.cam"], targets["object.v.cam"], None, targets["object.v_len"], K=1, return_nn=True
    )

    meta_info['nearest_r'] = idx_r
    meta_info['nearest_l'] = idx_l
    meta_info['nearest_o'] = idx_o

    return meta_info


def prepare_interfield(targets, max_dist):
    dist_min = 0.0
    dist_max = max_dist
    dist_ro, dist_ro_idx = inter.compute_dist_mano_to_obj(
        targets["mano.v3d.cam.r"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_lo, dist_lo_idx = inter.compute_dist_mano_to_obj(
        targets["mano.v3d.cam.l"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_or, dist_or_idx = inter.compute_dist_obj_to_mano(
        targets["mano.v3d.cam.r"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_ol, dist_ol_idx = inter.compute_dist_obj_to_mano(
        targets["mano.v3d.cam.l"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )

    targets["dist.ro"] = dist_ro
    targets["dist.lo"] = dist_lo
    targets["dist.or"] = dist_or
    targets["dist.ol"] = dist_ol

    targets["idx.ro"] = dist_ro_idx
    targets["idx.lo"] = dist_lo_idx
    targets["idx.or"] = dist_or_idx
    targets["idx.ol"] = dist_ol_idx
    return targets

def prepare_kp_interfield(targets, max_dist, alterkey = False):
    dist_min = 0.0
    dist_max = max_dist
    dist_ro, dist_ro_idx = inter.compute_dist_mano_to_obj(
        targets["mano.j3d.cam.r"],
        targets["object.kp3d.cam"],
        None,
        0.0,
        dist_max,
    )
    dist_lo, dist_lo_idx = inter.compute_dist_mano_to_obj(
        targets["mano.j3d.cam.l"],
        targets["object.kp3d.cam"],
        None,
        0.0,
        dist_max,
    )
    dist_or, dist_or_idx = inter.compute_dist_obj_to_mano(
        targets["mano.j3d.cam.r"],
        targets["object.kp3d.cam"],
        None,
        0.0,
        dist_max,
    )
    dist_ol, dist_ol_idx = inter.compute_dist_obj_to_mano(
        targets["mano.j3d.cam.l"],
        targets["object.kp3d.cam"],
        None,
        0.0,
        dist_max,
    )

    if alterkey:
        targets["dist.ro.kp.computed"] = dist_ro
        targets["dist.lo.kp.computed"] = dist_lo
        targets["dist.or.kp.computed"] = dist_or
        targets["dist.ol.kp.computed"] = dist_ol

        targets["idx.ro.kp.computed"] = dist_ro_idx
        targets["idx.lo.kp.computed"] = dist_lo_idx
        targets["idx.or.kp.computed"] = dist_or_idx
        targets["idx.ol.kp.computed"] = dist_ol_idx

    else:
        targets["dist.ro.kp"] = dist_ro
        targets["dist.lo.kp"] = dist_lo
        targets["dist.or.kp"] = dist_or
        targets["dist.ol.kp"] = dist_ol

        targets["idx.ro.kp"] = dist_ro_idx
        targets["idx.lo.kp"] = dist_lo_idx
        targets["idx.or.kp"] = dist_or_idx
        targets["idx.ol.kp"] = dist_ol_idx

    return targets

def prepare_norm_interfield(targets, max_dist):
    dist_min = 0.0
    dist_max = max_dist

    vertex_r, center, scale_r = normalize_point_cloud_torch(targets["mano.v3d.cam.r"])
    vertex_r_o = (targets["object.v.cam"] - center)/ scale_r

    vertex_l, center, scale_l = normalize_point_cloud_torch(targets["mano.v3d.cam.l"])
    vertex_l_o = (targets["object.v.cam"] - center)/ scale_l
      
    dist_ro, dist_ro_idx = inter.compute_dist_mano_to_obj(
        vertex_r,
        vertex_r_o,
        targets["object.v_len"],
        dist_min,
        (dist_max/scale_r).max(),
    )
    dist_lo, dist_lo_idx = inter.compute_dist_mano_to_obj(
        vertex_l,
        vertex_l_o,
        targets["object.v_len"],
        dist_min,
        (dist_max/scale_l).max(),
    )
    dist_or, dist_or_idx = inter.compute_dist_obj_to_mano(
        vertex_r,
        vertex_r_o,
        targets["object.v_len"],
        dist_min,
        (dist_max/scale_r).max(),
    )
    dist_ol, dist_ol_idx = inter.compute_dist_obj_to_mano(
        vertex_l,
        vertex_l_o,
        targets["object.v_len"],
        dist_min,
        (dist_max/scale_l).max(),
    )

    targets["dist.ro.norm"] = dist_ro
    targets["dist.lo.norm"] = dist_lo
    targets["dist.or.norm"] = dist_or
    targets["dist.ol.norm"] = dist_ol

    targets["idx.ro.norm"] = dist_ro_idx
    targets["idx.lo.norm"] = dist_lo_idx
    targets["idx.or.norm"] = dist_or_idx
    targets["idx.ol.norm"] = dist_ol_idx
    return targets

def prepare_normal(targets):
    
    ro_cloest_kp_idx = targets["idx.ro"].unsqueeze(-1).repeat(1, 1, 3)
    lo_cloest_kp_idx = targets["idx.lo"].unsqueeze(-1).repeat(1, 1, 3)
    ro_cloest_kp = targets["object.v.cam"].gather(dim=1, index = ro_cloest_kp_idx)
    lo_cloest_kp = targets["object.v.cam"].gather(dim=1, index = lo_cloest_kp_idx)

    or_cloest_kp_idx = targets['idx.or'].unsqueeze(-1).repeat(1, 1, 3)
    ol_cloest_kp_idx = targets['idx.ol'].unsqueeze(-1).repeat(1, 1, 3)
    or_cloest_kp = targets["mano.v3d.cam.r"].gather(dim=1, index = or_cloest_kp_idx)
    ol_cloest_kp = targets["mano.v3d.cam.l"].gather(dim=1, index = ol_cloest_kp_idx)
        
    vector_ro = ro_cloest_kp-targets["mano.v3d.cam.r"]
    vector_lo = lo_cloest_kp-targets["mano.v3d.cam.l"]
    vector_or = or_cloest_kp-targets["object.v.cam"]
    vector_ol = ol_cloest_kp-targets["object.v.cam"]

    dist_ro = torch.linalg.norm(vector_ro,dim=2,keepdim =True)
    dist_lo = torch.linalg.norm(vector_lo,dim=2,keepdim =True)
    dist_or = torch.linalg.norm(vector_or,dim=2,keepdim =True)
    dist_ol = torch.linalg.norm(vector_ol,dim=2,keepdim =True)
    
    direction_ro = vector_ro/dist_ro
    direction_lo = vector_lo/dist_lo
    direction_or = vector_or/dist_or
    direction_ol = vector_ol/dist_ol
    
    targets["direc.ro"] = direction_ro#778
    targets["direc.lo"] = direction_lo#778
    targets["direc.or"] = direction_or#4000
    targets["direc.ol"] = direction_ol#4000
    
    targets["field.ro"] = vector_ro#778
    targets["field.lo"] = vector_lo#778
    targets["field.or"] = vector_or#4000
    targets["field.ol"] = vector_ol#4000

    return targets

def prepare_kp_normal(targets):
    
    ro_cloest_kp_idx = targets["idx.ro.kp"].unsqueeze(-1).repeat(1, 1, 3)
    lo_cloest_kp_idx = targets["idx.lo.kp"].unsqueeze(-1).repeat(1, 1, 3)
    # ro_cloest_kp = targets["object.v.cam"].gather(dim=1, index = ro_cloest_kp_idx)
    # lo_cloest_kp = targets["object.v.cam"].gather(dim=1, index = lo_cloest_kp_idx)

    ro_cloest_kp = targets["object.kp3d.cam"].gather(dim=1, index = ro_cloest_kp_idx)
    lo_cloest_kp = targets["object.kp3d.cam"].gather(dim=1, index = lo_cloest_kp_idx)

    or_cloest_kp_idx = targets['idx.or.kp'].unsqueeze(-1).repeat(1, 1, 3)
    ol_cloest_kp_idx = targets['idx.ol.kp'].unsqueeze(-1).repeat(1, 1, 3)

    or_cloest_kp = targets["mano.j3d.cam.r"].gather(dim=1, index = or_cloest_kp_idx)
    ol_cloest_kp = targets["mano.j3d.cam.l"].gather(dim=1, index = ol_cloest_kp_idx)
        
    vector_ro = targets["mano.j3d.cam.r"]-ro_cloest_kp
    vector_lo = targets["mano.j3d.cam.l"]-lo_cloest_kp
    vector_or = targets["object.kp3d.cam"]-or_cloest_kp
    vector_ol = targets["object.kp3d.cam"]-ol_cloest_kp

    dist_ro = torch.linalg.norm(vector_ro,dim=2,keepdim =True)
    dist_lo = torch.linalg.norm(vector_lo,dim=2,keepdim =True)
    dist_or = torch.linalg.norm(vector_or,dim=2,keepdim =True)
    dist_ol = torch.linalg.norm(vector_ol,dim=2,keepdim =True)
    
    direction_ro = vector_ro/dist_ro
    direction_lo = vector_lo/dist_lo
    direction_or = vector_or/dist_or
    direction_ol = vector_ol/dist_ol
    
    targets["direc.ro"] = direction_ro#778
    targets["direc.lo"] = direction_lo#778
    targets["direc.or"] = direction_or#4000
    targets["direc.ol"] = direction_ol#4000
    
    targets["field.ro"] = vector_ro#778
    targets["field.lo"] = vector_lo#778
    targets["field.or"] = vector_or#4000
    targets["field.ol"] = vector_ol#4000

    return targets