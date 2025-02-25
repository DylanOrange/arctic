import torch
from scipy.spatial.distance import cdist

import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import src.callbacks.process.process_generic as generic
import src.utils.interfield as inter


def process_data(
    models, inputs, targets, meta_info, mode, args, field_max=float("inf")
):
    
    # add cano coordinates input, delete when run field SF
    batch_size = meta_info["intrinsics"].shape[0]

    (
        v0_r,
        v0_l,
        v0_o,
        v0_o_kp,
        pidx,
        v0_r_full,
        v0_l_full,
        v0_o_full,
        mask,
        cams,
    ) = generic.prepare_templates(
        batch_size,
        models["mano_r"],
        models["mano_l"],
        models["mesh_sampler"],
        models["arti_head"],
        meta_info["query_names"],
    )

    meta_info["v0.r"] = v0_r#64,216,3
    meta_info["v0.l"] = v0_l#64,216,3
    meta_info["v0.o"] = v0_o#64,600,3
    meta_info["v0.o.kp"] = v0_o_kp#64,600,3
    meta_info["cams0"] = cams
    meta_info["parts_idx"] = pidx
    meta_info["v0.r.full"] = v0_r_full#64,799,3
    meta_info["v0.l.full"] = v0_l_full#64,799,3
    meta_info["v0.o.full"] = v0_o_full#64,3997,3
    meta_info["mask"] = mask

    #to find the index of subsampled points in the full point cloud
    N_joint = 21
    # ridx = torch.argmin(torch.cdist(v0_r[:,N_joint:], v0_r_full[:,N_joint:], p=2), dim=2)#64,195
    # lidx = torch.argmin(torch.cdist(v0_l[:,N_joint:], v0_l_full[:,N_joint:], p=2), dim=2)#64,195
    # oidx = torch.argmin(torch.cdist(v0_o, v0_o_full, p=2), dim=2)#64,600

    # B = ridx.shape[0]
    img_res = 224
    K = meta_info["intrinsics"]
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
    gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

    gt_kp2d_b = targets["object.kp2d.norm.b"]  # 2D keypoints for object base
    gt_object_rot = targets["object.rot"].view(-1, 3)

    # pose the object without translation (call it object cano space)
    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )
    diameters = out["diameter"]
    parts_idx = out["parts_ids"]
    meta_info["part_ids"] = parts_idx
    meta_info["diameter"] = diameters

    # targets keypoints of hand and objects are in camera coord (full resolution image) space
    # map all entities from camera coord to object cano space based on the rigid-transform
    # between the object base keypoints in camera coord and object cano space
    # since R, T is used, relative distance btw hand and object is preserved
    num_kps = out["kp3d"].shape[1] // 2
    kp3d_b_cano = out["kp3d"][:, num_kps:]
    R0, T0 = tf.batch_solve_rigid_tf(targets["object.kp3d.full.b"], kp3d_b_cano)
    joints3d_r0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.r"], R0, T0)
    joints3d_l0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.l"], R0, T0)

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints
    gt_vertices_r = gt_out_r.vertices
    gt_root_cano_r = gt_out_r.joints[:, 0]

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints#joints:21
    gt_vertices_l = gt_out_l.vertices#vertics:778
    gt_root_cano_l = gt_out_l.joints[:, 0]

    # map MANO mesh to object canonical space, there is a translation between object cano joint coordinate and posed mano space coordinate
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_model_joints_l = joints3d_l0
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]

    # now that everything is in the object canonical space
    # find camera translation for rendering relative to the object

    # unnorm 2d keypoints
    gt_kp2d_b_cano = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)

    # estimate camera translation by solving 2d to 3d correspondence
    gt_transl = camera.estimate_translation_k(
        kp3d_b_cano,
        gt_kp2d_b_cano,
        meta_info["intrinsics"].cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    )

    # move to camera coord
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]
    gt_model_joints_r = gt_model_joints_r + gt_transl[:, None, :]
    gt_model_joints_l = gt_model_joints_l + gt_transl[:, None, :]

    ####
    gt_kp3d_o = out["kp3d"] + gt_transl[:, None, :]
    gt_bbox3d_o = out["bbox3d"] + gt_transl[:, None, :]

    # roots
    gt_root_cam_patch_r = gt_model_joints_r[:, 0]
    gt_root_cam_patch_l = gt_model_joints_l[:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l
    gt_cam_t_o = gt_transl

    targets["mano.cam_t.r"] = gt_cam_t_r
    targets["mano.cam_t.l"] = gt_cam_t_l
    targets["object.cam_t"] = gt_cam_t_o

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length,    img_res
    )

    gt_cam_t_wp_o = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_o, avg_focal_length, img_res
    )

    targets["mano.cam_t.wp.r"] = gt_cam_t_wp_r
    targets["mano.cam_t.wp.l"] = gt_cam_t_wp_l
    targets["object.cam_t.wp"] = gt_cam_t_wp_o

    # cam coord of patch
    targets["object.cam_t.kp3d.b"] = gt_transl

    targets["mano.v3d.cam.r"] = gt_vertices_r
    targets["mano.v3d.cam.l"] = gt_vertices_l
    targets["mano.j3d.cam.r"] = gt_model_joints_r
    targets["mano.j3d.cam.l"] = gt_model_joints_l
    targets["object.kp3d.cam"] = gt_kp3d_o#keypoints:32
    targets["object.bbox3d.cam"] = gt_bbox3d_o

    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )

    # GT vertices relative to right hand root
    targets["object.v.cam"] = out["v"] + gt_transl[:, None, :]#vertics:3997
    targets["object.v_len"] = out["v_len"]#number of vertics the object in every batch have

    targets["object.f"] = out["f"]
    targets["object.f_len"] = out["f_len"]

    targets = generic.prepare_kp_interfield(targets, max_dist = args.max_dist)
    targets = generic.prepare_interfield(targets, max_dist = args.max_dist)

    # dist_or, _ = inter.compute_dist_obj_to_mano(
    #     targets["mano.v3d.cam.r"],
    #     targets["object.kp3d.cam"],
    #     None,
    #     0.0,
    #     field_max,
    # )
    # dist_ol, _ = inter.compute_dist_obj_to_mano(
    #     targets["mano.v3d.cam.l"],
    #     targets["object.kp3d.cam"],
    #     None,
    #     0.0,
    #     field_max,
    # )
    
    # meta_info["dist.ro"] = targets["dist.ro"][torch.arange(B).unsqueeze(1), ridx].clone()#64,778->64,195
    # meta_info["dist.lo"] = targets["dist.lo"][torch.arange(B).unsqueeze(1), lidx].clone()#64,778->64,195
    # meta_info["dist.or"] = targets["dist.or"][torch.arange(B).unsqueeze(1), oidx].clone()#64,3997->64,600
    # meta_info["dist.ol"] = targets["dist.ol"][torch.arange(B).unsqueeze(1), oidx].clone()#64,3997->64,600

    #the first 21 points are joints
    # meta_info["dist.ro"] = targets["dist.ro"][:, :N_joint].clone()#64,778->64,21
    # meta_info["dist.lo"] = targets["dist.lo"][:, :N_joint].clone()#64,778->64,21
    # meta_info["dist.or"] = dist_or.clone()#64,32
    # meta_info["dist.ol"] = dist_ol.clone()#64,32

    return inputs, targets, meta_info
