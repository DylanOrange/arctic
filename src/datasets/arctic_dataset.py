import json
import os.path as op
import time

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d


class ArcticDataset(Dataset):
    def __getitem__(self, index):
        imgname = self.imgnames[index]
        data = self.getitem(imgname)
        return data

    def getitem(self, imgname, load_rgb=True):
        # start_getitem = time.time()
        args = self.args
        # LOADING START

        # start_loading = time.time()

        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        if sid == 's03':
                        
            obj_name = seq_name.split("_")[0]
            view_idx = int(view_idx)

            seq_data = self.data[f"{sid}/{seq_name}"]

            data_bbox = seq_data["bbox"]
            data_params = seq_data["params"]

            vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]

            if view_idx == 0:
                intrx = data_params["K_ego"][vidx].copy()
            else:
                intrx = np.array(self.intris_mat[sid][view_idx - 1])

            # distortion parameters for egocam rendering
            dist = data_params["dist"][vidx].copy()

            bbox = data_bbox[vidx, view_idx]  # original bbox
            is_egocam = "/0/" in imgname

            image_size = self.image_sizes[sid][view_idx]
            image_size = {"width": image_size[0], "height": image_size[1]}

            # SPEEDUP PROCESS
            bbox = dataset_utils.transform_bbox_for_speedup(
                speedup,
                is_egocam,
                bbox,
                args.ego_image_scale,
            )
            img_status = True
            if load_rgb:
                cv_img, img_status = read_img(imgname, (2800, 2000, 3))
            else:
                norm_img = None

            center = [bbox[0], bbox[1]]
            scale = bbox[2]
            self.aug_data = False

            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                args.flip_prob,
                args.noise_factor,
                args.rot_factor,
                args.scale_factor,
            )

            use_gt_k = args.use_gt_k
            if is_egocam:
                # no scaling for egocam to make intrinsics consistent
                use_gt_k = True
                augm_dict["sc"] = 1.0

            # data augmentation: image
            if load_rgb:
                img = data_utils.rgb_processing(
                    self.aug_data,
                    cv_img,
                    center,
                    scale,
                    augm_dict,
                    img_res=args.img_res,
                )
                img = torch.from_numpy(img).float()
                norm_img = self.normalize_img(img)

            # exporting starts
            inputs = {}
            targets = {}
            meta_info = {}
            inputs["img"] = norm_img
            sequence = sid+'_'+seq_name+'_'+str(view_idx)
            meta_data = self.pseudo_data[sequence]['meta_info']
            pred_data = self.pseudo_data[sequence]['preds']
            info_index = meta_data['meta_info.imgname'].index(imgname)

            meta_info["imgname"] = meta_data['meta_info.imgname'][info_index]
            meta_info["query_names"] = meta_data['meta_info.query_names'][info_index]
            meta_info["window_size"] = meta_data['meta_info.window_size'][info_index].clone().unsqueeze(0)#[]
            meta_info["intrinsics"] = meta_data['meta_info.intrinsics'][info_index].clone().to(dtype=torch.float32)#3,3
            meta_info["dist"] = meta_data['meta_info.dist'][info_index].clone().to(dtype=torch.float32)#8
            meta_info["center"] = np.array(meta_data['meta_info.center'][info_index].clone(), dtype=np.float32)#2
            meta_info["is_flipped"] = int(meta_data['meta_info.is_flipped'][info_index].clone())#[]
            meta_info["rot_angle"] = np.float32(meta_data['meta_info.rot_angle'][info_index].clone())#[]

            obj_idx = self.obj_names.index(obj_name)
            meta_info["kp3d.cano"] = self.kp3d_cano[obj_idx] / 1000#16,3
            meta_info['object_v_sub_idx'] = self.object_v_sub_idx[obj_idx]#600

            pose_r = pred_data['pred.mano.pose.r'][info_index].clone()
            pose_l = pred_data['pred.mano.pose.l'][info_index].clone()

            targets["mano.pose.r"] = matrix_to_axis_angle(pose_r.to(dtype=torch.float32)).reshape(-1)#16,3,3
            targets["mano.pose.l"] = matrix_to_axis_angle(pose_r.to(dtype=torch.float32)).reshape(-1)#16,3,3
            targets["mano.beta.r"] = pred_data['pred.mano.beta.r'][info_index].clone().to(dtype=torch.float32)#10
            targets["mano.beta.l"] = pred_data['pred.mano.beta.l'][info_index].clone().to(dtype=torch.float32)#10
            targets["mano.j2d.norm.r"] = pred_data['pred.mano.j2d.norm.r'][info_index].clone().to(dtype=torch.float32)#21,2
            targets["mano.j2d.norm.l"] = pred_data['pred.mano.j2d.norm.l'][info_index].clone().to(dtype=torch.float32)#21,2
            targets["object.kp2d.norm.b"] = pred_data['pred.object.kp2d.norm.b'][info_index].clone().to(dtype=torch.float32)#16,2
            targets["object.kp2d.norm.t"] = pred_data['pred.object.kp2d.norm.t'][info_index].clone().to(dtype=torch.float32)#16,2
            targets["object.bbox2d.norm.b"] = pred_data['pred.object.bbox2d.norm.b'][info_index].clone().to(dtype=torch.float32)#8,2
            targets["object.bbox2d.norm.t"] = pred_data['pred.object.bbox2d.norm.t'][info_index].clone().to(dtype=torch.float32)#8,2
            targets["object.bbox2d.norm"] = torch.concat((targets["object.bbox2d.norm.t"], targets["object.bbox2d.norm.b"]), dim=0).to(dtype=torch.float32)

            targets["object.kp3d.full.b"] = torch.rand((16,3), dtype = torch.float32)
            targets["object.kp3d.full.t"] = torch.rand((16,3), dtype = torch.float32)
            targets["object.bbox3d.full.b"] = torch.rand((8,3), dtype = torch.float32)
            targets["object.bbox3d.full.t"] = torch.rand((8,3), dtype = torch.float32)
            targets["mano.j3d.full.r"] = torch.rand((21,3), dtype = torch.float32)
            targets["mano.j3d.full.l"] = torch.rand((21,3), dtype = torch.float32)

            targets["object.rot"] = pred_data['pred.object.rot'][info_index].clone().view(1, 3).to(dtype=torch.float32)#1,3
            targets["object.radian"] = pred_data['pred.object.radian'][info_index].clone().to(dtype=torch.float32)#[]
            targets["object.kp2d.norm"] = pred_data['pred.object.kp2d.norm'][info_index].clone().to(dtype=torch.float32)#32,2

            targets["is_valid"] = float(1)#1.0
            targets["left_valid"] = float(1) * float(1)
            targets["right_valid"] = float(1) * float(1)
            targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
            targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

            targets["mano.cam_t.r"] = pred_data['pred.mano.cam_t.r'][info_index].clone().to(dtype=torch.float32)#3
            targets["mano.cam_t.l"] = pred_data['pred.mano.cam_t.l'][info_index].clone().to(dtype=torch.float32)#3
            targets["object.cam_t"] = pred_data['pred.object.cam_t'][info_index].clone().to(dtype=torch.float32)#3

            targets["mano.cam_t.wp.r"] = pred_data['pred.mano.cam_t.wp.r'][info_index].clone().to(dtype=torch.float32)#3
            targets["mano.cam_t.wp.l"] = pred_data['pred.mano.cam_t.wp.l'][info_index].clone().to(dtype=torch.float32)#3
            targets["object.cam_t.wp"] = pred_data['pred.object.cam_t.wp'][info_index].clone().to(dtype=torch.float32)#3

            targets["mano.v3d.cam.r"] = pred_data['pred.mano.v3d.cam.r'][info_index].clone().to(dtype=torch.float32)#778,3
            targets["mano.v3d.cam.l"] = pred_data['pred.mano.v3d.cam.l'][info_index].clone().to(dtype=torch.float32)#778,3
            targets["mano.j3d.cam.r"] = pred_data['pred.mano.j3d.cam.r'][info_index].clone().to(dtype=torch.float32)#21,3
            targets["mano.j3d.cam.l"] = pred_data['pred.mano.j3d.cam.l'][info_index].clone().to(dtype=torch.float32)#21,3

            targets["object.kp3d.cam"] = pred_data['pred.object.kp3d.cam'][info_index].clone().to(dtype=torch.float32)#32,3
            targets["object.bbox3d.cam"] = pred_data['pred.object.bbox3d.cam'][info_index].clone().to(dtype=torch.float32)#16,3
        
        else:
            obj_name = seq_name.split("_")[0]
            view_idx = int(view_idx)#index of sequence

            seq_data = self.data[f"{sid}/{seq_name}"]

            data_cam = seq_data["cam_coord"]
            data_2d = seq_data["2d"]
            data_bbox = seq_data["bbox"]
            data_params = seq_data["params"]

            vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]#index of image
            vidx, is_valid, right_valid, left_valid = get_valid(
                data_2d, data_cam, vidx, view_idx, imgname
            )

            if view_idx == 0:#egocentric or allocentric
                intrx = data_params["K_ego"][vidx].copy()
            else:
                intrx = np.array(self.intris_mat[sid][view_idx - 1])

            # hands
            joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())#pad one dimension after the 2D coordinate of joints 
            joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

            joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
            joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

            pose_r = data_params["pose_r"][vidx].copy()
            betas_r = data_params["shape_r"][vidx].copy()
            pose_l = data_params["pose_l"][vidx].copy()
            betas_l = data_params["shape_l"][vidx].copy()

            # distortion parameters for egocam rendering
            dist = data_params["dist"][vidx].copy()
            # NOTE:
            # kp2d, kp3d are in undistored space
            # thus, results for evaluation is in the undistorted space (non-curved)
            # dist parameters can be used for rendering in visualization

            # objects
            bbox2d = pad_jts2d(data_2d["bbox3d"][vidx, view_idx].copy())
            bbox3d = data_cam["bbox3d"][vidx, view_idx].copy()
            bbox2d_t = bbox2d[:8]
            bbox2d_b = bbox2d[8:]
            bbox3d_t = bbox3d[:8]
            bbox3d_b = bbox3d[8:]

            kp2d = pad_jts2d(data_2d["kp3d"][vidx, view_idx].copy())
            kp3d = data_cam["kp3d"][vidx, view_idx].copy()
            kp2d_t = kp2d[:16]
            kp2d_b = kp2d[16:]
            kp3d_t = kp3d[:16]
            kp3d_b = kp3d[16:]

            obj_radian = data_params["obj_arti"][vidx].copy()

            image_size = self.image_sizes[sid][view_idx]
            image_size = {"width": image_size[0], "height": image_size[1]}

            bbox = data_bbox[vidx, view_idx]  # original bbox
            is_egocam = "/0/" in imgname

            # end_loading = time.time()
            # print('one loading time is {}'.format(end_loading-start_loading))

            # LOADING END

            # SPEEDUP PROCESS
            (
                joints2d_r,
                joints2d_l,
                kp2d_b,
                kp2d_t,
                bbox2d_b,
                bbox2d_t,
                bbox,
            ) = dataset_utils.transform_2d_for_speedup(
                speedup,
                is_egocam,
                joints2d_r,
                joints2d_l,
                kp2d_b,
                kp2d_t,
                bbox2d_b,
                bbox2d_t,
                bbox,
                args.ego_image_scale,
            )
            img_status = True
            if load_rgb:
                if speedup:
                    imgname = imgname.replace("/images/", "/cropped_images/")
                imgname = imgname.replace(
                    "./arctic_data/", "/ssd/dylu/data/arctic/arctic_data/data/"
                ).replace("/data/data/", "/data/")
                cv_img, img_status = read_img(imgname, (2800, 2000, 3))
            else:
                norm_img = None

            center = [bbox[0], bbox[1]]
            scale = bbox[2]

            # start_aug = time.time()

            # augment parameters
            augm_dict = data_utils.augm_params(
                self.aug_data,
                args.flip_prob,
                args.noise_factor,
                args.rot_factor,
                args.scale_factor,
            )

            use_gt_k = args.use_gt_k
            if is_egocam:
                # no scaling for egocam to make intrinsics consistent
                use_gt_k = True
                augm_dict["sc"] = 1.0

            #data augmentation: 2d coordinates
            joints2d_r = data_utils.j2d_processing(
                joints2d_r, center, scale, augm_dict, args.img_res
            )
            joints2d_l = data_utils.j2d_processing(
                joints2d_l, center, scale, augm_dict, args.img_res
            )
            kp2d_b = data_utils.j2d_processing(
                kp2d_b, center, scale, augm_dict, args.img_res
            )
            kp2d_t = data_utils.j2d_processing(
                kp2d_t, center, scale, augm_dict, args.img_res
            )
            bbox2d_b = data_utils.j2d_processing(
                bbox2d_b, center, scale, augm_dict, args.img_res
            )
            bbox2d_t = data_utils.j2d_processing(
                bbox2d_t, center, scale, augm_dict, args.img_res
            )
            bbox2d = np.concatenate((bbox2d_t, bbox2d_b), axis=0)
            kp2d = np.concatenate((kp2d_t, kp2d_b), axis=0)

            # data augmentation: image
            if load_rgb:
                img = data_utils.rgb_processing(
                    self.aug_data,
                    cv_img,
                    center,
                    scale,
                    augm_dict,
                    img_res=args.img_res,
                )
                img = torch.from_numpy(img).float()
                norm_img = self.normalize_img(img)
            # end_aug = time.time()
            # print('one augmentation is {}'.format(end_aug-start_aug))

            # start_export = time.time()
            # exporting starts
            inputs = {}
            targets = {}
            meta_info = {}
            inputs["img"] = norm_img
            meta_info["imgname"] = imgname
            rot_r = data_cam["rot_r_cam"][vidx, view_idx]
            rot_l = data_cam["rot_l_cam"][vidx, view_idx]

            pose_r = np.concatenate((rot_r, pose_r), axis=0)
            pose_l = np.concatenate((rot_l, pose_l), axis=0)

            # hands
            targets["mano.pose.r"] = torch.from_numpy(
                data_utils.pose_processing(pose_r, augm_dict)
            ).float()#48
            targets["mano.pose.l"] = torch.from_numpy(
                data_utils.pose_processing(pose_l, augm_dict)
            ).float()#48
            targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
            targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
            targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
            targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

            # object
            targets["object.kp3d.full.b"] = torch.from_numpy(kp3d_b[:, :3]).float()#16,3
            targets["object.kp2d.norm.b"] = torch.from_numpy(kp2d_b[:, :2]).float()#16,2
            targets["object.kp3d.full.t"] = torch.from_numpy(kp3d_t[:, :3]).float()#16,3
            targets["object.kp2d.norm.t"] = torch.from_numpy(kp2d_t[:, :2]).float()#16,2

            targets["object.bbox3d.full.b"] = torch.from_numpy(bbox3d_b[:, :3]).float()#8,3
            targets["object.bbox2d.norm.b"] = torch.from_numpy(bbox2d_b[:, :2]).float()#8,2
            targets["object.bbox3d.full.t"] = torch.from_numpy(bbox3d_t[:, :3]).float()#8,3
            targets["object.bbox2d.norm.t"] = torch.from_numpy(bbox2d_t[:, :2]).float()#8,2
            targets["object.radian"] = torch.FloatTensor(np.array(obj_radian))

            targets["object.kp2d.norm"] = torch.from_numpy(kp2d[:, :2]).float()
            targets["object.bbox2d.norm"] = torch.from_numpy(bbox2d[:, :2]).float()

            # compute RT from cano space to augmented space
            # this transform match j3d processing
            obj_idx = self.obj_names.index(obj_name)
            meta_info["kp3d.cano"] = self.kp3d_cano[obj_idx] / 1000  # meter
            meta_info['object_v_sub_idx'] = self.object_v_sub_idx[obj_idx]
            kp3d_cano = meta_info["kp3d.cano"].numpy()
            kp3d_target = targets["object.kp3d.full.b"][:, :3].numpy()

            # rotate canonical kp3d to match original image
            R, _ = tf.solve_rigid_tf_np(kp3d_cano, kp3d_target)
            obj_rot = (
                rot.batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
            )

            # multiply rotation from data augmentation, canonical->image->augmented
            obj_rot_aug = rot.rot_aa(obj_rot, augm_dict["rot"])
            targets["object.rot"] = torch.FloatTensor(obj_rot_aug).view(1, 3)#1,3

            targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
            targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])
            targets["object.kp3d.full.b"] = torch.FloatTensor(kp3d_b[:, :3])

            meta_info["query_names"] = obj_name
            meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

            # scale and center in the original image space
            scale_original = max([image_size["width"], image_size["height"]]) / 200.0
            center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
            intrx = data_utils.get_aug_intrix(
                intrx,
                args.focal_length,
                args.img_res,
                use_gt_k,
                center_original[0],
                center_original[1],
                augm_dict["sc"] * scale_original,
            )

            if is_egocam and self.egocam_k is None:
                self.egocam_k = intrx
            elif is_egocam and self.egocam_k is not None:
                intrx = self.egocam_k

            meta_info["intrinsics"] = torch.FloatTensor(intrx)
            if not is_egocam:
                dist = dist * float("nan")
            meta_info["dist"] = torch.FloatTensor(dist)#distortion
            meta_info["center"] = np.array(center, dtype=np.float32)
            meta_info["is_flipped"] = augm_dict["flip"]
            meta_info["rot_angle"] = np.float32(augm_dict["rot"])
            # meta_info["sample_index"] = index

            # root and at least 3 joints inside image
            targets["is_valid"] = float(is_valid)
            targets["left_valid"] = float(left_valid) * float(is_valid)
            targets["right_valid"] = float(right_valid) * float(is_valid)
            targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
            targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

            targets["mano.cam_t.r"] = torch.rand((3), dtype = torch.float32)
            targets["mano.cam_t.l"] = torch.rand((3), dtype = torch.float32)
            targets["object.cam_t"] = torch.rand((3), dtype = torch.float32)

            targets["mano.cam_t.wp.r"] = torch.rand((3), dtype = torch.float32)
            targets["mano.cam_t.wp.l"] = torch.rand((3), dtype = torch.float32)
            targets["object.cam_t.wp"] = torch.rand((3), dtype = torch.float32)

            targets["mano.v3d.cam.r"] = torch.rand((778,3), dtype = torch.float32)
            targets["mano.v3d.cam.l"] = torch.rand((778,3), dtype = torch.float32)
            targets["mano.j3d.cam.r"] = torch.rand((21,3), dtype = torch.float32)
            targets["mano.j3d.cam.l"] = torch.rand((21,3), dtype = torch.float32)

            targets["object.kp3d.cam"] = torch.rand((32,3), dtype = torch.float32)
            targets["object.bbox3d.cam"] = torch.rand((16,3), dtype = torch.float32)

            # torch.cuda.synchronize()
            # end_export = time.time()
            # print('one export time is {}'.format(end_export-start_export))
            # end_getitem = time.time()
            # print('one get item time is {}'.format(end_getitem-start_getitem))

        return inputs, targets, meta_info

    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames
        
    def _load_image(self):

        img_list = []
        for img_name in self.imgnames:

            img_name = img_name.replace("/images/", "/cropped_images/")
            img_name = img_name.replace(
                "./arctic_data/", "/ssd/dylu/data/arctic/arctic_data/data/"
            ).replace("/data/data/", "/data/")
            # start_image = time.time()
            cv_img, img_status = read_img(img_name, (2800, 2000, 3))
            img_list.append(cv_img)
        
        self.imgs = torch.stack(img_list,dim=0)
        print('---')

    def _load_data(self, args, split, seq):
        logger.info(f'load data!')
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        # during inference, turn off
        if seq is not None:
            self.aug_data = False
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        data_p = op.join(
            f"/ssd/dylu/data/arctic/arctic_data/data/splits/{args.setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        data_test = op.join(
            f"/ssd/dylu/data/arctic/arctic_data/data/splits/p1_test.npy"
        )
        logger.info(f"Loading {data_test}")
        data_test = np.load(data_test, allow_pickle=True).item()

        #data only have these two dicts
        self.data = data["data_dict"]
        self.data.update(data_test["data_dict"])
        self.imgnames = data["imgnames"]

        with open("/ssd/dylu/data/arctic/arctic_data/data/meta/misc.json", "r") as f:
            misc = json.load(f)

        # unpack
        subjects = list(misc.keys())#different views e.g. s01,s02...
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam#camera extrinsics
        self.intris_mat = intris_mat#camera intrinsics
        self.image_sizes = image_sizes#image resolution
        self.ioi_offset = ioi_offset#a scalar value

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.object_v_sub_idx = object_tensors.obj_tensors["v_sub_idx"]
        self.egocam_k = None

    def __init__(self, args, split, seq=None):
        logger.info(f'initialize arctic dataset!')
        self._load_data(args, split, seq)
        self._process_imgnames(seq, split)
        
        if split.endswith("train"):
            self.pseudo_data = torch.load('/ssd/dylu/data/arctic/arctic_data/data/test_data.pt')
            #read imgname
            img_name = []
            for seq in self.pseudo_data.keys():
                img_name = img_name+self.pseudo_data[seq]['meta_info']['meta_info.imgname']  
            self.imgnames = self.imgnames+img_name
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        return len(self.imgnames)

    def getitem_eval(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        # SPEEDUP PROCESS
        bbox = dataset_utils.transform_bbox_for_speedup(
            speedup,
            is_egocam,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace(
                "./arctic_data/data/", "/ssd/dylu/data/arctic/arctic_data/data/"
            )
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]
        self.aug_data = False

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
            img = torch.from_numpy(img).float()
            norm_img = self.normalize_img(img)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        meta_info["imgname"] = imgname

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))
        obj_idx = self.obj_names.index(obj_name)
        meta_info['object_v_sub_idx'] = self.object_v_sub_idx[obj_idx]
        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")

        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        return inputs, targets, meta_info
