import torch.nn as nn

from common.xdict import xdict

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")
ce_loss = nn.CrossEntropyLoss(reduction="none")


def dist_loss(loss_dict, pred, gt, meta_info, weight=100.0):
    is_valid = gt["is_valid"]
    mask_o = meta_info["mask"]

    # interfield
    loss_ro = mse_loss(pred[f"dist.ro"], gt["dist.ro"])
    loss_lo = mse_loss(pred[f"dist.lo"], gt["dist.lo"])

    pad_olen = min(pred[f"dist.or"].shape[1], gt["dist.or"].shape[1])

    loss_or = mse_loss(pred[f"dist.or"][:, :pad_olen], gt["dist.or"][:, :pad_olen])
    loss_ol = mse_loss(pred[f"dist.ol"][:, :pad_olen], gt["dist.ol"][:, :pad_olen])

    # too many 10cm. Skip them in the loss to prevent overfitting
    bnd = 0.1  # 10cm
    bnd_idx_ro = gt["dist.ro"] == bnd
    bnd_idx_lo = gt["dist.lo"] == bnd
    bnd_idx_or = gt["dist.or"][:, :pad_olen] == bnd
    bnd_idx_ol = gt["dist.ol"][:, :pad_olen] == bnd

    #here we have only keypoint, no need for mask
    loss_or = loss_or * mask_o * is_valid[:, None]
    loss_ol = loss_ol * mask_o * is_valid[:, None]

    loss_ro = loss_ro * is_valid[:, None]
    loss_lo = loss_lo * is_valid[:, None]

    loss_or[bnd_idx_or] *= 0.1
    loss_ol[bnd_idx_ol] *= 0.1
    loss_ro[bnd_idx_ro] *= 0.1
    loss_lo[bnd_idx_lo] *= 0.1

    # weight = 100.0
    loss_dict[f"loss/dist/ro"] = (loss_ro.mean(), weight)
    loss_dict[f"loss/dist/lo"] = (loss_lo.mean(), weight)
    loss_dict[f"loss/dist/or"] = (loss_or.mean(), weight)
    loss_dict[f"loss/dist/ol"] = (loss_ol.mean(), weight)
    return loss_dict

def dist_loss_kp(loss_dict, pred, gt, weight=1.0):
    is_valid = gt["is_valid"]
    # mask_o = meta_info["mask"]

    # interfield
    loss_ro = mse_loss(pred[f"dist.ro.kp"], gt["dist.ro.kp"])
    loss_lo = mse_loss(pred[f"dist.lo.kp"], gt["dist.lo.kp"])

    pad_olen = min(pred[f"dist.or.kp"].shape[1], gt["dist.or.kp"].shape[1])

    loss_or = mse_loss(pred[f"dist.or.kp"][:, :pad_olen], gt["dist.or.kp"][:, :pad_olen])
    loss_ol = mse_loss(pred[f"dist.ol.kp"][:, :pad_olen], gt["dist.ol.kp"][:, :pad_olen])

    # too many 10cm. Skip them in the loss to prevent overfitting
    bnd = 0.1  # 10cm
    bnd_idx_ro = gt["dist.ro.kp"] == bnd
    bnd_idx_lo = gt["dist.lo.kp"] == bnd
    bnd_idx_or = gt["dist.or.kp"][:, :pad_olen] == bnd
    bnd_idx_ol = gt["dist.ol.kp"][:, :pad_olen] == bnd

    #here we have only keypoint, no need for mask
    # loss_or = loss_or * mask_o * is_valid[:, None]
    # loss_ol = loss_ol * mask_o * is_valid[:, None]

    loss_or = loss_or  * is_valid[:, None]
    loss_ol = loss_ol  * is_valid[:, None]

    loss_ro = loss_ro * is_valid[:, None]
    loss_lo = loss_lo * is_valid[:, None]

    loss_or[bnd_idx_or] *= 0.1
    loss_ol[bnd_idx_ol] *= 0.1
    loss_ro[bnd_idx_ro] *= 0.1
    loss_lo[bnd_idx_lo] *= 0.1

    # weight = 100.0
    loss_dict[f"loss/dist/ro"] = (loss_ro.mean(), weight)
    loss_dict[f"loss/dist/lo"] = (loss_lo.mean(), weight)
    loss_dict[f"loss/dist/or"] = (loss_or.mean(), weight)
    loss_dict[f"loss/dist/ol"] = (loss_ol.mean(), weight)
    return loss_dict

def computed_dist_loss(loss_dict, pred, gt, weight=100.0):
    is_valid = gt["is_valid"]
    # mask_o = meta_info["mask"]

    # interfield
    loss_ro = mse_loss(pred[f"dist.ro.kp.computed"], gt["dist.ro.kp"])
    loss_lo = mse_loss(pred[f"dist.lo.kp.computed"], gt["dist.lo.kp"])

    pad_olen = min(pred[f"dist.or.kp"].shape[1], gt["dist.or.kp"].shape[1])

    loss_or = mse_loss(pred[f"dist.or.kp.computed"][:, :pad_olen], gt["dist.or.kp"][:, :pad_olen])
    loss_ol = mse_loss(pred[f"dist.ol.kp.computed"][:, :pad_olen], gt["dist.ol.kp"][:, :pad_olen])

    # too many 10cm. Skip them in the loss to prevent overfitting
    bnd = 0.1  # 10cm
    bnd_idx_ro = gt["dist.ro.kp"] == bnd
    bnd_idx_lo = gt["dist.lo.kp"] == bnd
    bnd_idx_or = gt["dist.or.kp"][:, :pad_olen] == bnd
    bnd_idx_ol = gt["dist.ol.kp"][:, :pad_olen] == bnd

    #here we have only keypoint, no need for mask
    # loss_or = loss_or * mask_o * is_valid[:, None]
    # loss_ol = loss_ol * mask_o * is_valid[:, None]

    loss_or = loss_or  * is_valid[:, None]
    loss_ol = loss_ol  * is_valid[:, None]

    loss_ro = loss_ro * is_valid[:, None]
    loss_lo = loss_lo * is_valid[:, None]

    loss_or[bnd_idx_or] *= 0.1
    loss_ol[bnd_idx_ol] *= 0.1
    loss_ro[bnd_idx_ro] *= 0.1
    loss_lo[bnd_idx_lo] *= 0.1

    # weight = 100.0
    loss_dict[f"loss/dist/ro/computed"] = (loss_ro.mean(), weight)
    loss_dict[f"loss/dist/lo/computed"] = (loss_lo.mean(), weight)
    loss_dict[f"loss/dist/or/computed"] = (loss_or.mean(), weight)
    loss_dict[f"loss/dist/ol/computed"] = (loss_ol.mean(), weight)
    return loss_dict


def compute_loss(pred, gt, meta_info, args):
    loss_dict = xdict()
    loss_dict = dist_loss_kp(loss_dict, pred, gt, weight=100.0)
    return loss_dict
