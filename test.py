import torch
mano_r = '/data/dylu/project/JointTransformer/logs/baseline/submit/pose_p1_test/eval/s03_box_grab_01_1/meta_info/meta_info.imgname.pt'
mano_r =  torch.load(mano_r)
print(mano_r)