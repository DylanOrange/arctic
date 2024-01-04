from src.callbacks.loss.loss_interhand import compute_loss
from src.callbacks.process.process_arctic import process_data
import src.callbacks.vis.visualize_arctic as visualize_arctic
import src.callbacks.vis.visualize_field as visualize_field
from src.models.interhand.model import InterHand
from src.models.generic.wrapper import GenericWrapper


class InterHandWrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = InterHand(
            backbone="ViT-L",
            args=args
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            "cdev",
            "mrrpe",
            "mpjpe.ra",
            "aae",
            "success_rate",
            "avg_err_kp_field",
            "avg_err_field_computed",
        ]

        self.vis_fns = [visualize_arctic.visualize_all, visualize_field.visualize_all]

        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)
