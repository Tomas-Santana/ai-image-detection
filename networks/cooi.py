import torch
import torch.nn as nn
from typing import Tuple, List


class COOI:  # Coordinates On Original Image
    def __init__(self):
        self.stride: int = 32
        self.cropped_size: int = 224
        self.score_filter_size_list: List[List[int]] = [[3, 3], [2, 2]]
        self.score_filter_num_list: List[int] = [3, 3]
        self.score_nms_size_list: List[List[int]] = [[3, 3], [3, 3]]
        self.score_nms_padding_list: List[List[int]] = [[1, 1], [1, 1]]
        self.score_corresponding_patch_size_list: List[List[int]] = [
            [224, 224],
            [112, 112],
        ]
        self.score_filter_type_size: int = len(self.score_filter_size_list)

    def get_coordinates(
        self, fm: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        with torch.no_grad():
            batch_size, _, fm_height, fm_width = fm.size()
            scale_min = torch.min(scale, dim=1, keepdim=True)[0].long()
            scale_base = (scale - scale_min).long() // 2
            input_loc_list = []
            fps_loc_list = []
            for type_no in range(self.score_filter_type_size):
                score_avg = nn.functional.avg_pool2d(
                    fm, self.score_filter_size_list[type_no], stride=1
                )
                score_sum = torch.sum(score_avg, dim=1, keepdim=True)
                _, _, score_height, score_width = score_sum.size()
                patch_height, patch_width = self.score_corresponding_patch_size_list[
                    type_no
                ]

                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat = score_sum.view(batch_size, -1)
                    value_max, loc_max_flat = torch.max(score_sum_flat, dim=1)
                    loc_max = torch.stack(
                        (loc_max_flat // score_width, loc_max_flat % score_width), dim=1
                    )
                    fps_loc_list.append(loc_max)
                    top_patch = nn.functional.max_pool2d(
                        score_sum,
                        self.score_nms_size_list[type_no],
                        stride=1,
                        padding=self.score_nms_padding_list[type_no],
                    )
                    value_max = value_max.view(-1, 1, 1, 1)
                    erase = (top_patch != value_max).float()
                    score_sum = score_sum * erase

                    # location in the original images
                    loc_rate_h = (2 * loc_max[:, 0] + fm_height - score_height + 1) / (
                        2 * fm_height
                    )
                    loc_rate_w = (2 * loc_max[:, 1] + fm_width - score_width + 1) / (
                        2 * fm_width
                    )
                    loc_rate = torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center = (scale_base + scale_min * loc_rate).long()
                    loc_top = loc_center[:, 0] - patch_height // 2
                    loc_bot = loc_center[:, 0] + patch_height // 2 + patch_height % 2
                    loc_lef = loc_center[:, 1] - patch_width // 2
                    loc_rig = loc_center[:, 1] + patch_width // 2 + patch_width % 2
                    loc_tl = torch.stack((loc_top, loc_lef), dim=1)
                    loc_br = torch.stack((loc_bot, loc_rig), dim=1)

                    # For boundary conditions
                    loc_below = loc_tl.detach().clone()  # too low
                    loc_below[loc_below > 0] = 0
                    loc_br -= loc_below
                    loc_tl -= loc_below
                    loc_over = loc_br - scale.long()  # too high
                    loc_over[loc_over < 0] = 0
                    loc_tl -= loc_over
                    loc_br -= loc_over
                    loc_tl[loc_tl < 0] = 0  # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor = torch.stack(tensors=input_loc_list, dim=1)  # (7,6,4)

            return input_loc_tensor, fps_loc_list
