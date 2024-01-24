import torch.nn as nn
import torch
import numpy as np
from typing import List, cast
from torch import Tensor, einsum
from boundary_utils import simplex, probs2one_hot, one_hot
from boundary_utils import one_hot2hd_dist

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class BDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(),
                                      padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target, weight=None, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        # BD_loss = self._adaptive_size(inputs[:, 0], target[:, 0])
        BD_loss = self._adaptive_size(inputs, target)
        return BD_loss


# class SurfaceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert not one_hot(dist_maps)
#         probs = torch.sigmoid(probs)
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         dc = dist_maps[:, self.idc, ...].type(torch.float32)
#
#         multipled = einsum("bkwh,bkwh->bkwh", pc, dc)
#
#         loss = multipled.mean()
#
#         return loss
#
#
# BoundaryLoss = SurfaceLoss
#
#
# class HausdorffLoss():
#     """
#     Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
#     """
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")
#
#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert simplex(target)
#         assert probs.shape == target.shape
#
#         B, K, *xyz = probs.shape  # type: ignore
#
#         pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
#         tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
#         assert pc.shape == tc.shape == (B, len(self.idc), *xyz)
#
#         target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
#                                               for b in range(B)], axis=0)
#         assert target_dm_npy.shape == tc.shape == pc.shape
#         tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)
#
#         pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
#         pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
#                                             for b in range(B)], axis=0)
#         assert pred_dm_npy.shape == tc.shape == pc.shape
#         pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)
#
#         delta = (pc - tc)**2
#         dtm = tdm**2 + pdm**2
#
#         multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)
#
#         loss = multipled.mean()
#
#         return loss
#
#
#
