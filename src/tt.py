from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from focal_loss import FocalLoss
from torch.nn.modules.loss import _WeightedLoss
import numpy as np


class FocalLoss(_WeightedLoss):
    """
    Reimplementation of the Focal Loss described in:
        - "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
        - "AnatomyNet: Deep learning for fast and fully automated whole‐volume segmentation of head and neck anatomy",
          Zhu et al., Medical Physics 2018
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        """
        Args:
            gamma: value of the exponent gamma in the definition of the Focal loss.
            weight (tensor): weights to apply to the voxels of each class. If None no weights are applied.
                This corresponds to the weights `\alpha`.
        reduction: {``"none"``, ``"mean"``, ``"sum"``}
            Specifies the reduction to apply to the output. Defaults to ``"mean"``.
            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.
        Example:
            .. code-block:: python
                import torch
                from monai.losses import FocalLoss
                pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
                grnd = torch.tensor([[0], [1], [0]], dtype=torch.int64)
                fl = FocalLoss()
                fl(pred, grnd)
        """
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Args:
            input: (tensor): the shape should be BCH[WD].
                where C is the number of classes.
            target: (tensor): the shape should be B1H[WD] or BCH[WD].
                If the target's shape is B1H[WD], the target that this loss expects should be a class index
                in the range [0, C-1] where C is the number of classes.
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))

        i = input
        t = target

        if i.ndim != t.ndim:
            raise ValueError(f"input and target must have the same number of dimensions, got {i.ndim} and {t.ndim}")

        if target.shape[1] != 1 and target.shape[1] != i.shape[1]:
            raise ValueError(
                "target must have one channel or have the same shape as the input. "
                "If it has one channel, it should be a class index in the range [0, C-1] "
                f"where C is the number of classes inferred from 'input': C={i.shape[1]}. "
            )
        # Change the shape of input and target to
        # num_batch x num_class x num_voxels.
        if input.dim() > 2:
            i = i.view(i.size(0), i.size(1), -1)  # N,C,H,W => N,C,H*W
            t = t.view(t.size(0), t.size(1), -1)  # N,1,H,W => N,1,H*W or N,C,H*W
        else:  # Compatibility with classification.
            i = i.unsqueeze(2)  # N,C => N,C,1
            t = t.unsqueeze(2)  # N,1 => N,1,1 or N,C,1

        # Compute the log proba (more stable numerically than softmax).
        logpt = F.log_softmax(i, dim=1)  # N,C,H*W
        # Keep only log proba values of the ground truth class for each voxel.
        if target.shape[1] == 1:
            logpt = logpt.gather(1, t.long())  # N,C,H*W => N,1,H*W
            logpt = torch.squeeze(logpt, dim=1)  # N,1,H*W => N,H*W

        # Get the proba
        pt = torch.exp(logpt)  # N,H*W or N,C,H*W

        if self.weight is not None:
            self.weight = self.weight.to(i)
            # Convert the weight to a map in which each voxel
            # has the weight associated with the ground-truth label
            # associated with this voxel in target.
            at = self.weight[None, :, None]  # C => 1,C,1
            at = at.expand((t.size(0), -1, t.size(2)))  # 1,C,1 => N,C,H*W
            if target.shape[1] == 1:
                at = at.gather(1, t.long())  # selection of the weights  => N,1,H*W
                at = torch.squeeze(at, dim=1)  # N,1,H*W => N,H*W
            # Multiply the log proba by their weights.
            logpt = logpt * at

        # Compute the loss mini-batch.
        weight = torch.pow(-pt + 1.0, self.gamma)
        if target.shape[1] == 1:
            loss = torch.mean(-weight * logpt, dim=1)  # N
        else:
            loss = torch.mean(-weight * t * logpt, dim=-1)  # N,C

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError(f"reduction={self.reduction} is invalid.")


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss



def ignore_padding(output, target, padding_idx=2):
    target = target.view(-1)
    output = output.view(-1, 2)
    num_padding = 0
    n = len(target)
    for i in range(n-1, -1, -1):
        if target[i] == padding_idx:
            num_padding += 1
    target = target[:n-num_padding]
    output = output[:n-num_padding]
    return output, target

# def main():
#     Floss = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
#     FM = MultiFocalLoss(3, alpha=[0.2, 0.5, 0.3])
#     CEloss = nn.CrossEntropyLoss()
#
#     output = torch.tensor([[[2.3043, -2.1607],
#                             [2.2500, -2.1672],
#                             [2.2517, -2.1451],
#                             [2.2304, -2.1985],
#                             [2.2209, -2.1839],
#                             [2.2165, -2.1889],
#                             [2.1822, -2.1244],
#                             [2.2945, -2.1647],
#                             [2.2062, -2.1689],
#                             [2.2012, -2.1764],
#                             [2.2538, -2.1148],
#                             [2.2086, -2.1339]],
#                            [[1.2197, -1.5007],
#                             [-0.8276,  0.9224],
#                             [1.5889, -1.6813],
#                             [1.7248, -1.9020],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390],
#                             [0.0070, -0.0390]]])
#     target = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]])
#     output, target = ignore_padding(output, target, padding_idx=2)
#     print(output.dim())
#     print(Floss(output, target.view(-1).unsqueeze(1)))
#     output = torch.tensor([[-1.1420,  2.4045, -1.5674],
#         [-1.0236,  2.2133, -1.5358],
#         [-1.2186,  2.3259, -1.5159],
#         [-1.0814,  2.4525, -1.5142]])
#     target = torch.tensor([2, 1, 1, 1])
#     print(FM(output.view(-1, 3), target.view(-1).unsqueeze(1)))
#     print(100 * '*')
#     # print(CEloss(output.view(-1, 2), target.view(-1)))
#     print(100 * '*')
#     # print(Floss(output.view(-1, 2), target.view(-1).unsqueeze(1)))


import torch
import argparse
from transformers import AutoTokenizer
import random

from dataset.loader import SciFactJointDataset
from train_model import train, train_base
from get_prediction import get_predictions
from utils import predictions2jsonl
from evaluation.evaluation_model import merge_rationale_label, evaluate_rationale_selection, evaluate_label_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/claims_train_retrieved.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--claim_test_path', type=str,
                        default='../data/claims_test.jsonl')
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl')
    parser.add_argument('--abstract_retrieval', type=str,
                        default='prediction/abstract_retrieval.jsonl')
    parser.add_argument('--rationale_selection', type=str,
                        default='prediction/rationale_selection.jsonl')
    parser.add_argument('--save', type=str, default='model/',
                        help='Folder to save the weights')
    parser.add_argument('--output_label', type=str, default='prediction/label_predictions.jsonl')
    parser.add_argument('--merge_results', type=str, default='prediction/merged_predictions.jsonl')
    parser.add_argument('--output', type=str, default='prediction/result_evaluation.json',
                        help='The predictions.')
    parser.add_argument('--rationale_selection_tfidf', type=str, default='prediction/rationale_selection_tfidf.jsonl')

    # model parameters.
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--rationale_model', type=str, default='')
    parser.add_argument('--label_model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5, required=False)
    parser.add_argument('--only_rationale', action='store_true')
    parser.add_argument('--batch_size_gpu', type=int, default=8,
                        help='The batch size to send through GPU')
    parser.add_argument('--batch-size-accumulated', type=int, default=256,
                        help='The batch size for each gradient update')
    parser.add_argument('--lr-base', type=float, default=1e-5)
    parser.add_argument('--lr-linear', type=float, default=5e-6)
    parser.add_argument('--mode', type=str, default='claim_and_rationale',
                        choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
    parser.add_argument('--filter', type=str, default='structured',
                        choices=['structured', 'unstructured'])

    parser.add_argument('--embedding', type=str, default='roberta')

    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension")
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_label", type=int, default=2, help="numbers of the label")
    parser.add_argument("--class_num_label", type=int, default=1,
                        help="max number of the label for one class")
    parser.add_argument("--embed_size", type=int, default=768, help="embedding size")
    parser.add_argument("--cnn_num_filters", type=int, default=128,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")
    parser.add_argument('--vocab_size', type=int, default=31116)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop rate")
    parser.add_argument('--k', type=int, default=10, help="tfidf")

    return parser.parse_args()

def printf(args):
    pass


def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader dataset
    split = True
    if split:
        # split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/train_data.jsonl'
        claim_dev_path = '../data/dev_data.jsonl'
        claim_test_path = '../data/claims_dev_retrieved_tfidf.jsonl'
        # print(claim_test_path)
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path
        claim_test_path = args.claim_test_path

    args.model = 'allenai/scibert_scivocab_cased'
    # args.model = 'model/SciBert_checkpoint'
    # args.model = 'roberta-large'
    args.epochs = 40
    args.lr_base = 1e-5
    args.lr_linear = 1e-3
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 30
    args.hidden_dim = 768  # 768
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')


if __name__ == "__main__":
    main()