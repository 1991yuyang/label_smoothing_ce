import torch as t
from torch import nn
from torch.nn import functional as F


class LabelSmoothCE(nn.Module):

    def __init__(self, epsilon, num_classes, weight=None, reduce="mean"):
        """

        :param epsilon: float, label smooth超参数
        :param num_classes: int, 类别数目
        :param weight: float tensor， 各类别的权重
        :param reduce:
        """
        super(LabelSmoothCE, self).__init__()
        if weight is not None:
            assert weight.size()[0] == num_classes, "number of weight of class should be num_classes"
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.eps = epsilon
        self.num_classes = num_classes
        self.weight = weight
        self.reduce = reduce

    def forward(self, model_output, target):
        """

        :param model_output: (N, C), float
        :param target: (N,), Long
        :return:
        """
        orig_target = target.cpu().detach().numpy()
        target = (F.one_hot(target, self.num_classes) * (1 - self.eps) + self.eps / self.num_classes) * 1.0
        log_softmax_result = self.log_softmax(model_output)
        if self.weight is not None:
            weight_new = self.weight[orig_target]
            loss = -t.sum(log_softmax_result * target, dim=1) * weight_new
        else:
            loss = -t.sum(log_softmax_result * target, dim=1, keepdim=True)
        if self.reduce == "mean":
            if self.weight is not None:
                loss = t.sum(loss) / t.sum(weight_new)
            else:
                loss = t.mean(loss)
        else:
            loss = t.sum(loss)
        return loss


if __name__ == "__main__":
    d = t.tensor([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])
    l = t.tensor([0, 1, 2])
    ce = nn.CrossEntropyLoss(reduction="mean")
    smooth_ce = LabelSmoothCE(0, num_classes=3)
    loss1 = ce(d, l)
    loss2 = smooth_ce(d, l)
    print(loss1)
    print(loss2)
