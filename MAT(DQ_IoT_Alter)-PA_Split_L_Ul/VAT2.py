import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE hyper-parameters we use in VAT
# n_power: a number of power iteration for approximation of r_vadv
# XI: a small float for the approx. of the finite difference method
# epsilon: the value for how much deviate from original data point X


class VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model):
        super(VAT, self).__init__()
        self.model = model
        self.n_power = 1
        self.XI = 0.01
        self.epsilon = 1.0

    def forward(self, X):
        vat_loss = virtual_adversarial_loss(X, self.model, self.n_power, self.XI, self.epsilon)
        return vat_loss  # already averaged


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def generate_virtual_adversarial_perturbation(x, model, n_power, XI, epsilon):
    d = torch.randn_like(x)

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        logit = model(x)
        logit_m = model(x + d)
        dist = kl_divergence_with_logit(logit[1], logit_m[1])
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()

    return epsilon * get_normalized_vector(d)   #扰动强度*扰动方向


def virtual_adversarial_loss(x, model, n_power, XI, epsilon):
    '''
    :param x: 用于生成对抗样本的有效样本
    :param model: 模型
    :param n_power: 对抗参数
    :param XI: 对抗参数
    :param epsilon: 对抗参数
    :return:
    '''
    r_vadv = generate_virtual_adversarial_perturbation(x, model, n_power, XI, epsilon) # 求扰动大小
    logit_p = model(x)
    logit_m = model(x + r_vadv)
    loss = kl_divergence_with_logit(logit_p[1], logit_m[1])
    return loss