import torch
import torch.nn as nn

from .builder import LOSS


def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


@LOSS.register_module
class L1LossDimSMPL(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPL, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'][:, :pred_uvd.shape[1]]
        target_uvd_weight = labels['target_weight_29'][:, :pred_uvd.shape[1]]
        loss_uvd = weighted_l1_loss(output.pred_uvd_jts, target_uvd, target_uvd_weight, self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        return loss
