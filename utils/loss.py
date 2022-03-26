from torch import nn as nn, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.ops import sigmoid_focal_loss


class SDLLoss(nn.Module):
    def __init__(self, weight=0.3, reduction='mean', wordvec_array=None):
        super(SDLLoss, self).__init__()
        self.eps = 1e-7
        assert reduction in ['mean', 'sum']
        self.reduction_type = reduction
        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.wordvec_array = wordvec_array
        self.embed_len = 300
        self.weight = weight

    def forward(self, x, y):
        # Update Aux Loss weight
        # if not provided the aux loss will not be used (regularization of matrix var)
        use_var_aux_loss = False
        if self.weight is not None:
            use_var_aux_loss = True
            weight = self.weight

        # Compute dot product for the output matrix A with every wordvec in the dictionary
        # init loss variable
        batch_size = y.size()[0]
        l = torch.zeros(batch_size).cuda()

        # X is in a vector of size kx300, we need to unflat it into a matrix A
        un_flat = x.view(x.shape[0], self.embed_len, -1)

        k = un_flat.shape[2]  # k = number of rows in the matrix

        # Compute dot product between A and all available word vectors (number of tags x 300)
        dot_prod_all = [torch.sum((un_flat[:, :, i].unsqueeze(2) * self.wordvec_array), dim=1).unsqueeze(2) for i in
                        range(k)]

        # Apply max on A dot wordvecs
        dot_prod_all = torch.max(torch.cat(dot_prod_all, dim=2), dim=-1)
        dot_prod_all = dot_prod_all.values

        # For loop over all batch
        for i in range(0, batch_size):
            # Separate Positive and Negative labels
            # y==1 means positive labels
            dot_prod_pos = dot_prod_all[i, y[i] == 1]
            # unknown are treated as negatives (-1,0)
            dot_prod_neg = dot_prod_all[i, (1 - y[i]).bool()]
            # dot_prod_neg = dot_prod_all[i, y[i] == 0]  # unknown are not used as negatives (0)

            # Compute v = max(An) - max(Ap)
            # v.shape = [num_pos, num_negatives]
            if len(dot_prod_neg) == 0:  # if no negative labels
                v = -dot_prod_pos.unsqueeze(1)
            else:
                v = dot_prod_neg.unsqueeze(0) - dot_prod_pos.unsqueeze(1)

            # Final loss equation (1/num_classes) * sum(log(1+exp(max(An_i) - max(Ap_i))))
            num_pos = dot_prod_pos.shape[0]
            # num_neg = dot_prod_neg.shape[0]
            total_var = calc_diversity(self.wordvec_array, y[i])

            l[i] = (1 + total_var) * \
                torch.sum(torch.log(1 + torch.exp(v))) / (num_pos)

            if use_var_aux_loss:  # compute variance based auxiliary loss
                l1_err = var_regularization(un_flat[i])
                l[i] = 2 * ((1 - weight) * (l[i]) + weight * l1_err)

        return self.reduction(l)


def calc_diversity(wordvec_array, y_i):
    rel_vecs = wordvec_array[:, :, y_i == 1]
    rel_vecs = rel_vecs.squeeze(0)
    if rel_vecs.shape[1] == 1:
        sig = rel_vecs * 0  # det_c = 0
    else:
        sig = torch.var(rel_vecs, dim=1)

    return sig.sum()


def var_regularization(x_i):
    sig2 = torch.var(x_i, dim=1)
    l1_err = torch.norm(sig2, dim=-1, p=1)

    return l1_err


class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading during forward pass'''

    def __init__(self, args, class_task=None):
        super(AsymmetricLoss, self).__init__()
        self.args = args
        self.gamma_neg = args.gamma_neg if args.gamma_neg is not None else 4
        self.gamma_pos = args.gamma_pos if args.gamma_pos is not None else 0.05
        self.clip = args.clip if args.clip is not None else 0.05
        # self.class_task = class_task
        # self.multiset_rank = args.multiset_rank  # Used also to identify multi-task training

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.targets_weights = None

    def forward(self, logits, targets_inputs):
        # if not self.training:  # this is a complicated loss. for validation, just return 0
        #     return 0

        if self.targets is None or self.targets.shape != targets_inputs.shape:
            self.targets = targets_inputs.clone()
        else:
            self.targets.copy_(targets_inputs)
        targets = self.targets

        # initial calculations
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        targets_weights = self.targets_weights
        targets, targets_weights, xs_neg = edit_targets_parital_labels(self.args, targets, targets_weights,
                                                                       xs_neg)
        anti_targets = 1 - targets

        # construct weight matrix for multi-set
        # if False and self.multiset_rank is not None:
        #     self.targets_weights = get_multiset_target_weights(self.targets, self.targets_weights,
        #                                                        self.class_task,
        #                                                        self.multiset_rank)

        # One sided clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # CE loss calculation
        BCE_loss = targets * torch.log(torch.clamp(xs_pos, min=1e-8))
        # if self.args.alpha_pos is not None:
        #     BCE_loss.mul_(self.args.alpha_pos)
        neg_loss = anti_targets * torch.log(torch.clamp(xs_neg, min=1e-8))
        # if self.args.alpha_neg is not None:
        #     neg_loss.mul_(self.args.alpha_neg)
        # BCE_loss.add_(neg_loss)

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets - xs_neg * anti_targets,
                                     self.gamma_pos * targets + self.gamma_neg * anti_targets)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        # multi-task weights
        if hasattr(self, "weight_task_batch"):
            BCE_loss *= self.weight_task_batch

        return -BCE_loss.sum()


def edit_targets_parital_labels(args, targets, targets_weights, xs_neg):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if args.partial_loss_mode is None:
        targets_weights = 1.0
    elif args.partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets[targets == -1] = 0
        targets_weights = 1.0
    elif args.partial_loss_mode == 'negative_backprop':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(
                targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_confused_classes_to_remove_backprop = args.num_classes_to_remove_negative_backprop * \
            targets_weights.shape[0]  # 50 per sample
        negative_backprop_fun_jit(targets, xs_neg, targets_weights,
                                  num_top_confused_classes_to_remove_backprop)

        # set all unsure targets as negative
        targets[targets == -1] = 0

    elif args.partial_loss_mode == 'real_partial':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(
            targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    return targets, targets_weights, xs_neg


def negative_backprop_fun_jit(targets: Tensor, xs_neg: Tensor, targets_weights: Tensor,
                              num_top_confused_classes_to_remove_backprop: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_flatten = xs_neg.flatten()
        ind_class_sort = torch.argsort(xs_neg_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_confused_classes_to_remove_backprop]]] = 0


class PiecewiseLoss(nn.Module):
    def __init__(self, ms=0.1, md=0.3):
        super(PiecewiseLoss, self).__init__()
        self.ms = torch.tensor(ms)
        self.md = torch.tensor(md)

    def forward(self, output1, output2, label1, label2, event):
        Dg = label1 - label2
        Dp = torch.sigmoid(output1 @ event) - torch.sigmoid(output2 @ event)
        Dp = Dp.view(-1, 1)
        # print(Dp.shape)
        # print(event.shape)
        # print(Dp)
        # print(Dp @ event)
        # output1 *= event
        # output2 *= event
        # output1[output1 !=0]
        # output2[output2 !=0]
        loss = 0
        mask_1 = Dg < self.ms
        mask_2 = (Dg >= self.ms) & (Dg <= self.md)
        mask_3 = Dg > self.md

        # if Dg < self.ms:
        loss += (torch.pow(torch.clamp(
            torch.abs(Dp[mask_1])-self.ms, min=0.0), 2)).sum() / 2

        # elif Dg >= self.ms and Dg <= self.md:
        loss += ((torch.pow(torch.clamp(self.ms-Dp[mask_2], min=0.0), 2) +
                  torch.pow(torch.clamp(Dp[mask_2]-self.md, min=0.0), 2))).sum() / 2
        # elif Dg > self.md:

        loss += (torch.pow(torch.clamp(self.md -
                 Dp[mask_3], min=0.0), 2)).sum() / 2

        return loss


class FocalLoss(nn.Module):
    """
    The focal loss for fighting against class-imbalance
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-12  # prevent training from Nan-loss error

    def forward(self, logits, target):
        """
        logits & target should be tensors with shape [batch_size, num_classes]
        """
        probs = torch.sigmoid(logits)
        one_subtract_probs = 1.0 - probs
        # add epsilon
        probs_new = probs + self.epsilon
        one_subtract_probs_new = one_subtract_probs + self.epsilon
        # calculate focal loss
        log_pt = target * \
            torch.log(probs_new) + (1.0 - target) * \
            torch.log(one_subtract_probs_new)
        pt = torch.exp(log_pt)
        focal_loss = -1.0 * (self.alpha * (1 - pt) ** self.gamma) * log_pt
        return torch.mean(focal_loss)


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets *
                       torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                  self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                              self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss
