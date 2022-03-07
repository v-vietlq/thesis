import torch
import numpy as np
# from fastai2.torch_core import to_detach, flatten_check, store_attr


epsilon = 1e-8

def average_precision(output, target):
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def AP_partial(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    cnt_class_with_no_neg = 0
    cnt_class_with_no_pos = 0
    cnt_class_with_no_labels = 0

    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]

        # Filter out samples without label
        idx = (targets != -1)
        scores = scores[idx]
        targets = targets[idx]
        if len(targets) == 0:
            cnt_class_with_no_labels += 1
            ap[k] = -1
            continue
        elif sum(targets) == 0:
            cnt_class_with_no_pos += 1
            ap[k] = -1
            continue
        if sum(targets == 0) == 0:
            cnt_class_with_no_neg += 1
            ap[k] = -1
            continue
        # compute average precision
        ap[k] = average_precision(scores, targets)

    print('#####DEBUG num -1 classes {} '.format(sum(ap == -1)))
    idx_valid_classes = np.where(ap != -1)[0]
    ap_valid = ap[idx_valid_classes]
    map = 100 * np.mean(ap_valid)

    # Compute macro-map
    targs_macro_valid = targs[:, idx_valid_classes].copy()
    targs_macro_valid[targs_macro_valid <= 0] = 0  # set partial labels as negative
    n_per_class = targs_macro_valid.sum(0)  # get number of targets for each class
    n_total = np.sum(n_per_class)
    map_macro = 100 * np.sum(ap_valid * n_per_class / n_total)

    return map

def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def enable_detach():
    to_detach.__defaults__ = (True, True)

def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred,targ = flatten_check(inp.argmax(dim=axis), targ)
    return (pred == targ).float().mean()


def average_precision(output, target):
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_/(total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100*ap.mean()



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self): self.reset()

    def reset(self): self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(model, val_loader):

    preds, targets = [], []
    with torch.no_grad():
        for input, target in val_loader:
            input = input.squeeze(0).cuda()
            target = target.cuda()
            logits = model(input)
            pred = torch.sigmoid(logits).cpu()        
            preds.append(pred.cpu())
            if target.size(0) != pred.size()[0]:
                target = target.repeat(pred.size()[0], 1)
            targets.append(target.cpu())
    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    enable_detach()

    return mAP_score





def validate_model(model, val_loader, threshold):


  preds, targs = [], []
  with torch.no_grad():
    for input, target in val_loader:
    #   batch_size, time_steps, channels, height, width = input.size()
    #   input = input.view(batch_size * time_steps, channels, height, width)
   
      input = input.cuda()
      target = target.cuda()

      logits = model(input)
      pred = torch.sigmoid(logits)
      # np.where(preds > thresh)[0]
      pred[(pred >= threshold)] = 1
      pred[(pred < threshold)] = 0
      preds.append(pred)
      targs.append(target)

      
  preds,targs = torch.cat(preds).cpu().detach().numpy() , torch.cat(targs).cpu().detach().numpy()
   # acc = accuracy(preds, targs)
  #for multi-label
  # acc = mAP(targs, preds)
  acc = AP_partial(targs, preds)

  return acc