import numpy as np
import torch
from scipy import interpolate
from sklearn.metrics import roc_curve, roc_auc_score
import torch.nn.functional as F

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

def metric(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob


def get_err_threhold(pred, target, pos_label=1):
    fpr, tpr, threshold = roc_curve(target, pred, pos_label=pos_label)
    RightIndex=(tpr+(1-fpr)-1)
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th


def calculate(threshold, dist, actual_issame):
    """
    Calculate tp,fp,tn,fn
    :param threshold:
    :param dist: predict
    :param actual_issame: real label
    :return: tp,fp,tn,fn
    """
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def calculate_tpr_fpr(tp,fp,tn,fn):
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    return tpr, fpr

def ACER(tp, fp, tn, fn):
    apcer = fp / (tn*1.0 + fp*1.0)
    npcer = fn / (fn * 1.0 + tp * 1.0)
    acer = (apcer + npcer) / 2.0
    return acer, apcer, npcer

def TPR_FPR(actual_issame, dist):
    fpr, tpr, thresholds = roc_curve(actual_issame, dist)
    _tpr = (tpr)
    _fpr = (fpr)
    tpr = tpr.reshape((tpr.shape[0], 1))
    fpr = fpr.reshape((fpr.shape[0], 1))
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(_fpr, _tpr)
    y = function(scale)
    znew = abs(scale + y - 1)
    eer = scale[np.argmin(znew)]
    FPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    for i, (key, value) in enumerate(FPRs.items()):
        index = np.argwhere(scale == value)
        score = y[index]
        TPRs[key] = float(np.squeeze(score))
    # auc = roc_auc_score(actual_issame, dist)
    return TPRs, eer

def model_performances(probs, labels):
    # err, threshold = get_err_threhold(probs, labels)
    threshold = 0.5
    tp, fp, tn, fn = calculate(threshold, probs, labels)
    acer, apcer, npcer = ACER(tp, fp, tn, fn)
    acc = float(tp + tn) / probs.shape[0]
    # eval = {"ACER": acer, "APCER": apcer, "BPCER": npcer, "ACC": acc, "FPR": FPR, "TPR": TPR, "fpr": fpr, "tpr": tpr}
    tpr_fpr, err = TPR_FPR(labels, probs)
    eval = {"ACER": acer, "APCER": apcer, "BPCER": npcer, "ACC": acc, "ERR": err}

    return eval, tpr_fpr, threshold