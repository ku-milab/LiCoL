import tensorflow.keras as keras
from sklearn import metrics
import numpy as np
import CMG_utils as utils
import CMG_layers as layers
import CMG_config as config
K = keras.backend

def ACC(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

def MAUC(data, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.
    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    import itertools as itertools
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]
    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(data, zero_label=pairing[0], one_label=pairing[1]) + a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0
    return sum_avals * (2 / float(num_classes * (num_classes - 1)))  # Eqn 7

def a_value(probabilities, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.
    Returns:
        The A-value as a floating point.
    """
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0].item(), instance[zero_label + 1].item()))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])
    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class
    # print('Before: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
    if n0 == 0:
        n0 = 1e-10
    elif n1 == 0:
        n1 = 1e-10
    else:
        pass
    # print('After: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
    return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1)  # Eqn 3

def evaluation_matrics(y_true, y_pred):
    acc = K.mean(K.equal(y_true, np.argmax(y_pred, axis=-1)))
    auc = metrics.roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, np.argmax(y_pred, axis=-1)).ravel()
    sen = tp / (tp + fn)
    spe = tn / (fp + tn)
    return acc, auc, sen, spe

def ncc(a, v, zero_norm=False):
    """
    zero_norm = False:
    :return NCC
    zero_norm = True:
    :return ZNCC
    """
    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)
    else:
        a = (a) / (np.std(a) * len(a))  # obser = layers.flatten()(np.expand_dims(observed_map[idx], axis=0))
        v = (v) / np.std(v)  # pred = layers.flatten()(np.expand_dims(pred_map[idx], axis=0))
    return np.correlate(a, v)

def ncc_evaluation(cls_model, gen_model):
    long_cn, long_mci, long_ad = utils.Utils().load_longitudinal_data()  # ADNI or GARD
    gt_adTOcn, gt_adTOmci, gt_mciTOcn = (long_cn - long_ad), (long_mci - long_ad), (long_cn - long_mci)
    gt_cnTOad, gt_mciTOad, gt_cnTOmci = (-gt_adTOcn), (-gt_adTOmci), (-gt_mciTOcn)

    if config.scenario == "CN_MCI":
        zero_dat, zero_lbl, one_dat, one_lbl = long_cn, np.zeros(len(long_cn)).astype("int32"), long_mci, np.ones(len(long_mci)).astype("int32")
        zero_cls, one_cls = cls_model(zero_dat)["cls_out"], cls_model(one_dat)["cls_out"]
        gt_ncc_p, gt_ncc_n = gt_mciTOcn, gt_cnTOmci

    elif config.scenario == "MCI_AD":
        zero_dat, zero_lbl, one_dat, one_lbl = long_mci, np.zeros(len(long_mci)).astype("int32"), long_ad, np.ones(len(long_ad)).astype("int32")
        zero_cls, one_cls = cls_model(zero_dat)["cls_out"], cls_model(one_dat)["cls_out"]
        gt_ncc_p, gt_ncc_n = gt_adTOmci, gt_mciTOad

    elif config.scenario == "CN_AD":
        zero_dat, zero_lbl, one_dat, one_lbl = long_cn, np.zeros(len(long_cn)).astype("int32"), long_ad, np.ones(len(long_ad)).astype("int32")
        zero_cls, one_cls = cls_model(zero_dat)["cls_out"], cls_model(one_dat)["cls_out"]
        gt_ncc_p, gt_ncc_n = gt_adTOcn, gt_cnTOad

    else: print("Scenario selection error!!")

    # NCC (+) #
    ncc_p = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=zero_cls)
    cfmap = gen_model({"gen_in": one_dat, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = one_dat + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    z_like_acc = K.mean(K.equal(zero_lbl, pred))
    gtmap, cfmap = layers.flatten()(np.array(gt_ncc_p)).astype("float32"), layers.flatten()(cfmap)

    for i in range(len(cfmap)):
        ncc_p += ncc(a=gtmap[i], v=cfmap[i])
    ncc_p /= len(cfmap)

    # NCC (-) #
    ncc_n = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=one_cls)
    cfmap = gen_model({"gen_in": zero_dat, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = zero_dat + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    o_like_acc = K.mean(K.equal(one_lbl, pred))
    gtmap, cfmap = layers.flatten()(np.array(gt_ncc_n)).astype("float32"), layers.flatten()(cfmap)

    for i in range(len(cfmap)):
        ncc_n += ncc(a=gtmap[i], v=cfmap[i])
    ncc_n /= len(cfmap)

    return ncc_p, ncc_n, z_like_acc, o_like_acc
