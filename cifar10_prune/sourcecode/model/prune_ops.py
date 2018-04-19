# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import _pickle as pickle
import model.config
import model.papl


def apply_prune(tensors,prune_Rate,sess):
    """Pruning with given weight.

    Args:
        tensors: Tensor dict.
    Returns:
        pruning index dict.
    """
    # Store nonzero index for each weights.
    dict_nzidxs = {}

    #for each untarget layers
    for untarget in model.config.untarget_layer:
        wl = "w_" + untarget
        tensor = tensors[wl]
        weight = tensor.eval()
        nzidxs1 = abs(weight) != 0
        dict_nzidxs[wl] = nzidxs1
    # For each target layers,
    for target in model.config.target_layer:
        wl = "w_" + target
        #print(wl + " threshold:\t" + str(model.config.th[wl]))
        # Get target layer's weights
        tensor = tensors[wl]
        weight = tensor.eval()

        # Apply pruning
        weight, nzidxs = model.papl.prune_dense(weight, name=wl,prune_rate=prune_Rate)

        # Store pruned weights as tensorflow objects
        dict_nzidxs[wl] = nzidxs
        #tensor.assign(weight)
        #tensor.eval()
        sess.run(tensor.assign(weight))

    return dict_nzidxs,sess


def apply_prune_on_grads(grads_and_vars, dict_nzidxs):
    """Apply pruning on gradients.
    Mask gradients with pruned elements.

    Args:
        grads_and_vars: computed gradient by Optimizer.
        dict_nzidxs: dictionary for each tensor with nonzero indexs.

    Returns:

    """
    # For each pruned weights with nonzero index list,
    for key, nzidxs in dict_nzidxs.items():
        count = 0
        # For each gradients and variables,
        for grad, var in grads_and_vars:
            # Find matched tensor
            if var.name == key + ":0":
                nzidx_obj = tf.cast(tf.constant(nzidxs), tf.float32)
                grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
            count += 1

    return grads_and_vars