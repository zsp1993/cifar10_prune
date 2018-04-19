#!/usr/bin/python
import model.thspace as ths


def _complex_concat(a, b):
    tmp = []
    for i in a:
        for j in b:
            tmp.append(i+j)
    return tmp


def _add_prefix(l, prefix="w_"):
    return [prefix + elem for elem in l]


# Pruning threshold setting (90 % off)
th = ths.th90

# CNN settings for pruned training
untarget_layer = ["conv1", "fc1", "fc2", "fc3"]
target_layer = ["conv2","conv3"]

# Retrain iteration after pruning
retrain_iterations = 10

# Data settings
show_zero = False

# Train directory
train_dir = 'train'

# Output data lists: do not change this
target_all_layer = _add_prefix(target_layer)

target_dat = _complex_concat(target_all_layer, [".dat"])
target_p_dat = _complex_concat(target_all_layer, ["_p.dat"])
target_tp_dat = _complex_concat(target_all_layer, ["_tp.dat"])

weight_all = target_dat + target_p_dat + target_tp_dat
syn_all = ["in_conv1.syn", "in_conv2.syn", "in_fc1.syn", "in_fc2.syn"]

# Graph settings
alpha = 0.75
color = "green"
pdf_prefix = ""

