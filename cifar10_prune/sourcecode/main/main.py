# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
from model.mynet import *


datasets_file_path = os.path.dirname(os.path.dirname(os.getcwd()))+'/datasets/cifar-10-batches-bin'

model1 = MyNet()
#选项1表示测试，选项2表示训练,选项3表示剪枝再训练,选项4表示转换模型
model1.process(datasets_file_path,3)




