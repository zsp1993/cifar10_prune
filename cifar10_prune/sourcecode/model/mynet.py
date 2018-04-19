# -*- coding: utf-8 -*-

import model.input_dataset
import time
import os
import tarfile
import tensorflow as tf
import model.prune_ops
import numpy as np
import shutil
from tensorflow.python.framework import ops

#dropoup参数
keep_prob = tf.placeholder("float")
#学习率
learn_rate = tf.placeholder("float")
#样本大小
picture = tf.placeholder("float", shape=[None,24,24,3])

#卷积过程
def conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.conv2d(x,w,
                        strides,padding='SAME')
def depthwise_conv2d(x,w,strides=[1,2,2,1]):
    return tf.nn.depthwise_conv2d(x,w,
                                  strides,padding='SAME')
#池化过程
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                        strides=[1,2,2,1],padding='SAME')

def onehot(y):
    '''转化为one-hot 编码'''
    size1 = tf.size(y)
    y = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, size1, 1), 1)
    concated = tf.concat([indices, y], 1)
    y = tf.sparse_to_dense(concated, tf.stack([size1, 10]), 1.0, 0.0)
    return y

class MyNet():
    def __init__(self,looP=200000,batch_Size=128,model_Save_path='mynet/save_net.ckpt',model_pruned_Save_path='mynet_pruned/save_net.ckpt',
                 prune_Rate=0.9,retrain_looP=20000,model_Translate_path = 'mynet/translate_model'):
        self.loop = looP
        self.retrain_loop =retrain_looP
        self.batch_size = batch_Size
        self.model_save_path = model_Save_path
        self.model_pruned_save_path = model_pruned_Save_path
        self.model_translate_save_path = model_Translate_path
        self.prune_rate = prune_Rate
        self.dense_w={
            "w_conv1":tf.Variable(tf.truncated_normal([3,3,3,32],stddev=0.13),name="w_conv1"),
            "b_conv1": tf.Variable(tf.constant(0.13, shape=[32]), name="b_conv1"),
            "w_conv2": tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name="w_conv2"),
            "b_conv2": tf.Variable(tf.constant(0.13, shape=[64]), name="b_conv2"),
            "w_conv3": tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.13), name="w_conv3"),
            "b_conv3": tf.Variable(tf.constant(0.13, shape=[128]), name="b_conv3"),
            "w_fc1": tf.Variable(tf.truncated_normal([24 * 24 * 128, 128], stddev=0.13), name="w_fc1"),
            "b_fc1": tf.Variable(tf.constant(0.13, shape=[128]), name="b_fc1"),
            "w_fc2": tf.Variable(tf.truncated_normal([128, 128], stddev=0.13), name="w_fc2"),
            "b_fc2": tf.Variable(tf.constant(0.13, shape=[128]), name="b_fc2"),
            "w_fc3": tf.Variable(tf.truncated_normal([128, 10], stddev=0.13), name="w_fc3"),
            "b_fc3": tf.Variable(tf.constant(0.13, shape=[10]), name="b_fc3")
        }
    def gennet(self,x):
        #第一卷积层
        h_conv1 = tf.nn.relu(conv2d(x,self.dense_w["w_conv1"],[1,1,1,1]) + self.dense_w["b_conv1"])

        #第二卷积层
        h_conv2 = tf.nn.relu(conv2d(h_conv1,self.dense_w["w_conv2"],[1,1,1,1]) + self.dense_w["b_conv2"])

        #第三卷积层
        h_conv3 = tf.nn.relu(conv2d(h_conv2,self.dense_w["w_conv3"],[1,1,1,1]) + self.dense_w["b_conv3"])

        #第一全连接层
        #h_conv3_flat = tf.reshape(h_conv3,[-1,24*24*(int(128/self.ratio)+1)])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 24 * 24 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,self.dense_w["w_fc1"]) + self.dense_w["b_fc1"])

        #第二全连接层
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1,self.dense_w["w_fc2"]) + self.dense_w["b_fc2"])

        # 防止过拟合的dropout
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        # 第三全连接层
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, self.dense_w["w_fc3"]) + self.dense_w["b_fc3"])

        return y_conv

    def test(self,data_path,message="test accuracy"):
        """Test.
            Returns:
                proportion for correct prediction.
            """
        # To avoid OOM, run validation with 500/10000 test dataset
        result = 0

        #batch = mnist.test.next_batch(500)
        img_batch, label_batch = model.input_dataset.preprocess_input_data(data_path,500,True)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        y_label = tf.cast(x=label_batch,
                              dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
        y_infer = self.gennet(img_batch)
        y_label = onehot(y_label)
        correct_prediction = tf.equal(tf.argmax(y_infer, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        with tf.Session() as sess:
            # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
            saver2 = tf.train.Saver()
            save_path = saver2.restore(sess, self.model_save_path)
            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            print("gen testing###############################")
            for i in range(20):
                result += accuracy.eval(feed_dict={keep_prob: 1.0})
            result /= 20

            print(message + " %g\n" % result)

            coord.request_stop()
            coord.join(threads)

    def train(self,data_path):
        img_batch, label_batch = model.input_dataset.preprocess_input_data(data_path,
                                                                           self.batch_size)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        y_label = tf.cast(x=label_batch,dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
        y_infer= self.gennet(img_batch)

        #cross_entropy = -tf.reduce_sum(y_label*tf.log(y_infer))
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_infer, labels=y_label, name='likelihood_loss')
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
        trainer = tf.train.AdamOptimizer(learn_rate)
        grads_and_vars = trainer.compute_gradients(cross_entropy_loss)
        train_step = trainer.apply_gradients(grads_and_vars)
        y_label = onehot(y_label)
        correct_prediction = tf.equal(tf.argmax(y_infer, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            #判断是否有模型保存文件，有的话用以初始化，否则随机初始化
            try:
                print("mark2_____________________")
                # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
                saver2 = tf.train.Saver()
                save_path = saver2.restore(sess, self.model_save_path)
            except:
                print("mark1_____________________")
                init = tf.global_variables_initializer()
                sess.run(init)
            #线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            # threads = tf.train.start_queue_runners(sess=sess)
            new_learn_rate = 1e-4
            rate = 10 ** (-4.0 / (self.loop / 2000))
            t0 = time.time()

            #训练过程
            for i in range(self.loop):
                if (i + 1) % 2000 == 0:
                    new_learn_rate = new_learn_rate * rate
                    save_path = saver.save(sess, self.model_save_path)
                    # 输出保存路径
                    print('Save to path: ', save_path)
                # with tf.device("/gpu:0"):
                # mse,_=sess.run([model,train_step],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
                if i % 30 == 0:
                    cost1,_,accuracy1 = sess.run([cross_entropy_loss, train_step,accuracy], feed_dict={keep_prob: 1.0, learn_rate: new_learn_rate})
                    # print(sess.run(y_conv, feed_dict={keep_prob: 1}))
                    print("step %d,cross_entropy_loss is %g, training accuracy %g\n,time is %g" % (i,cost1, accuracy1,time.time()-t0))
                    print("####################################")
                else:
                    _,_,accuracy1 = sess.run([cross_entropy_loss,train_step,accuracy], feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})

            coord.request_stop()
            coord.join(threads)
    def readmeta(self,path):
        from tensorflow.python.tools import freeze_graph
        ret = freeze_graph._parse_input_meta_graph_proto(path, True)
        print(ret)
    def prune_and_retrain(self,data_path,irr):
        '''
        # self.readmeta("/home/zhangsp/PycharmProjects/cifar10_prune/sourcecode/main/mynet/save_net.ckpt.meta")
        self.readmeta("/home/zhangsp/PycharmProjects/cifar10_prune/sourcecode/main/mynet_pruned/save_net.ckpt.meta")
        raise
        '''
        img_batch, label_batch = model.input_dataset.preprocess_input_data(data_path,self.batch_size)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作

        y_label = tf.cast(x=label_batch, dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
        y_infer = self.gennet(img_batch)
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_infer, labels=y_label,name='likelihood_loss')
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')

        y_label = onehot(y_label)
        correct_prediction = tf.equal(tf.argmax(y_infer, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #trainer = tf.train.AdamOptimizer(learn_rate)
        #grads_and_vars = trainer.compute_gradients(cross_entropy_loss)
        #grads_and_vars = model.prune_ops.apply_prune_on_grads(grads_and_vars, dict_nzidx)
        #train_step = trainer.apply_gradients(grads_and_vars)
        #train_step = trainer.apply_gradients(grads_and_vars)
        save_var = []
        for value in self.dense_w.values():
            save_var.append(value)
        saver2 = tf.train.Saver(save_var)
        saver = tf.train.Saver(save_var)
        # 循环剪枝
        with tf.Session() as sess:
        #for irr in range(20):
            # save_path = saver.save(sess, self.model_pruned_save_path)
            print("------------------------------------------------------------------------")
            print("current prune part ", 1 - self.prune_rate**(irr+1))
            print("------------------------------------------------------------------------")
            #g1 = tf.Graph()
            #剪枝并再次训练
            #恢复初始模型
            if irr==0:
                saver2.restore(sess, self.model_save_path)
            else:
                saver2.restore(sess,self.model_pruned_save_path)

            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            #threads = tf.train.start_queue_runners(sess=sess)
            new_learn_rate = 1e-5
            t0 = time.time()
            rate = 10 ** (-4.0 / (self.retrain_loop / 2000))

            #对权值参数剪枝,修剪掉多少比例的权值
            #对网络进行再训练
            # Compute gradient and remove change for pruning weight
            trainer = tf.train.AdamOptimizer(learn_rate)
            grads_and_vars = trainer.compute_gradients(cross_entropy_loss)

            dict_nzidx, sess = model.prune_ops.apply_prune(self.dense_w, 1-self.prune_rate ** (irr + 1), sess)
            grads_and_vars_new = model.prune_ops.apply_prune_on_grads(grads_and_vars, dict_nzidx)
            train_step = trainer.apply_gradients(grads_and_vars_new)

            # Initialize firstly touched variables (mostly from accuracy calc.)
            for var in tf.global_variables():
                if tf.is_variable_initialized(var).eval() == False:
                    sess.run(tf.variables_initializer([var]))
            # 训练过程
                # saver = tf.train.Saver(save_var)
            mark_zsp = 0
            for i in range(self.retrain_loop):
                '''
                if (i + 1) % 2000 == 0:
                    new_learn_rate = new_learn_rate * rate
                    #path = os.path.dirname(self.model_pruned_save_path)                        #shutil.rmtree(path)
                    #save_path = saver.save(sess, self.model_pruned_save_path)
                    save_path = saver.save(sess, self.model_pruned_save_path)
                    # 输出保存路径
                    print('Save to path: ', save_path)
                '''
                # with tf.device("/gpu:0"):
                # mse,_=sess.run([model,train_step],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
                if i % 30 == 0:
                    cost1, _, accuracy1 = sess.run([cross_entropy_loss, train_step, accuracy],feed_dict={keep_prob: 1.0, learn_rate: new_learn_rate})
                    # print(sess.run(y_conv, feed_dict={keep_prob: 1}))
                    print("step %d,cross_entropy_loss is %g, training accuracy %g\n,time is %g" % (
                    i, cost1, accuracy1, time.time() - t0))
                    #print(self.dense_w["w_fc2"][1:10,1:30].eval())
                    print("#####################")
                    if accuracy1 > 0.61:
                        mark_zsp += 1
                        print ("mark_zsp is ：",mark_zsp)
                    else:
                        mark_zsp = 0
                    if mark_zsp >=5:
                        break
                else:
                    _, _, accuracy1 = sess.run([cross_entropy_loss, train_step, accuracy],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
                #save_path = saver.save(sess, self.model_pruned_save_path)
                #path = os.path.dirname(self.model_pruned_save_path)
                #shutil.rmtree(path)
            save_path = saver.save(sess, self.model_pruned_save_path)
            coord.request_stop()
            coord.join(threads)

    def fine_tune_after_pruned(self,data_path,ratio=8):
        #默认一次剪掉10%

        img_batch, label_batch = model.input_dataset.preprocess_input_data(data_path,
                                                                           self.batch_size)  # 输入图像的预处理，包括亮度、对比度、图像翻转等操作
        y_label = tf.cast(x=label_batch, dtype=tf.int32)  # 强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
        save_var = []
        for value in self.dense_w.values():
            save_var.append(value)
        saver = tf.train.Saver(save_var)

        with tf.Session() as sess:
            saver.restore(sess, self.model_pruned_save_path)
            #新建剪枝后的第一层
            target = model.config.target_layer[0]
            wl = "w_" + target
            wl_b = "b_" + target
            tensor = self.dense_w[wl]
            tensor_b = self.dense_w[wl_b]
            weight = tensor.eval()
            bias = tensor_b.eval()
            out_channel = np.shape(weight)[-1]
            out_channel_new = int(out_channel/ratio)
            shape_new = list(np.shape(weight))
            shape_new[-1] = out_channel_new + 1
            weight_new = np.zeros(shape_new)
            bias_new = np.zeros(shape_new[-1])

            for i in range(out_channel_new-1):
                for j in range(ratio):
                    weight_new[:, :, :, i] += weight[:, :, :, i*ratio+j]
                    bias_new[i] += bias[i * ratio + j]
            for k in range(ratio*(out_channel_new-1),out_channel):
                weight_new[:, :, :, out_channel_new-1] += weight[:, :, :, k]
                bias_new[out_channel_new - 1] += bias[k]

            tensor_new_out = tf.Variable(tf.truncated_normal(shape_new, stddev=0.13), name=wl + "_new")
            tensor_new_b = tf.Variable(tf.truncated_normal([shape_new[-1]], stddev=0.13), name=wl_b + "_new")
            sess.run(tensor_new_out.assign(weight_new))
            sess.run(tensor_new_b.assign(bias_new))
            self.dense_w[wl] = tensor_new_out
            self.dense_w[wl_b] = tensor_new_b

            print(np.shape(self.dense_w[wl]))
            #新建剪枝后的第二层
            target = model.config.target_layer[1]
            wl = "w_" + target
            tensor = self.dense_w[wl]
            weight = tensor.eval()
            in_channel = np.shape(weight)[-2]
            in_channel_new = int(in_channel / ratio)
            shape_new = list(np.shape(weight))
            shape_new[-2] = in_channel_new + 1
            weight_new = np.zeros(shape_new)

            for i in range(in_channel_new - 1):
                for j in range(ratio):
                    weight_new[:, :,i, :] += weight[:, :, i * ratio + j, :]
            for k in range(ratio * (in_channel_new - 1), in_channel):
                weight_new[:,:,in_channel_new - 1,:] += weight[:,:,k,:]

            tensor_new_in = tf.Variable(tf.truncated_normal(shape_new, stddev=0.13), name=wl + "_new")

            sess.run(tensor_new_in.assign(weight_new))
            self.dense_w[wl] = tensor_new_in
            print(np.shape(self.dense_w[wl]))

            y_infer = self.gennet(img_batch)

            # cross_entropy = -tf.reduce_sum(y_label*tf.log(y_infer))
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_infer, labels=y_label,
                                                                                name='likelihood_loss')
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
            trainer = tf.train.AdamOptimizer(learn_rate)
            grads_and_vars = trainer.compute_gradients(cross_entropy_loss)
            train_step = trainer.apply_gradients(grads_and_vars)
            y_label = onehot(y_label)
            correct_prediction = tf.equal(tf.argmax(y_infer, 1), tf.argmax(y_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # 线程需要同步
            coord = tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(coord=coord)
            # threads = tf.train.start_queue_runners(sess=sess)
            new_learn_rate = 1e-4
            rate = 10 ** (-4.0 / (self.loop / 2000))
            t0 = time.time()

            # Initialize firstly touched variables (mostly from accuracy calc.)
            for var in tf.global_variables():
                if tf.is_variable_initialized(var).eval() == False:
                    sess.run(tf.variables_initializer([var]))

            # 训练过程
            for i in range(self.loop):
                if (i + 1) % 2000 == 0:
                    new_learn_rate = new_learn_rate * rate
                    save_path = saver.save(sess, self.model_translate_save_path)
                    # 输出保存路径
                    print('Save to path: ', save_path)
                # with tf.device("/gpu:0"):
                # mse,_=sess.run([model,train_step],feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})
                if i % 30 == 0:
                    cost1, _, accuracy1 = sess.run([cross_entropy_loss, train_step, accuracy],
                                                   feed_dict={keep_prob: 1.0, learn_rate: new_learn_rate})
                    # print(sess.run(y_conv, feed_dict={keep_prob: 1}))
                    print("step %d,cross_entropy_loss is %g, training accuracy %g\n,time is %g" % (
                    i, cost1, accuracy1, time.time() - t0))
                    print("####################################")
                else:
                    _, _, accuracy1 = sess.run([cross_entropy_loss, train_step, accuracy],
                                               feed_dict={keep_prob: 0.5, learn_rate: new_learn_rate})

            coord.request_stop()
            coord.join(threads)





    def process(self,datapath,option=1):
        if option==1:
            self.test(datapath)
        if option==2:
            self.train(datapath)
        if option==3:
            #0.9**25=0.072
            for irr in range(7,26):
                self.prune_and_retrain(datapath,irr)
        if option==4:
            self.fine_tune_after_pruned(datapath)


if __name__ == "main":
    MyNet().readmeta()




























