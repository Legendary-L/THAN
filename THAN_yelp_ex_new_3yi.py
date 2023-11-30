import os
import re
import numpy as np
import tensorflow as tf
from models import HeteGAT_multi
from utils import process

import numpy as np
from sklearn.model_selection import train_test_split
os.environ['KERAS_BACKEND']='tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from tcn import TCN
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from sklearn.metrics import f1_score





config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'acm'
# featype = 'fea'
featype = 'adj'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 200
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



















def load_data_dblp1(path='/data/wby_data/code/yelp1.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['BRURB'] - np.eye(N), data['BSB'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]

    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp1()

if featype == 'adj':
    fea_list = adj_list

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]



train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],

                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        print('start knn, kmean.....')
        xx1 = np.expand_dims(jhy_final_embedding, axis=0)[train_mask+val_mask+test_mask]
        print("xx1shape: ", end=" ")
        print(xx1.shape)


        # xx = xx / LA.norm(xx, axis=1)
        yy1 = y_test[train_mask+val_mask+test_mask]
        print("yy1shape: ", end=" ")
        print(yy1.shape)
        # print(yy)
        # print(yy[0])

        print('xx1: {}, yy1: {}'.format(xx1.shape, yy1.shape))

        sess.close()
print("HAN1结束了")














#
#
#
# def load_data_dblp2(path='/data/wby_data/code/yelp2.mat'):
#     data = sio.loadmat(path)
#     truelabels, truefeatures = data['label'], data['feature'].astype(float)
#     N = truefeatures.shape[0]
#     rownetworks = [data['BRURB'] - np.eye(N), data['BSB'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
#
#     y = truelabels
#     train_idx = data['train_idx']
#     val_idx = data['val_idx']
#     test_idx = data['test_idx']
#
#     train_mask = sample_mask(train_idx, y.shape[0])
#     val_mask = sample_mask(val_idx, y.shape[0])
#     test_mask = sample_mask(test_idx, y.shape[0])
#
#     y_train = np.zeros(y.shape)
#     y_val = np.zeros(y.shape)
#     y_test = np.zeros(y.shape)
#     y_train[train_mask, :] = y[train_mask, :]
#     y_val[val_mask, :] = y[val_mask, :]
#     y_test[test_mask, :] = y[test_mask, :]
#
#     # return selected_idx, selected_idx_2
#     print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
#                                                                                           y_val.shape,
#                                                                                           y_test.shape,
#                                                                                           train_idx.shape,
#                                                                                           val_idx.shape,
#                                                                                           test_idx.shape))
#     truefeatures_list = [truefeatures, truefeatures, truefeatures]
#     return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask
#
#
# # use adj_list as fea_list, have a try~
# adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp2()
#
# if featype == 'adj':
#     fea_list = adj_list
#
# nb_nodes = fea_list[0].shape[0]
# ft_size = fea_list[0].shape[1]
# nb_classes = y_train.shape[1]
#
# # adj = adj.todense()
#
# # features = features[np.newaxis]  # [1, nb_node, ft_size]
# fea_list = [fea[np.newaxis] for fea in fea_list]
# adj_list = [adj[np.newaxis] for adj in adj_list]
# y_train = y_train[np.newaxis]
# y_val = y_val[np.newaxis]
# y_test = y_test[np.newaxis]
#
#
#
# train_mask = train_mask[np.newaxis]
# val_mask = val_mask[np.newaxis]
# test_mask = test_mask[np.newaxis]
#
# biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]
#
# print('build graph...')
# with tf.Graph().as_default():
#     with tf.name_scope('input'):
#         ftr_in_list = [tf.placeholder(dtype=tf.float32,
#                                       shape=(batch_size, nb_nodes, ft_size),
#                                       name='ftr_in_{}'.format(i))
#                        for i in range(len(fea_list))]
#         bias_in_list = [tf.placeholder(dtype=tf.float32,
#                                        shape=(batch_size, nb_nodes, nb_nodes),
#                                        name='bias_in_{}'.format(i))
#                         for i in range(len(biases_list))]
#         lbl_in = tf.placeholder(dtype=tf.int32, shape=(
#             batch_size, nb_nodes, nb_classes), name='lbl_in')
#         msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
#                                 name='msk_in')
#         attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
#         ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
#         is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
#     # forward
#     logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
#                                                        attn_drop, ffd_drop,
#                                                        bias_mat_list=bias_in_list,
#                                                        hid_units=hid_units, n_heads=n_heads,
#                                                        residual=residual, activation=nonlinearity)
#
#     # cal masked_loss
#     log_resh = tf.reshape(logits, [-1, nb_classes])
#     lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
#     msk_resh = tf.reshape(msk_in, [-1])
#     loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
#     accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
#     # optimzie
#     train_op = model.training(loss, lr, l2_coef)
#
#     saver = tf.train.Saver()
#
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#
#     vlss_mn = np.inf
#     vacc_mx = 0.0
#     curr_step = 0
#
#     with tf.Session(config=config) as sess:
#         sess.run(init_op)
#
#         train_loss_avg = 0
#         train_acc_avg = 0
#         val_loss_avg = 0
#         val_acc_avg = 0
#
#         saver.restore(sess, checkpt_file)
#         print('load model from : {}'.format(checkpt_file))
#         ts_size = fea_list[0].shape[0]
#         ts_step = 0
#         ts_loss = 0.0
#         ts_acc = 0.0
#
#         while ts_step * batch_size < ts_size:
#             # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
#             fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
#                    for i, d in zip(ftr_in_list, fea_list)}
#             fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
#                    for i, d in zip(bias_in_list, biases_list)}
#             fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
#                    msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
#
#                    is_train: False,
#                    attn_drop: 0.0,
#                    ffd_drop: 0.0}
#
#             fd = fd1
#             fd.update(fd2)
#             fd.update(fd3)
#             loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
#                                                                   feed_dict=fd)
#             ts_loss += loss_value_ts
#             ts_acc += acc_ts
#             ts_step += 1
#
#         print('Test loss:', ts_loss / ts_step,
#               '; Test accuracy:', ts_acc / ts_step)
#
#         print('start knn, kmean.....')
#         xx2 = np.expand_dims(jhy_final_embedding, axis=0)[train_mask+val_mask+test_mask]
#         print("xx2shape: ", end=" ")
#         print(xx2.shape)
#
#
#         # xx = xx / LA.norm(xx, axis=1)
#         yy2 = y_test[train_mask+val_mask+test_mask]
#         print("yy2shape: ", end=" ")
#         print(yy2.shape)
#         # print(yy)
#         # print(yy[0])
#
#         print('xx2: {}, yy2: {}'.format(xx2.shape, yy2.shape))
#
#         sess.close()
# print("HAN2结束了")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def load_data_dblp3(path='/data/wby_data/code/yelp3.mat'):
#     data = sio.loadmat(path)
#     truelabels, truefeatures = data['label'], data['feature'].astype(float)
#     N = truefeatures.shape[0]
#     rownetworks = [data['BRURB'] - np.eye(N), data['BSB'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]
#
#     y = truelabels
#     train_idx = data['train_idx']
#     val_idx = data['val_idx']
#     test_idx = data['test_idx']
#
#     train_mask = sample_mask(train_idx, y.shape[0])
#     val_mask = sample_mask(val_idx, y.shape[0])
#     test_mask = sample_mask(test_idx, y.shape[0])
#
#     y_train = np.zeros(y.shape)
#     y_val = np.zeros(y.shape)
#     y_test = np.zeros(y.shape)
#     y_train[train_mask, :] = y[train_mask, :]
#     y_val[val_mask, :] = y[val_mask, :]
#     y_test[test_mask, :] = y[test_mask, :]
#
#     # return selected_idx, selected_idx_2
#     print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
#                                                                                           y_val.shape,
#                                                                                           y_test.shape,
#                                                                                           train_idx.shape,
#                                                                                           val_idx.shape,
#                                                                                           test_idx.shape))
#     truefeatures_list = [truefeatures, truefeatures, truefeatures]
#     return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask
#
#
# # use adj_list as fea_list, have a try~
# adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp3()
#
# if featype == 'adj':
#     fea_list = adj_list
#
# nb_nodes = fea_list[0].shape[0]
# ft_size = fea_list[0].shape[1]
# nb_classes = y_train.shape[1]
#
# # adj = adj.todense()
#
# # features = features[np.newaxis]  # [1, nb_node, ft_size]
# fea_list = [fea[np.newaxis] for fea in fea_list]
# adj_list = [adj[np.newaxis] for adj in adj_list]
# y_train = y_train[np.newaxis]
# y_val = y_val[np.newaxis]
# y_test = y_test[np.newaxis]
#
#
#
# train_mask = train_mask[np.newaxis]
# val_mask = val_mask[np.newaxis]
# test_mask = test_mask[np.newaxis]
#
# biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]
#
# print('build graph...')
# with tf.Graph().as_default():
#     with tf.name_scope('input'):
#         ftr_in_list = [tf.placeholder(dtype=tf.float32,
#                                       shape=(batch_size, nb_nodes, ft_size),
#                                       name='ftr_in_{}'.format(i))
#                        for i in range(len(fea_list))]
#         bias_in_list = [tf.placeholder(dtype=tf.float32,
#                                        shape=(batch_size, nb_nodes, nb_nodes),
#                                        name='bias_in_{}'.format(i))
#                         for i in range(len(biases_list))]
#         lbl_in = tf.placeholder(dtype=tf.int32, shape=(
#             batch_size, nb_nodes, nb_classes), name='lbl_in')
#         msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
#                                 name='msk_in')
#         attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
#         ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
#         is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
#     # forward
#     logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
#                                                        attn_drop, ffd_drop,
#                                                        bias_mat_list=bias_in_list,
#                                                        hid_units=hid_units, n_heads=n_heads,
#                                                        residual=residual, activation=nonlinearity)
#
#     # cal masked_loss
#     log_resh = tf.reshape(logits, [-1, nb_classes])
#     lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
#     msk_resh = tf.reshape(msk_in, [-1])
#     loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
#     accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
#     # optimzie
#     train_op = model.training(loss, lr, l2_coef)
#
#     saver = tf.train.Saver()
#
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#
#     vlss_mn = np.inf
#     vacc_mx = 0.0
#     curr_step = 0
#
#     with tf.Session(config=config) as sess:
#         sess.run(init_op)
#
#         train_loss_avg = 0
#         train_acc_avg = 0
#         val_loss_avg = 0
#         val_acc_avg = 0
#
#         saver.restore(sess, checkpt_file)
#         print('load model from : {}'.format(checkpt_file))
#         ts_size = fea_list[0].shape[0]
#         ts_step = 0
#         ts_loss = 0.0
#         ts_acc = 0.0
#
#         while ts_step * batch_size < ts_size:
#             # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
#             fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
#                    for i, d in zip(ftr_in_list, fea_list)}
#             fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
#                    for i, d in zip(bias_in_list, biases_list)}
#             fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
#                    msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
#
#                    is_train: False,
#                    attn_drop: 0.0,
#                    ffd_drop: 0.0}
#
#             fd = fd1
#             fd.update(fd2)
#             fd.update(fd3)
#             loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
#                                                                   feed_dict=fd)
#             ts_loss += loss_value_ts
#             ts_acc += acc_ts
#             ts_step += 1
#
#         print('Test loss:', ts_loss / ts_step,
#               '; Test accuracy:', ts_acc / ts_step)
#
#         print('start knn, kmean.....')
#         xx3 = np.expand_dims(jhy_final_embedding, axis=0)[train_mask+val_mask+test_mask]
#         print("xx3shape: ", end=" ")
#         print(xx3.shape)
#
#
#         # xx = xx / LA.norm(xx, axis=1)
#         yy3 = y_test[train_mask+val_mask+test_mask]
#         print("yy3shape: ", end=" ")
#         print(yy3.shape)
#         # print(yy)
#         # print(yy[0])
#
#         print('xx3: {}, yy3: {}'.format(xx3.shape, yy3.shape))
#
#         sess.close()
# print("HAN3结束了")
#
#










XX=tf.stack([xx1,xx1,xx1],axis=1)
YY=tf.stack([yy1,yy1,yy1],axis=1)

label_new=np.full((2614,1),0)

het_walk_f = open("data/yelp/" + "business_category.txt", "r")
for line in het_walk_f:
    line = line.strip()
    node_id = str(re.split("\t", line)[0])
    neigh_list = str(re.split("\t", line)[1])
    label_new[int(node_id)][0] = int(neigh_list)

y_new =label_new


# 将张量转换为数组并变形为二维数组
with tf.Session() as sess:
    x_train_val = sess.run(XX)
x_train_array = np.array(x_train_val)



# 假设X和Y已经存在，且长度相等
# 打乱X和Y的顺序
idx = np.arange(x_train_array.shape[0])
np.random.shuffle(idx)
X_shuffled = x_train_array[idx]
Y_shuffled = y_new[idx]

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_shuffled, Y_shuffled, test_size=0.2, random_state=42)




Y_test2 = np.squeeze(Y_test).T



def scheduler(epoch):
    # 每隔50个epoch，学习率减小为原来的1/10
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(tcn.optimizer.lr)
        if lr>1e-5:
            K.set_value(tcn.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
    return K.get_value(tcn.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)










early_stopping = EarlyStopping(monitor='loss',
                               patience=20,
                               min_delta=1e-5,
                               mode='auto',
                               restore_best_weights=False,
                               verbose=2)
                                #r是否从具有监测数量的最佳值的时期恢复模型权重
batch_size=None

timesteps=X_train.shape[1]
input_dim=X_train.shape[2] #输入维数



tcn = Sequential()
input_layer =Input(batch_shape=(batch_size,timesteps,input_dim))

tcn.add(input_layer)
tcn.add(TCN(nb_filters=64, #在卷积层中使用的过滤器数。可以是列表。
        kernel_size=3, #在每个卷积层中使用的内核大小。
        nb_stacks=1,   #要使用的残差块的堆栈数。
        dilations=[2 ** i for i in range(6)], #扩张列表。示例为：[1、2、4、8、16、32、64]。
        #用于卷积层中的填充,值为'causal' 或'same'。
        #“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。
        #“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
        padding='causal',
        use_skip_connections=True, #是否要添加从输入到每个残差块的跳过连接。
        dropout_rate=0.1, #在0到1之间浮动。要下降的输入单位的分数。
        return_sequences=False,#是返回输出序列中的最后一个输出还是完整序列。
        activation='relu', #残差块中使用的激活函数 o = Activation(x + F(x)).
        kernel_initializer='he_normal', #内核权重矩阵（Conv1D）的初始化程序。
        use_batch_norm=True, #是否在残差层中使用批处理规范化。
        use_layer_norm=True, #是否在残差层中使用层归一化。
        name='tcn' #使用多个TCN时，要使用唯一的名称
        ))
tcn.add(tf.keras.layers.Dense(64))
tcn.add(tf.keras.layers.LeakyReLU(alpha=0.3))
tcn.add(tf.keras.layers.Dense(32))
tcn.add(tf.keras.layers.LeakyReLU(alpha=0.3))
tcn.add(tf.keras.layers.Dense(16))
tcn.add(tf.keras.layers.LeakyReLU(alpha=0.3))
# 替换最后一层的激活函数
tcn.add(tf.keras.layers.Dense(3, activation='softmax'))



# tcn.compile(optimizer='adam', loss='mse', metrics=[macro_f1_score,MicroF1Score()])
tcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tcn.summary()


history=tcn.fit(X_train,Y_train, epochs=80,batch_size=16,callbacks=[reduce_lr],steps_per_epoch=1)




predict1 = tcn.predict(X_test,steps=1)


# 使用argmax函数找到每一行中最大值所在的列索引
max_indices = np.argmax(predict1, axis=1)

# 将列索引转化为0、1、2
converted_indices = np.zeros_like(max_indices)
for i in range(len(max_indices)):
    converted_indices[i] = max_indices[i]

# 打印结果
# print(converted_indices)


macro_f1 = f1_score(Y_test2, converted_indices, average='macro')
micro_f1 = f1_score(Y_test2, converted_indices, average='micro')
print("Macro-F1:", macro_f1)
print("Micro-F1:", micro_f1)
