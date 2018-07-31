import numpy as np
import tensorflow as tf
import matplotlib as plt
from TF import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
print('data loaded')

# 神经网络结构
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784
num_classes = 10

# Input and Out
x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, num_classes])

# 神经网络参数
stddev = 0.1
weights = {
    'w1': tf.Variable(tf.random_normal([num_input, num_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([num_hidden_2, num_classes], stddev=stddev))
}

biases = {
    'b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
print('network ready')

# 定义前向传播
def multilayer(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])

# 定义反向传播
pred  = multilayer(x, weights, biases)

# 损失函数为softmax的交叉熵函数
# pred为输入值，即一次前向传播的结果，与实际的label y值进行比较
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(correct, "float"))

init = tf.global_variables_initializer()
print("function ready")

training_epoachs = 20
batch_size = 100
display_step = 4

sess = tf.Session()
sess.run(init)
for epoach in range(training_epoachs):
    avg_cost = 0
    all_batch = int(mnist.train.num_examples / batch_size)
    for i in range(all_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optimizer, feed_dict=feeds)
    avg_cost += sess.run(cost, feed_dict=feeds)
    if (epoach+1) % display_step == 0:
        print("Epoach: %03d/%03d cost: %.9f" % (epoach, training_epoachs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_accr = sess.run(accr, feed_dict=feeds)
        print("Train accr: %.3f" % (train_accr))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_accr = sess.run(accr, feed_dict=feeds)
        print("Test accr: %.3f" % (test_accr))
print('already done')

