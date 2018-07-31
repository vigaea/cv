import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from TF import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print('MNIST already loaded')
print(trainimg.shape)
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)

# 逻辑回归采用softmax
x = tf.placeholder("float", [None, 784])  # None代表值的数量未知、无穷
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784, 10]))      # W为784行向量，x则为784的列向量，10为10分类
b = tf.Variable(tf.zeros([10]))           # 为了方便使用zeros为0值初始化，一般为高斯初始化

# softmax的输入为 样本属于每个分类的得分值
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# loss=-logP，与y相乘后就会得到属于真实分类的得分值
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# prediction
# 对比 预测值的索引 与 label值的索引是否相同
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, "float"))
init = tf.global_variables_initializer()


training_epochs = 50
batch_size = 100
display_step = 5

sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost +=  sess.run(loss, feed_dict=feeds) / num_batch
    #Display
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y:batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accuracy, feed_dict = feeds_train)
        test_acc = sess.run(accuracy, feed_dict = feeds_test )
        print("Epoach: %03d/%03d, cost: %.9f, train_acc: %.3f, test_acc: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))

print("Already done")