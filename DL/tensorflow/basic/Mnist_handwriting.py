import matplotlib.pyplot as plt
import tensorflow as tf
from TF import input_data

learning_rate = 1e-4
training_iteration = 2500
drop_out = 0.5
batch_size = 50
validation_size = 2000

mnist = input_data.read_data_sets('data', one_hot=True)


# w、b的初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # w进行高斯初始化,shape维度大小
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # b进行常量初始化
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # ksize窗口大小，strides滑动步长


# 输入、输出数据
x = tf.placeholder('float', [None, 784])  # batch=None,实际数据使用时再指定batch大小
y_ = tf.placeholder('float', [None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])  # 5x5窗口,w左连像素图，右连L1层，1、32则代表w所连接的深度(channel数)
b_conv1 = bias_variable([32])  # 所以28x28x1的灰度图，在L1生成32个特征图，w1....w32. 都进行滑动窗口的特征提取操作

image = tf.reshape(x, [-1, 28, 28, 1])  # batchsize=-1,因为要根据实际推断出batchsize大小

# first
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool2x2(h_conv1)  # (,28,28,32) 压缩--> (,14,14,32)

# second
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool2x2(h_conv2)  # (,14,14,64) 压缩--> (,7,7,64)

# fully connected 1 将特征图转换成向量的格式
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # (,7,7,64) ==> (,3136)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fully connected 2  /softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 交叉熵
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # equal 判断预测值是否等于实际值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "double"))

train_accuracies = []
test_accuracies = []
x_range = []

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5})
        print("step %d, training accuracy-->%.4f, test_accuracy-->%.4f" %
              (i, train_accuracy, test_accuracy))
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        x_range.append(i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

plt.plot(x_range, train_accuracies, '-b', label='Training')
plt.plot(x_range, test_accuracies, '-g', label='Test')
plt.legend(loc='best')  # frameon=False无边框
plt.ylim(ymax=1.1, ymin=0.7)
plt.ylabel('Accuracy')
plt.yticks(rotation=90)
plt.xlabel('Step')
plt.show()
