import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

learning_rate = 1e-4
training_iteration = 2500
drop_out = 0.5
batch_size = 50

validation_size = 2000
image_display = 10

data = pd.read_csv(r'D:/DATA/ML/mnist_train.csv')
print(data.shape)
print('data({0[0]},{0[1]})'.format(data.shape))
# print(data.head())


# 归一化 0~255 --> 0~1
images = data.iloc[:, 1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0/255.0)
print(images.shape)
print('images({0[0]},{0[1]})'.format(images.shape))

# 图片的大小、长和高
image_size = images.shape[1]
image_height = image_width = np.sqrt(image_size).astype(np.int)
print(image_size, image_height, image_width)

def display(img):
    img1 = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(img1, cmap=cm.binary)
# display(images[image_display])

labels_num = data.ix[:,0].ravel()
labels_count = np.unique(labels_num)
print('labels_num({0})'.format(len(labels_num)))
print(labels_count.size)


# one-hot编码   0 ==> [1 0 0 0 0 0 0 0 0 0 0]
def dense2_onehot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_all = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_all + labels_dense] = 1
    return labels_onehot

labels = dense2_onehot(labels_num, labels_count)
labels = labels.astype(np.int)
print('labels[{0}] ==> {1}'.format(image_display,labels[image_display]))

# validation
validation_images = images[:validation_size]
validation_labels = images[:validation_size]
train_images = images[validation_size:]
train_labels = images[validation_size:]

# w、b的初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # w进行高斯初始化,shape维度大小
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # b进行常量初始化
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  #ksize窗口大小，strides滑动步长

# 输入、输出数据
x = tf.placeholder('float', [None, 784])  # batch=None,实际数据使用时再指定batch大小
y_ = tf.placeholder('float', [None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])  # 5x5窗口,w左连像素图，右连L1层，1、32则代表w所连接的深度(channel数)
b_conv1 = bias_variable([32])             # 所以28x28x1的灰度图，在L1生成32个特征图，w1....w32. 都进行滑动窗口的特征提取操作

image = tf.reshape(x, [-1, 28, 28, 1])  # batchsize=-1,因为要根据实际推断出batchsize大小

# first
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool2x2(h_conv1)  # (,28,28,32) 压缩--> (,14,14,32)

# second
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool2x2(h_conv2)  #(,14,14,64) 压缩--> (,7,7,64)

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


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  # 交叉熵
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  # equal 判断预测值是否等于实际值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "double"))
# predict = tf.argmax(y, 1)


epochs_completed = 0
index_epoch = 0
num_exp = train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global index_epoch
    global epochs_completed

    start = index_epoch
    index_epoch += batch_size

    # 当训练数据用完时，随机重新排序
    if index_epoch > num_exp:
        epochs_completed += 1
        perm = np.arange(num_exp)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        start = 0
        index_epoch = batch_size
        assert batch_size <= num_exp
    end = index_epoch
    return train_images[start:end],train_labels[start:end]

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

train_accuracies = []
validation_accuracices = []
x_range = []
display_step = 1

for i in range(training_iteration):
    batch_xs, batch_ys = next_batch(batch_size)
    if i % display_step== 0 or (i+1) == training_iteration:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
        if(validation_size):
            validation_accuracy = accuracy.eval(feed_dict={
                x: validation_images[0:batch_size],
                y_: validation_labels[0:batch_size],
                keep_prob : 1.0
            })
            print('training_accuracy / validation_accuracy --> %.2f / %.2f for step %d' %
                  (train_accuracy, validation_accuracy, i))
            validation_accuracices.append(validation_accuracy)
        else:
            print('training_accuracy --> %.4f for step %d' % (train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)

        if i % (display_step*10) == 0 and i:
            display_step *= 10
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, keep_prob:drop_out})

if(validation_size):
    validation_accuracy = accuracy.eval(feed_dict={
        x: validation_images, y_: validation_labels, keep_prob: 1.0
    })
    print('validation_accuracy --> %.4f' % validation_accuracy)
    plt.plot(x_range, train_accuracies, '-b', label='Training')
    plt.plot(x_range, validation_accuracices, '-g', label='Validation')
    plt.legend(loc='best', frameon=False)
    plt.ylim(ymax=1.1, ymin=0.7)
    plt.ylabel('Accuracy')
    plt.xlabel('Step')
    plt.title('Accuracy of handwriting by CNN')
    plt.show()