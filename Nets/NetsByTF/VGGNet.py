# VGG16

from datetime import datetime
import time
import math
import tensorflow as tf

batch_size = 32
num_batches = 100


# kh,kw=kernel's height,width  dh,dw=stride's height,width
def conv_op(input_op, name, kh, kw, n_out, dh, dw, params):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, [1, dh, dw, 1], padding='SAME')
        bias_inital = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_inital, trainable='True', name='bias')
        result = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(result, scope)
        params += [kernel, biases]
        return activation


# fully connected
def fc_op(input_op, name, n_out, params):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.constant(0.1, shape=[n_out]), dtype=tf.float32)
        activation = tf.nn.relu_layer(input_op, kernel, bias, name=scope)
        params += [kernel, bias]


# max pool
def maxpool(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1],
                          padding='SAME', name=name)


# build net
# keep_prob = placeholder(ratio of dropout)
def inference(input_op, keep_prob):
    params = []
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, params=params)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, params=params)
    pool1 = maxpool(conv1_2, name='pool1', kh=2, kw=2, dh=2, dw=2)

    conv2_1 = conv_op(pool1, name='conv1_1', kh=3, kw=3, n_out=128, dh=1, dw=1, params=params)
    conv2_2 = conv_op(conv2_1, name='conv1_2', kh=3, kw=3, n_out=128, dh=1, dw=1, params=params)
    pool2 = maxpool(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, params=params)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, params=params)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, params=params)
    pool3 = maxpool(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, params=params)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, params=params)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, params=params)
    pool4 = maxpool(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)

    # conv5's out=7x7x512
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, params=params)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, params=params)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, params=params)
    pool5 = maxpool(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)

    # flat conv5's out   7x7x512 --> 1x25088
    shape = pool5.get_shape()
    flatten = shape[1].value * shape[2].value * shape[3].value
    resh1 = tf.reshape(pool5, [-1, flatten], name='resh1')

    fc6 = fc_op(resh1, name='fc6', n_out=4096, params=params)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, params=params)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, params=params)
    softmax = tf.nn.softmax(fc8)
    prediction = tf.argmax(softmax, 1)
    return prediction, softmax, fc8, params


def runtime_tf(session, target, feed, info_string):
    numstep_burnin = 10
    total_duration = 0.0
    total_dyration_squared = 0.0
    for i in range(num_batches + numstep_burnin):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i > numstep_burnin:
            if not i % 10:
                print('%s:step %d, duration=%.3f' % (datetime.now(), i - numstep_burnin, duration))
            total_duration += duration
            total_dyration_squared = duration * duration
    mn = total_duration / num_batches
    sd = total_dyration_squared / num_batches - mn * mn
    sd = math.sqrt(sd)
    print('%s:%s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        img_size = 224
        imgs = tf.Variable(tf.random_normal([batch_size, img_size, img_size, 3],
                                            dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        prediction, softmax, fc8, params = inference(imgs, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        runtime_tf(sess, prediction, {keep_prob: 1.0}, "Forward")
        object = tf.nn.l2_loss(fc8)
        grad = tf.gradients(object, params)
        runtime_tf(sess, grad, {keep_prob: 0.5}, "Backward")


if __name__ == '__main__':
    run_benchmark()
