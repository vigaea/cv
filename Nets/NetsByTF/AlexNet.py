from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


# print every layer tensor's size
def print_tensor(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope:
        # initialize parameters, kernel=11x11,num=64
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_tensor(conv1)

    # maxpooling
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_tensor(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        # kernal=5x5, input=64,kernel's num=192
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192]), dtype=tf.float32,
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(conv, name=scope)
        parameters += [kernel, biases]
        print_tensor(conv2)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_tensor(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32,
                                                 stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384]), dtype=tf.float32,
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(conv, name=scope)
        parameters += [kernel, biases]
        print_tensor(conv3)

    # conv4
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                                                 stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]), dtype=tf.float32,
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(conv, name=scope)
        parameters += [kernel, biases]
        print_tensor(conv4)

    # conv5
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256]), dtype=tf.float32,
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(conv, name=scope)
        parameters += [kernel, biases]
        print_tensor(conv5)

    pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    print_tensor(pool3)
    return pool3, parameters


def runtime_tf(session, target, info_string):
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
        pool3, parameters = inference(imgs)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        runtime_tf(sess, pool3, "Forward")
        object = tf.nn.l2_loss(pool3)
        grad = tf.gradients(object, parameters)
        runtime_tf(sess, grad, "Backward")


if __name__ == '__main__':
    run_benchmark()
