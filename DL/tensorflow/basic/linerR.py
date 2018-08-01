import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_p = 1000
vectors = []
for p in range(num_p):
    xs = np.random.normal(0.0, 0.55)
    ys = 0.1 * xs + 0.3 + np.random.normal(0.0, 0.05)
    vectors.append([xs, ys])

xs_data = [v[0] for v in vectors]
ys_data = [v[1] for v in vectors]

plt.scatter(xs_data, ys_data, c='r')
plt.show()


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')  # 生成一维矩阵，取值为（-1，1）的随机值
b = tf.Variable(tf.zeros([1]), name='b')
y = W * xs_data + b

loss = tf.reduce_mean(tf.square(y - ys_data), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print("W=",sess.run(W), "b=",sess.run(b), "loss=",sess.run(loss))
for i in range(20):
    sess.run(optimizer)
    print("W=",sess.run(W), "b=",sess.run(b), "loss=",sess.run(loss))