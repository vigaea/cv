import tensorflow as tf
a = 3
# 定义变量和操作
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)
print(y)

# 全局初始化、session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(y.eval())

norm = tf.random_normal([2,3], mean=-1, stddev=4)
c = tf.constant([[1,2],[3,4],[5,6]])
shuff = tf.random_shuffle(c)
with tf.Session() as sess:
    print(sess.run(norm))
    print(sess.run(shuff))

# placeholder相对于Variable，不会提前赋值，而是提前设置数据的规模与格式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
out = tf.mul(input1, input2)
with tf.Session() as sess:
    print(sess.run([out], feel_dict={input1:[7.], input2:[2.]}))
