import tensorflow as tf

x = tf.placeholder([None, 5], tf.float32)
y = tf.placeholder([None, 5], tf.float32)

W = tf.Variable([.3], tf.float32)
b = tf.Variable([.3], tf.float32)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	