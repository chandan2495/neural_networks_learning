import numpy as np
import tensorflow as tf

#Model parameters
W = tf.Variable([.3], tf.float32, name="weight")
b = tf.Variable([.3], tf.float32, name="bias")

#input and output
x = tf.placeholder(tf.float32, name="input_x")
y = tf.placeholder(tf.float32, name="actual_output")

#model
linear_model = W * x + b

#loss
loss = tf.reduce_sum(tf.square(linear_model-y))	# sum of squares
# reduce_sum docuementation
# https://www.tensorflow.org/api_docs/python/tf/reduce_sum
# Reduces input_tensor along the dimensions given in axis. Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
# If keep_dims is true, the reduced dimensions are retained with length 1.If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# tf.reduce_sum(x)  # 6
# tf.reduce_sum(x, 0)  # [2, 2, 2]
# tf.reduce_sum(x, 1)  # [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True)  # [[3], [3]]
# tf.reduce_sum(x, [0, 1])  # 6

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) # learning rate of 0.01
train = optimizer.minimize(loss)

#training_data
x_train = [1,2,3,4]
y_train = [2,4,6,8]

#training loop
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter('./logs', sess.graph)
	for i in range(1000):
		sess.run(train, {x:x_train, y:y_train})
	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
	print("W: %s b: %s loss: %.4f" % (curr_W, curr_b, curr_loss))
	# predict
	predicted_y = sess.run(W*[11] + b)
	print(predicted_y)
