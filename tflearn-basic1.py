import tflearn
import tensorflow as tf
import numpy as np

#training data
x_train = [1,2,3,4,5]
y_train = [2,4,6,8,10]

#testing data 
x_test = [6,7,8,9,10]
y_test = [12,14,16,18,20]

tf.reset_default_graph()

# build neural network
net = tflearn.input_data(shape=[None])
net = tflearn.single_unit(net)
# net = tflearn.fully_connected(net,1, activation='softmax')
net = tflearn.regression(net, loss='mean_square', optimizer='adam', learning_rate=0.01)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


model.fit(x_train, y_train, n_epoch=1000, show_metric=True)

classes = model.predict(x_test)
print(classes)