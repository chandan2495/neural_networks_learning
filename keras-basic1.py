from keras.models import Sequential
from keras.layers import Dense, Activation

#training data
x_train = [1,2,3,4,5]
y_train = [2,4,6,8,10]

#testing data 
x_test = [6,7,8,9,10]
y_test = [12,14,16,18,20]

model = Sequential()

model.add(Dense(units=1, input_dim=1))
# model.add(Dense(32, input_shape=(16,)))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)
# after the first layer, you don't need to specify the size of the input anymore:
# Dense implements the operation: output = activation(dot(input, kernel) + bias) 
# where activation is the element-wise activation function passed as the activation argument, 
# kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
# model.add(Activation('tanh'))
# model.add(Dense(1))
# model.add(Dense(1))
# model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150)

loss_and_metrics = model.evaluate(x_test, y_test)

classes = model.predict(x_test)
print(classes)