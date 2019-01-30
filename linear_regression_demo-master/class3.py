import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

training_set, testting_set, validation_set = imdb.load_data(path = "imdb.pkl", n_words = 10000)

train_X, train_Y = training_set
test_X, test_Y = testting_set


#preprocessing 

train_X = pad_sequences(training_set, maxlen = 100)
test_X = pad_sequences(training_set, maxlen = 100)

train_Y = to_categorical(train_Y, nb_classes = 2) 
test_Y = to_categorical(test_Y, nb_classes = 2)

net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim = 10000, output_dim = 128)
net = tflearn.lstm(net, 128, dropout = 0.8)
net = tflearn.fully_connected(net, 2, activation="softmax")
net = tflearn.regression(net, optimizer = "adam", learning_rate = 0.0001, loss = "categorical_crossentropy")

model = tflearn.DNN(net, tensorboard_verbose = 0)
model.fit(train_X, train_Y, validation_set = (test_X, test_Y), show_metric=True, batch_size = 32) 