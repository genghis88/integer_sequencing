import pandas as pd
import sys

from sqlalchemy.sql.functions import concat

trainingData = pd.read_csv('data/train.csv')

print(trainingData)

minLen = 5
min = sys.maxsize
max = 0

for index, row in trainingData.iterrows():
  nums = [int(x) for x in str(row['Sequence']).split(',')]
  if min > nums.__len__():
    min = nums.__len__()
  if max < nums.__len__():
    max = nums.__len__()

y = {}
xTrain = {}
for index, row in trainingData.iterrows():
  nums = [int(x) for x in str(row['Sequence']).split(',')]
  if nums.__len__() > 1:
    xTrain[row['Id']] = nums[:-1]
    y[row['Id']] = nums[nums.__len__() - 1]

y = pd.DataFrame.from_dict(y, orient='index')
xTrain = pd.DataFrame.from_dict(xTrain, orient='index', dtype='float')

xTrain.fillna(value=0, inplace=True)

print('Min ' + str(min) + ' Max ' + str(max))

print('Y ' + str(y))

print('xTrain ' + str(xTrain))

xTrain = xTrain.as_matrix()
y = y.as_matrix()

from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

xTrain = normalize(xTrain)
xTrain = scale(xTrain)

import random
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam, Adamax
from keras.regularizers import l2

xTrain = xTrain.reshape(xTrain.shape + (1,))

net = Sequential()
net.add(LSTM(128, input_shape=(347, 1), return_sequences=False, go_backwards=False, consume_less='gpu',
               W_regularizer=l2(0.005), U_regularizer=l2(0.005),
               inner_init='glorot_normal', init='glorot_normal', activation='tanh'))  # try using a GRU instead, for fun
net.add(Dense(1, activation='linear'))

# try using different optimizers and different optimizer configs
optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
net.compile(loss='mse', optimizer=optimizer)

#net.add(Dense(320, input_dim=xTrain.shape[1], init='glorot_uniform', activation='tanh'))
#net.add(Dropout(0.5))
#net.add(Dense(160, init='glorot_uniform', activation='tanh'))
#net.add(Dense(1, init='uniform', activation='linear'))

net.fit(xTrain, y, batch_size=256, nb_epoch=5)

y_pred = net.predict(xTrain)

y_pred = pd.DataFrame(y_pred)

y_pred.to_csv('preds/pred1.csv')