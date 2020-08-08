# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:50:49 2019
learning route: 
    A: 1.0 -> 1.1 -> 1.2 -> 1.3 -> 2.0 -> 2.1 ->... -> 2.5
    B: [1.0->..>1.3] 3.1-> 2.2 ->...-> 2.5
    C: [1.0->..>1.3] 3.2-> 2.2 ->...-> 2.5
    D: [1.0->..>1.3] 3.3-> 2.2 ->...-> 2.5
@author: hao
"""

import keras
keras.__version__

# In[1.0]: retreieve source data
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# In[1.1]: decode
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# In[1.2]: data preprocess
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# In[1.3]: get samples
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# In[2.1] build a model structure
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# In[2.2] specify the model paramters 
from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

# In[2.3] derive the model from train data with validation
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# In[2.4]
import matplotlib.pyplot as plt

history_dict = history.history
acc = history.history['binary_accuracy']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# In[2.5]
results = model.evaluate(x_test, y_test)
results

# In[3.1]: A. Reduce network size from 16 to 4
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# In[3.2]: B. Add regularizer l2 to kernels

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer =regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer =regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# In[3.3]: C. Add dropout 

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
