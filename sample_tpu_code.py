#import libraries
import math, re, os
from statistics import mode
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import h5py

print("Tensorflow version " + tf.__version__)


# detect TPUs
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

print("Number of accelerators: ", tpu_strategy.num_replicas_in_sync)


## Load the data matrix

with h5py.File('trX.h5', 'r') as hf:
    trX = hf['trX'][:]

with h5py.File('trY.h5', 'r') as hf:
    trY = hf['trY'][:]

with h5py.File('vaX.h5', 'r') as hf:
    vaX = hf['vaX'][:]

with h5py.File('vaY.h5', 'r') as hf:
    vaY = hf['vaY'][:]

print(trX.shape, trY.shape, vaX.shape, vaY.shape)
    
seq_len = 110
input_shape = (seq_len, 4)
lr = 1e-3
batch_size = 1024
epochs = 500
model_path = 'sample.h5'

with tpu_strategy.scope():
    
    #take input
    inputs = tf.keras.Input(shape=[seq_len, 4,])

    ### add layers

    
    # final prediciton layer
    pred = tf.keras.layers.Dense(1)(flatten)

    # define model 
    model = tf.keras.Model(inputs=[inputs], outputs= [pred])

    opt = tf.keras.optimizers.Adam(lr) #tf.keras.optimizers.Adam(lr=lr)#
    model.compile(optimizer=opt, loss='mean_squared_error',metrics=[r_square])
    model.summary()

train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
val_data = tf.data.Dataset.from_tensor_slices((vaX, vaY))

train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

csvlogger = tf.keras.callbacks.CSVLogger("loss_history.tsv", separator='\t', append=False)
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_r_square', 
                                             verbose=1, save_best_only=True, mode='max')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_r_square', patience=10, mode='max') 
callbacks_list = [checkpoint, early_stop, csvlogger]

model.fit(train_data, validation_data = val_data,
          batch_size=batch_size  , epochs=epochs , callbacks=callbacks_list, steps_per_epoch = len(train_data),
          validation_steps = len(val_data),validation_batch_size = batch_size)
