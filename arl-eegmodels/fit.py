import os
from datetime import datetime

import tensorflow as tf
# enable syntax highlighting
keras = tf.keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from EEGModels import EEGNet
from utils import get_data

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

model = EEGNet

train_ds, val_ds, test_ds, class_weights, max_len = get_data(batch_size=64)

channels = 1
samples = 3000

model = EEGNet(nb_classes = 5, Chans = channels, Samples = samples, 
               dropoutRate = 0.5, kernLength = 64, F1 = 64, D = 1, F2 = 64, 
               dropoutType = 'Dropout')

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# Make directories required
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
experiment_name = "initial2"
os.makedirs(f"models/eegNet/{experiment_name}", exist_ok=True)

logdir = f"logs/eegNet/{experiment_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

callbacks = [
    ModelCheckpoint(
        filepath="models/eegNet/initial2/version-n{epoch:02d}.h5",
        save_best_only=True),
    TensorBoard(
        log_dir=logdir,
        histogram_freq=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        start_from_epoch=50
    )
]

fittedModel = model.fit(train_ds, epochs = 150, 
                        verbose = 2, validation_data=val_ds,
                        callbacks=callbacks, class_weight = class_weights)

model.save("models/eegNet/initial2/trained_model.h5")