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

train_ds, val_ds, test_ds, class_weights, max_len = get_data(dataset='motor_imagery', batch_size=128)

channels = 2
samples = 640

model = EEGNet(nb_classes = 1, Chans = channels, Samples = samples, 
               dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

model.summary()

opt = keras.optimizers.legacy.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# Make directories required
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
experiment_name = "motor_imagery"
split_name="split7"
os.makedirs(f"models/eegNet/{experiment_name}/{split_name}/{split_name}", exist_ok=True)

logdir = f"logs/eegNet/{experiment_name}/{split_name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

callbacks = [
    ModelCheckpoint(
        filepath="models/eegNet/motor_imagery/split7/version-n{epoch:02d}.h5",
        save_best_only=True),
    TensorBoard(
        log_dir=logdir,
        histogram_freq=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=50,
        start_from_epoch=150
    )
]

fittedModel = model.fit(train_ds, epochs = 300, 
                        verbose = 2, validation_data=val_ds,
                        callbacks=callbacks, class_weight = class_weights)

model.save("models/eegNet/motor_imagery/split7/trained_model.h5")