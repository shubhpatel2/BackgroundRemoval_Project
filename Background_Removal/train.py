import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.datasets import BackgroundRemovalDataset
from Models.Unet import unet_model  

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Parameters
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 5
EPOCHS = 5
MODEL_PATH = 'unet_best_model.keras'  

# 1. Load Dataset
dataset = BackgroundRemovalDataset(
    image_dir='Data/DUTS-TE-Image',
    mask_dir='Data/DUTS-TE-Mask', 
    image_size=IMAGE_SIZE
)
X_train, X_val, y_train, y_val = dataset.get_splits(test_size=0.2)

# 2. Create TensorFlow Dataset
def data_generator():
    for x, y in zip(X_train, y_train):
        yield x, y

train_dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=X_train.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=y_train.shape[1:], dtype=tf.float32),
    )
).batch(BATCH_SIZE).shuffle(100)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

# 3. Build Model
model = unet_model(input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))  

model.summary()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.MeanIoU(num_classes=2, name='iou')
    ]
)

# 4. Callbacks
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
]

# 5. Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n Training complete. Best model saved to {MODEL_PATH}")
