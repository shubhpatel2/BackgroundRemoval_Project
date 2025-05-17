import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
import segmentation_models as sm  # NEW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils1.datasets import BackgroundRemovalDataset

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 10
EPOCHS = 10
MODEL_PATH = 'pretrained_unet_best_model.keras'

# Set segmentation_models framework
sm.set_framework('tf.keras')
sm.framework()

# Pretrained backbone settings
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# 1. Load Dataset
dataset = BackgroundRemovalDataset(
    image_dir='Data/DUTS-TE-Image',
    mask_dir='Data/DUTS-TE-Mask',
    image_size=IMAGE_SIZE
)

X_train, X_val, y_train, y_val = dataset.get_splits(test_size=0.2)

# Preprocess images using encoder's preprocessing function
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# 2. TensorFlow Dataset
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

# 3. Build Pretrained U-Net
model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    encoder_weights='imagenet',  # Use pretrained weights
    classes=1,
    activation='sigmoid'
)

model.compile(
    optimizer='adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score, 'accuracy']
)

model.summary()

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

print(f"\nTraining complete. Best model saved to {MODEL_PATH}")
