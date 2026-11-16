"""
Train a starter plant-leaf classifier from a PlantVillage-style folder dataset.

Expected folder layout:
dataset/
  Apple___Apple_scab/
    image1.jpg
  Apple___Black_rot/
    image2.jpg
  ...
Run: python train_model.py
"""
import json
import tensorflow as tf
from pathlib import Path

DATA_DIR = Path("dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

if not DATA_DIR.exists():
    raise SystemExit("Create a 'dataset' folder and put one subfolder per class inside it.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
Path("class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("plant_disease_model.keras", save_best_only=True)
]

model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks)
model.save("plant_disease_model.keras")
print("Finished. Saved plant_disease_model.keras and class_names.json")
