import argparse, os, json, itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
# Evaluate on validation split after training
import numpy as np
from sklearn.metrics import classification_report

def eval_on_dataset(model, ds, class_names):
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(p, axis=1).tolist())
    print(classification_report(y_true, y_pred, target_names=class_names))

def build_datasets(data_dir, img_size=224, batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    def _prep(x, y):
        return preprocess_input(tf.cast(x, tf.float32)), y

    train_ds = train_ds.map(_prep).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(_prep).prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model(num_classes, img_size=224, base_trainable=False):
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=(img_size, img_size, 3))
    base.trainable = base_trainable
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]  # Sparse-safe; maps to SparseCategoricalAccuracy
)

    return model

def plot_history(h, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for metric in ["loss","accuracy"]:
        plt.figure()
        plt.plot(h.history[metric], label="train")
        plt.plot(h.history["val_"+metric], label="val")
        plt.legend(); plt.title(metric)
        plt.savefig(os.path.join(out_dir, f"{metric}.png"), bbox_inches="tight")
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models/landcover_mnv2.h5")
    args = ap.parse_args()

    train_ds, val_ds, class_names = build_datasets(args.data_dir, args.img_size, args.batch)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(class_names, open("models/labels.json","w"))

    model = build_model(len(class_names), args.img_size)
    ckpt  = ModelCheckpoint(args.out, save_best_only=True, monitor="val_accuracy", verbose=1)
    early = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    rlr   = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=[ckpt, early, rlr])
    plot_history(hist, "models/training_plots")

    print(f"✅ Saved model → {args.out}")

if __name__ == "__main__":
    main()
