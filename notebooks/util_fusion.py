
import os, ast, json, pickle, math
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,GRU
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
import re
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                      MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D,Rescaling)
from lime.lime_text import LimeTextExplainer
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.preprocessing import RobustScaler

from util_fusion import (labels_to_vector,preprocess_text,build_joint_fusion,
                         build_joint_dataset, optimize_thresholds, plot_fusion_curves,
                         plot_confusion_matrices, load_and_preprocess, build_dataset )




def labels_to_vector(label_list):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label in label_list:
        if label in ALL_LABELS:
            vec[ALL_LABELS.index(label)] = 1.0
    return vec

def preprocess_text(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    tokens = caption.split()
    tokens = [t for t in tokens if t not in stop_words]    
    return ' '.join(tokens)


def build_joint_fusion(image_encoder_weights_model,embedding_matrix, num_classes=18):
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

    base = EfficientNetB0(
        weights=None,           
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.set_weights(
        image_encoder_weights_model.get_layer("efficientnetb0").get_weights()
    )
    base.trainable = False  

    x_img = base(input_img, training=False)
    x_img = layers.Dense(128, activation="relu", name="img_proj")(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    x_img = x_img 

    input_text = Input(shape=(MAX_LEN,), name="text", dtype="int32")
    x_txt = layers.Embedding(MAX_WORDS, EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              trainable=True)(input_text)
    x_txt = layers.SpatialDropout1D(0.2)(x_txt)
    x_txt = layers.GRU(128, name="gru_joint")(x_txt)
    x_txt = layers.Dense(128, activation="relu", name="txt_proj")(x_txt)
    x_txt = layers.BatchNormalization()(x_txt)
    x_txt = layers.Dropout(0.3)(x_txt)

    fused  = layers.Concatenate(name="fusion")([x_img, x_txt])
    x      = layers.Dense(128, activation="relu")(fused)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs=[input_img, input_text], outputs=output,
                  name="joint_fusion")
    return model, base


def build_joint_dataset(df, X_text, Y, augment=False, shuffle=False):
    img_paths = [os.path.join(data_dir, fname) for fname in df["ImageID"]]

    def load_sample(img_path, text_seq, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return {"image": img, "text": text_seq}, label

    ds = tf.data.Dataset.from_tensor_slices((
        img_paths,
        X_text.astype("int32"),
        Y.astype("float32")
    ))
    if shuffle:
        ds = ds.shuffle(5000, seed=42)
    ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def optimize_thresholds(y_true, y_proba, label_names):
    thresholds = np.zeros(NUM_CLASSES)
    print("=== Optimisation des seuils par classe ===\n")
    for i in range(NUM_CLASSES):
        best_thresh = 0.3
        best_f1     = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba[:, i] >= thresh).astype(int)
            f1     = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = thresh
        thresholds[i] = best_thresh
        print(f"{label_names[i]:25s} → seuil={best_thresh:.2f}  F1={best_f1:.3f}")
    return thresholds  


def plot_fusion_curves(h_early, h_joint_p1, h_joint_p2):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0][0].plot(h_early.history["auc_pr"],     label="Train", color="#5886B5")
    axes[0][0].plot(h_early.history["val_auc_pr"], label="Val",   color="#D87340")
    axes[0][0].set_title("Early Fusion — AUC-PR", fontweight="bold")
    axes[0][0].set_xlabel("Époque")
    axes[0][0].set_ylabel("AUC-PR")
    axes[0][0].legend()

    axes[1][0].plot(h_early.history["loss"],     label="Train", color="#5886B5")
    axes[1][0].plot(h_early.history["val_loss"], label="Val",   color="#D87340")
    axes[1][0].set_title("Early Fusion — Loss", fontweight="bold")
    axes[1][0].set_xlabel("Époque")
    axes[1][0].set_ylabel("Loss")
    axes[1][0].legend()

    auc_joint     = h_joint_p1.history["auc_pr"]     + h_joint_p2.history["auc_pr"]
    val_auc_joint = h_joint_p1.history["val_auc_pr"] + h_joint_p2.history["val_auc_pr"]
    p1_end        = len(h_joint_p1.history["auc_pr"])

    axes[0][1].plot(auc_joint,     label="Train", color="#5886B5")
    axes[0][1].plot(val_auc_joint, label="Val",   color="#D87340")
    axes[0][1].axvline(p1_end, color="gray", linestyle="--", alpha=0.6, label="Début phase 2")
    axes[0][1].set_title("Joint Fusion — AUC-PR (P1 + P2)", fontweight="bold")
    axes[0][1].set_xlabel("Époque")
    axes[0][1].set_ylabel("AUC-PR")
    axes[0][1].legend()

    loss_joint     = h_joint_p1.history["loss"]     + h_joint_p2.history["loss"]
    val_loss_joint = h_joint_p1.history["val_loss"] + h_joint_p2.history["val_loss"]

    axes[1][1].plot(loss_joint,     label="Train", color="#5886B5")
    axes[1][1].plot(val_loss_joint, label="Val",   color="#D87340")
    axes[1][1].axvline(p1_end, color="gray", linestyle="--", alpha=0.6, label="Début phase 2")
    axes[1][1].set_title("Joint Fusion — Loss (P1 + P2)", fontweight="bold")
    axes[1][1].set_xlabel("Époque")
    axes[1][1].set_ylabel("Loss")
    axes[1][1].legend()

    plt.tight_layout()
    plt.savefig("training_curves_fusion.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(y_true, y_pred_proba, threshold=THRESHOLD):
    y_pred = (y_pred_proba >= threshold).astype(int)
    cms    = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    axes = axes.flatten()

    for i, (cm, name) in enumerate(zip(cms, label_names)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    ax=axes[i], cbar=False)
        axes[i].set_title(name, fontsize=9)

    plt.tight_layout()
    plt.show()



def load_and_preprocess(img_path, label, augment=False):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    img = tf.cast(img, tf.float32)

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 255.0)

    return img, label


def build_dataset(dataframe, img_dir, augment=False, shuffle=False):
    """Construit un tf.data.Dataset à partir d'un DataFrame."""
    img_paths = [os.path.join(img_dir, fname) for fname in dataframe["ImageID"]]
    labels    = np.stack([labels_to_vector(l) for l in dataframe["Labels"]])

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=42)

    ds = ds.map(
        lambda path, label: load_and_preprocess(path, label, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds