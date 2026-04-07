import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img
import re
import math
import ast

from PIL import Image
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from sklearn.metrics import (multilabel_confusion_matrix,
                              classification_report,
                              f1_score, precision_score,
                              recall_score, hamming_loss)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                      MaxPooling2D, Flatten, Dense, Dropout,GlobalAveragePooling2D,Rescaling)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC,Precision,Recall

from tensorflow.keras import Input
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB5
from sklearn.metrics import accuracy_score, hamming_loss

from lime import lime_image
from PIL import Image
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries



def labels_to_vector(label_list):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label in label_list:
        if label in ALL_LABELS:
            vec[ALL_LABELS.index(label)] = 1.0
    return vec


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


def evaluate_model(model, dataset, y_true, threshold=0.5, split_name=""):

    y_pred_proba = model.predict(dataset)
    y_pred       = (y_pred_proba > threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"ÉVALUATION — {split_name} | seuil = {threshold}")
    print(f"{'='*60}")

    print(f"\nF1 macro     : {f1_score(y_true, y_pred, average='macro',    zero_division=0):.4f}")
    print(f"F1 micro     : {f1_score(y_true, y_pred, average='micro',    zero_division=0):.4f}")
    print(f"F1 weighted  : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 samples   : {f1_score(y_true, y_pred, average='samples',  zero_division=0):.4f}")
    print(f"Precision mac: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall macro : {recall_score(y_true, y_pred, average='macro',    zero_division=0):.4f}")

    print(f"\nRapport par classe :")
    report = classification_report(y_true, y_pred,
                                 target_names=[f"Label {l}" for l in ALL_LABELS],
                                #  output_dict=True,
                                 zero_division=0)

    print(report)
    return y_pred_proba, y_pred,report

def plot_confusion_matrices(y_true, y_pred, n_labels=6):

    cms  = multilabel_confusion_matrix(y_true, y_pred)
    ncols = 5
    nrows = math.ceil(n_labels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()


    for i in range(n_labels):
        tn, fp, fn, tp = cms[i].ravel()
        cm_display = np.array([[tn, fp],[fn,tp]])
        sns.heatmap(cms[i], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0",   "True 1"],
                    ax=axes[i])


        axes[i].set_title(f"Label {ALL_LABELS[i]}")

    for j in range(n_labels, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Matrices de confusion par classe", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Courbe de loss")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["auc"],     label="Train")
    axes[1].plot(history.history["val_auc"], label="Validation")
    axes[1].set_title("Courbe AUC")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_predictions(indices, val_ds, y_pred, threshold=0.5):
    all_images, all_labels = [], []
    for imgs, labs in val_ds:
        all_images.append(imgs.numpy())
        all_labels.append(labs.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    n    = len(indices)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for ax, idx in zip(axes, indices):
        img    = all_images[idx].astype(np.uint8)
        gt     = all_labels[idx]
        pred   = (y_pred[idx] >= threshold).astype(int)


        LABEL_IDS  = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]
        CLASS_NAMES = {
            1:"Personne", 2:"Vélo", 3:"Road", 4:"Moto", 5:"Avion",
            6:"Bus", 7:"Train", 8:"Truck", 9:"Bateau", 10:"Feux tricolore",
            11:"Bouche incendie", 13:"Feux signalisation", 14:"Parking meter",
            15:"Banc", 16:"Oiseau", 17:"Chat", 18:"Chien", 19:"Cheval"
        }

        gt_names   = [CLASS_NAMES[LABEL_IDS[i]] for i in range(18) if gt[i]   == 1]
        pred_names = [CLASS_NAMES[LABEL_IDS[i]] for i in range(18) if pred[i] == 1]


        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f" Réels : {', '.join(gt_names ) or 'aucun'}\n"
            f" Prédits : {', '.join(pred_names) or 'aucun'}",
            fontsize=8, loc="left"
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()



def plot_predictions_custom_seuils(indices, val_ds, y_pred, thresholds):

    all_images, all_labels = [], []
    for imgs, labs in val_ds:
        all_images.append(imgs.numpy())
        all_labels.append(labs.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    LABEL_IDS   = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]
    CLASS_NAMES = {
        1:"Personne", 2:"Vélo", 3:"Road", 4:"Moto", 5:"Avion",
        6:"Bus", 7:"Train", 8:"Truck", 9:"Bateau", 10:"Feux tricolore",
        11:"Bouche incendie", 13:"Feux signalisation", 14:"Parking meter",
        15:"Banc", 16:"Oiseau", 17:"Chat", 18:"Chien", 19:"Cheval"
    }

    n    = len(indices)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for ax, idx in zip(axes, indices):
        img  = all_images[idx].astype(np.uint8)
        gt   = all_labels[idx]

        pred = (y_pred[idx] >= thresholds).astype(int)

        gt_names   = [CLASS_NAMES[LABEL_IDS[i]] for i in range(18) if gt[i]   == 1]
        pred_names = [CLASS_NAMES[LABEL_IDS[i]] for i in range(18) if pred[i] == 1]

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f" Réels   : {', '.join(gt_names)   or 'aucun'}\n"
            f" Prédits : {', '.join(pred_names) or 'aucun'}",
            fontsize=8, loc="left"
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()



def find_best_thresholds(y_true, y_pred_probs, thresholds=np.arange(0.1, 0.9, 0.05)):
 
    best_thresholds = []
    best_f1s        = []

    for label_idx in range(y_true.shape[1]):
        best_t  = 0.5
        best_f1 = 0

        for t in thresholds:
            preds = (y_pred_probs[:, label_idx] >= t).astype(int)
            f1    = f1_score(y_true[:, label_idx], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t  = t

        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
        print(f"Label {label_idx + 1:2d} - seuil optimal : {best_t:.2f} | F1 : {best_f1:.4f}")

    return np.array(best_thresholds), np.array(best_f1s)




def downsample_pure_label1(dataframe, frac=0.5, random_state=42):
 
    pure_label1_mask = dataframe["Labels"].apply(
        lambda x: len(x) == 1 and x[0] == 1
    )

    pure_label1_df = dataframe[pure_label1_mask].sample(frac=frac, random_state=random_state)
    rest_df        = dataframe[~pure_label1_mask]

    result = pd.concat([pure_label1_df, rest_df]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Label 1 pur avant  : {pure_label1_mask.sum()}")
    print(f"Label 1 pur après  : {len(pure_label1_df)}")
    print(f"Reste (inchangé)   : {len(rest_df)}")
    print(f"Total avant        : {len(dataframe)}")
    print(f"Total après        : {len(result)}")

    return result




def load_image_for_lime(image_id, data_dir):
 
    img_path = os.path.join(data_dir, image_id)
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    return np.array(img)


def predict_fn(images):
 
    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    preds = efficient_model.predict(images_tensor, verbose=0)
    return preds


def explain_all_active_labels(image_np, true_labels, pred_probs, threshold=0.3):

    active_labels = np.where(pred_probs >= threshold)[0]

    for label_idx in active_labels:
        label_name = f"Label {label_idx + 1}"
        prob       = pred_probs[label_idx]
        true       = true_labels[label_idx]
        print(f"\n→ {label_name} | prob={prob:.2f} | vrai label={true}")
        explain_image_for_label(image_np, label_idx, label_name)



def explain_image_for_label(image_np, label_idx, label_name, num_samples=1000):

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=18,
        hide_color=0,
        num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        label_idx,
        positive_only=False,
        num_features=5,
        hide_rest=False
    )

    img_boundary = mark_boundaries(temp / 255.0, mask)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np.astype(np.uint8))
    plt.title("Image originale")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_boundary)
    plt.title(f"LIME — {label_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()