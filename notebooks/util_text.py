
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix


from sklearn.metrics import (multilabel_confusion_matrix,
                              classification_report,
                              f1_score)

from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.layers import GRU
from lime.lime_text import LimeTextExplainer
from sklearn.metrics.pairwise import cosine_similarity




def labels_to_vector(label_list):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for label in label_list:
        if label in ALL_LABELS:
            vec[ALL_LABELS.index(label)] = 1.0
    return vec


def evaluate(y_true, y_pred_proba, threshold=THRESHOLD):
    y_pred = (y_pred_proba >= threshold).astype(int)

    print("=== Métriques agrégées ===")
    print(f"F1 macro     : {f1_score(y_true, y_pred, average='macro',    zero_division=0):.4f}")
    print(f"F1 micro     : {f1_score(y_true, y_pred, average='micro',    zero_division=0):.4f}")
    print(f"F1 weighted  : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 samples   : {f1_score(y_true, y_pred, average='samples',  zero_division=0):.4f}")

    print("\n=== Métriques par classe ===")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))
    return y_pred


def plot_confusion_matrices(y_true, y_pred_proba, threshold=THRESHOLD):
    y_pred = (y_pred_proba >= threshold).astype(int)
    cms    = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    axes = axes.flatten()

    for i, (cm, name) in enumerate(zip(cms, LABEL_NAMES)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    ax=axes[i], cbar=False)
        axes[i].set_title(name, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_predictions(df, y_true, y_pred_proba, threshold=THRESHOLD, n=5, label_filter=None):
    y_pred = (y_pred_proba >= threshold).astype(int)

    if label_filter is not None:
        label_idx = ALL_LABELS.index(label_filter)
        indices   = np.where(y_true[:, label_idx] == 1)[0]
        indices   = np.random.choice(indices, min(n, len(indices)), replace=False)
    else:
        indices = np.random.choice(len(df), n, replace=False)

    for idx in indices:
        true_labels = [LABEL_NAMES[i] for i in range(NUM_CLASSES) if y_true[idx][i] == 1]
        pred_labels = [LABEL_NAMES[i] for i in range(NUM_CLASSES) if y_pred[idx][i] == 1]

        print(f"Caption : {df['Caption'].iloc[idx]}")
        print(f"Réels   : {true_labels}")
        print(f"Prédits : {pred_labels}")
        print("-" * 60)

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



def predict_proba_lime(texts):
    sequences  = tokenizer.texts_to_sequences(texts)
    padded     = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    return model_gru_glove.predict(padded)



def explain_prediction(idx, label_name):
    text = df_val["Caption_clean"].iloc[idx]

    print(f"Caption : {df_val['Caption'].iloc[idx]}")
    print(f"Réels   : {[LABEL_NAMES[i] for i in range(NUM_CLASSES) if y_val[idx][i] == 1]}")
    print(f"Label expliqué : {label_name}")

    label_idx = LABEL_NAMES.index(label_name)

    exp = explainer.explain_instance(
        text,
        predict_proba_lime,
        num_features=10,
        labels=[label_idx]
    )
    exp.show_in_notebook(text=True)
