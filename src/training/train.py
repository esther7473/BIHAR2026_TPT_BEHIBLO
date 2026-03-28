import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from mlflow.models.signature import infer_signature
from src.common.common import CONFIG
from src.training.preprocessing import run_preprocessing
from src.data.database import load_weather_data, save_predictions
import pickle


# ─────────────────────────────────────────
# Modèle
# ─────────────────────────────────────────

def build_model(input_shape, horizon, lstm_units_l1, lstm_units_l2, dropout, lr):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(lstm_units_l1, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units_l2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(horizon)
    ])
    model.compile(optimizer=RMSprop(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────
# Inverse transform
# ─────────────────────────────────────────

def inverse_transform_multi(preds, scaler, n_features):
    """Inverse le scaling sur la colonne température uniquement."""
    result = []
    for i in range(preds.shape[1]):
        temp = np.zeros((len(preds), n_features))
        temp[:, 0] = preds[:, i]
        inv = scaler.inverse_transform(temp)[:, 0]
        result.append(inv)
    return np.array(result).T


# ─────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────

def train():
    # ── Params ──
    lookback      = CONFIG["model"]["lookback"]
    horizon       = CONFIG["model"]["horizon"]
    epochs        = CONFIG["model"]["epochs"]
    batch_size    = CONFIG["model"]["batch_size"]
    model_name    = CONFIG["model"]["name"]
    lstm_units_l1 = CONFIG["model"]["lstm_units_l1"]
    lstm_units_l2 = CONFIG["model"]["lstm_units_l2"]
    dropout       = CONFIG["model"]["dropout"]
    lr            = CONFIG["model"]["lr"]
    n_splits      = CONFIG["data"]["n_splits"]

    # ── Données ──
    df   = load_weather_data()
    data = run_preprocessing(df)

    X_train    = data["X_train"]
    y_train    = data["y_train"]
    X_test     = data["X_test"]
    y_test     = data["y_test"]
    scaler     = data["scaler"]
    n_features = X_train.shape[2]

    # ── Cross-Validation ──
    print(f"\n Cross-Validation ({n_splits} folds)...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores, rmse_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"  Fold {fold+1}/{n_splits}")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model_cv = build_model(
            (X_tr.shape[1], X_tr.shape[2]),
            horizon, lstm_units_l1, lstm_units_l2, dropout, lr
        )
        model_cv.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        preds_cv = model_cv.predict(X_val, verbose=0)

        # ✅ inverse transform en °C
        preds_cv_real = inverse_transform_multi(preds_cv, scaler, n_features)
        y_val_real    = inverse_transform_multi(y_val,    scaler, n_features)

        mae_scores.append(mean_absolute_error(y_val_real,  preds_cv_real))
        rmse_scores.append(root_mean_squared_error(y_val_real, preds_cv_real))

        print(f"    MAE: {mae_scores[-1]:.2f}°C | RMSE: {rmse_scores[-1]:.2f}°C")

        del model_cv
        tf.keras.backend.clear_session()

    mae_cv  = round(float(np.mean(mae_scores)),  2)
    rmse_cv = round(float(np.mean(rmse_scores)), 2)
    print(f"\n  ✅ MAE CV moyen  : {mae_cv:.2f}°C")
    print(f"  ✅ RMSE CV moyen : {rmse_cv:.2f}°C")

    # ── MLflow ──
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    run_name = (
        f"lstm"
        f"_lb{lookback}"
        f"_h{horizon}"
        f"_u{lstm_units_l1}-{lstm_units_l2}"
        f"_dr{dropout}"
        f"_lr{lr}"
        f"_bs{batch_size}"
        f"_rmsprop"
    )

    with mlflow.start_run(run_name=run_name):

        # ── Log hyperparamètres ──
        mlflow.log_params({
            "lookback":       lookback,
            "horizon":        horizon,
            "epochs":         epochs,
            "batch_size":     batch_size,
            "lstm_units_l1":  lstm_units_l1,
            "lstm_units_l2":  lstm_units_l2,
            "dropout":        dropout,
            "lr":             lr,
            "optimizer":      "rmsprop",
            "n_splits":       n_splits,
        })

        # ── Log métriques CV ──
        mlflow.log_metrics({
            "cv_mae":  mae_cv,
            "cv_rmse": rmse_cv,
        })

        # ── Entraînement final ──
        print("\n  Entraînement final...")
        model = build_model(
            (X_train.shape[1], X_train.shape[2]),
            horizon, lstm_units_l1, lstm_units_l2, dropout, lr
        )

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        )

        # ── Signature ──
        sample_input  = X_train[:5]
        sample_output = model.predict(sample_input, verbose=0)
        signature     = infer_signature(
            model_input=sample_input,
            model_output=sample_output
        )

        # ── Évaluation scaled ──
        preds_scaled = model.predict(X_test, verbose=0)
        mae_scaled   = mean_absolute_error(y_test, preds_scaled)
        rmse_scaled  = root_mean_squared_error(y_test, preds_scaled)

        # ── Évaluation réelle ──
        preds_real = inverse_transform_multi(preds_scaled, scaler, n_features)
        y_real     = inverse_transform_multi(y_test,       scaler, n_features)
        mae_real   = mean_absolute_error(y_real, preds_real)
        rmse_real  = root_mean_squared_error(y_real, preds_real)

        print(f"\n Scaled  — MAE: {mae_scaled:.4f} | RMSE: {rmse_scaled:.4f}")
        print(f" Réel    — MAE: {mae_real:.2f}°C  | RMSE: {rmse_real:.2f}°C")

        # ── Log métriques epoch par epoch ──
        for epoch, (loss, val_loss) in enumerate(zip(
            history.history["loss"],
            history.history["val_loss"]
        )):
            mlflow.log_metrics({
                "train_loss": round(loss,     4),
                "val_loss":   round(val_loss, 4),
            }, step=epoch)

        # ── Log métriques finales ──
        mlflow.log_metrics({
            "mae_scaled":  round(mae_scaled,  2),
            "rmse_scaled": round(rmse_scaled, 2),
            "mae_real":    round(mae_real,    2),
            "rmse_real":   round(rmse_real,   2),
            "epochs_run":  len(history.history["loss"]),
        })

        # ── Sauvegarde scaler ──
        scaler_path = CONFIG["paths"]["scaler_path"]
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        print(f" Scaler sauvegardé : {scaler_path}")

        # ── Sauvegarde modèle ──
        mlflow.keras.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name=model_name
        )
        model.save(CONFIG["paths"]["model_path"])
        print(f" Modèle sauvegardé : {CONFIG['paths']['model_path']}")

   

if __name__ == "__main__":
    train()