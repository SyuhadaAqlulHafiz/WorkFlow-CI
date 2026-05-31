import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.utils import estimator_html_repr


def main(data_path):

    # ==========================================================
    # MLFLOW CONFIG
    # ==========================================================
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns"
    )

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "Job_Salary_Classification_Local"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"📍 Tracking URI : {tracking_uri}")
    print(f"🧪 Experiment   : {experiment_name}")

    # Aktifkan autolog tetapi model dicatat manual
    mlflow.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False
    )

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Dataset tidak ditemukan: {data_path}"
        )

    print(f"📦 Memuat data dari: {data_path}")

    df = pd.read_csv(data_path)

    # ==========================================================
    # BERSIHKAN KELAS LANGKA
    # ==========================================================
    counts = df["salary_bin"].value_counts()

    kelas_langka = counts[counts < 2].index.tolist()

    if len(kelas_langka) > 0:
        print(
            f"⚠️ Membersihkan kelas langka kurang dari 2 sampel: "
            f"{kelas_langka}"
        )

        df = df[
            df["salary_bin"].isin(
                counts[counts >= 2].index
            )
        ]

    # ==========================================================
    # PREPROCESSING
    # ==========================================================
    X = pd.get_dummies(
        df.drop(columns=["salary_bin"])
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df["salary_bin"]
    )

    # ==========================================================
    # TRAIN TEST SPLIT
    # ==========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ==========================================================
    # TRAINING
    # ==========================================================
    with mlflow.start_run(
        run_name="Random_Forest_Base_Model"
    ):

        print("\n🏋️ Melatih Model...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ======================================================
        # LOG MODEL EXPLICITLY
        # Akan tersimpan di:
        # artifacts/model/
        # ======================================================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("✅ Model berhasil dicatat ke artifacts/model")

        # ======================================================
        # EVALUASI
        # ======================================================
        y_pred = model.predict(X_test)

        acc = accuracy_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # ======================================================
        # LOG METRICS
        # ======================================================
        mlflow.log_metrics({
            "accuracy": acc,
            "custom_f1_score": f1,
            "custom_precision": precision,
            "custom_recall": recall
        })

        # ======================================================
        # ARTIFACT 1
        # metric_info.json
        # ======================================================
        print("📊 Membuat artefak...")

        metric_info = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }

        with open(
            "metric_info.json",
            "w"
        ) as f:
            json.dump(
                metric_info,
                f,
                indent=4
            )

        mlflow.log_artifact(
            "metric_info.json"
        )

        # ======================================================
        # ARTIFACT 2
        # Confusion Matrix
        # ======================================================
        cm = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            "Confusion Matrix - Random Forest"
        )

        plt.tight_layout()

        plt.savefig(
            "training_confusion_matrix.png"
        )

        plt.close()

        mlflow.log_artifact(
            "training_confusion_matrix.png"
        )

        # ======================================================
        # ARTIFACT 3
        # estimator.html
        # ======================================================
        with open(
            "estimator.html",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(
                estimator_html_repr(model)
            )

        mlflow.log_artifact(
            "estimator.html"
        )

        print("\n✅ Semua artefak berhasil dicatat")
        print(f"🎯 Accuracy  : {acc:.4f}")
        print(f"🎯 F1 Score  : {f1:.4f}")
        print(f"🎯 Precision : {precision:.4f}")
        print(f"🎯 Recall    : {recall:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join(
            "JobDataset_preprocessing",
            "job_dataset_preprocessed.csv"
        )

    main(args.data_path)
