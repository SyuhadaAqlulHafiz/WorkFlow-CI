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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import estimator_html_repr

def main(data_path):
    # CATATAN: mlflow.set_tracking_uri() dan mlflow.set_experiment() DENGAN SENDIRINYA
    # dikonfigurasi melalui Env Variables di GitHub Actions maupun CLI lokal agar fleksibel.
    
    # Otomatis mencatat params, metrics bawaan, input_example, dan folder 'model' saat .fit() dipanggil
    mlflow.autolog(log_input_examples=True, log_model_signatures=True)

    # === 2. Load Data ===
    if not os.path.exists(data_path):
        print(f"❌ File tidak ditemukan di {data_path}. Pastikan folder dan file sudah benar!")
        return

    print(f"📦 Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)

    # --- PROTEKSI VALUERROR: Bersihkan kelas yang sampelnya kurang dari 2 ---
    counts = df['salary_bin'].value_counts()
    kelas_langka = counts[counts < 2].index.tolist()
    if len(kelas_langka) > 0:
        print(f"⚠️ Membersihkan kelas langka kurang dari 2 sampel: {kelas_langka}")
        df = df[df['salary_bin'].isin(counts[counts >= 2].index)]
    # ------------------------------------------------------------------------

    X = pd.get_dummies(df.drop(columns=["salary_bin"]))
    y = LabelEncoder().fit_transform(df["salary_bin"])

    # === 3. Split Dataset ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === 4. Train Model (Di dalam Run MLflow) ===
    with mlflow.start_run(run_name="Random_Forest_Autolog_Base"):
        print("\n🏋️ Melatih Model...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train) # <--- Folder 'model' OTOMATIS dibuat di sini oleh autolog

        # === 5. Evaluate ===
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # Catat metrik tambahan kustom jika diperlukan
        mlflow.log_metrics({
            "custom_f1_score": f1,
            "custom_precision": precision,
            "custom_recall": recall
        })

        # === 6. Pembuatan Artefak Kustom ===
        print("📊 Membuat dan mengunggah artefak kustom...")
        
        # A. Buat & Log metric_info.json
        metric_info = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }
        with open("metric_info.json", "w") as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        # B. Buat & Log training_confusion_matrix.png
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Base Model")
        plt.tight_layout()
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")

        # C. Buat & Log estimator.html
        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(model))
        mlflow.log_artifact("estimator.html")

        print("\n✅ Model & Semua Artefak Kustom sukses dicatat!")
        print(f"🔢 Akurasi: {acc:.4f} | F1-score: {f1:.4f}")

if __name__ == "__main__":
    # Menggunakan argparse agar dinamis menerima input dari MLProject entry point saat di CI pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    # Fallback jika dijalankan manual tanpa argumen (tombol play/run VS Code), mengarah ke file lokal
    if args.data_path is None:
        args.data_path = os.path.join("JobDataset_preprocessing", "job_dataset_preprocessed.csv")
        # Jika dijalankan lokal di luar folder MLProject, sesuaikan dengan mendaftarkan URI lokal server Anda:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment("Job_Salary_Classification_Local")

    main(args.data_path)
