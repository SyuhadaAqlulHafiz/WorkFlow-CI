# # import os
# # import json
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# # from sklearn.utils import estimator_html_repr

# # # Pustaka MLflow Lokal
# # import mlflow
# # import mlflow.sklearn

# # def main():
# #     # 1. INISIALISASI MLFLOW LOKAL
# #     # Mengarahkan tracking ke server lokal yang berjalan di port 5000
# #     mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
# #     # Set nama eksperimen di server lokal
# #     mlflow.set_experiment("Job_Salary_Classification_Local")

# #     # Mengaktifkan autolog untuk otomatis mencatat parameter dan metrik bawaan scikit-learn
# #     mlflow.autolog()

# #     # 2. MUAT DATASET
# #     # Menyesuaikan path relatif untuk workspace VS Code
# #     data_path = os.path.join("JobDataset_preprocessing", "Job_dataset_preprocessed.csv")
    
# #     if not os.path.exists(data_path):
# #         print(f"❌ File tidak ditemukan di {data_path}. Pastikan folder dan file sudah benar!")
# #         return

# #     print(f"📦 Memuat data dari: {data_path}")
# #     df = pd.read_csv(data_path)

# #     # --- [PROTEKSI VALUERROR]: Bersihkan kelas langka kurang dari 2 sampel ---
# #     counts = df['salary_bin'].value_counts()
# #     kelas_langka = counts[counts < 2].index.tolist()
# #     if len(kelas_langka) > 0:
# #         print(f"⚠️ Membersihkan kelas langka kurang dari 2 sampel: {kelas_langka}")
# #         df = df[df['salary_bin'].isin(counts[counts >= 2].index)]
# #     # ------------------------------------------------------------------------

# #     # Pisahkan Fitur dan Target
# #     X = df.drop(columns=['salary_bin'])
# #     y = df['salary_bin']
    
# #     # Split data dengan stratify agar distribusi kelas seimbang
# #     X_train, X_val, y_train, y_val = train_test_split(
# #         X, y, test_size=0.2, random_state=42, stratify=y
# #     )

# #     # Ambil contoh input untuk signature model di MLflow
# #     input_example = X_train[0:5]

# #     # 3. RUN EXPERIMENT DI MLFLOW LOKAL
# #     with mlflow.start_run(run_name="Random_Forest_Local_Base"):
# #         print("\n🏋️ Melatih Model")

# #         # Inisialisasi hyperparameter (Bisa disesuaikan sesuka hati)
# #         n_estimators = 100
# #         max_depth = None

# #         model = RandomForestClassifier(
# #             n_estimators=n_estimators, 
# #             max_depth=max_depth, 
# #             random_state=42
# #         )
# #         model.fit(X_train, y_train)

# #         # Prediksi & Hitung Evaluasi tambahan
# #         y_pred = model.predict(X_val)
# #         acc = accuracy_score(y_val, y_pred)

# #         # Log metrik manual (opsional, karena autolog sudah mencatat sebagian besar metrik)
# #         mlflow.log_metric("custom_val_accuracy", acc)

# #         # 4. PEMBUATAN DAN LOGGING ARTEFAK KUSTOM
# #         # Buat folder temporer di workspace lokal untuk menyimpan file sebelum di-upload
# #         temp_artifacts_dir = "temp_local_artifacts"
# #         os.makedirs(temp_artifacts_dir, exist_ok=True)

# #         # ARTEFAK 1: metric_info.json
# #         report_dict = classification_report(y_val, y_pred, output_dict=True)
# #         with open(os.path.join(temp_artifacts_dir, "metric_info.json"), "w") as f:
# #             json.dump(report_dict, f, indent=4)

# #         # ARTEFAK 2: training_confusion_matrix.png
# #         plt.figure(figsize=(8, 6))
# #         cm = confusion_matrix(y_val, y_pred)
# #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #                     xticklabels=model.classes_, yticklabels=model.classes_)
# #         plt.title("Confusion Matrix - Local Base Model")
# #         plt.ylabel('Actual')
# #         plt.xlabel('Predicted')
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(temp_artifacts_dir, "training_confusion_matrix.png"))
# #         plt.close()

# #         # ARTEFAK 3: estimator.html (Visualisasi Pohon/Struktur Model)
# #         with open(os.path.join(temp_artifacts_dir, "estimator.html"), "w", encoding="utf-8") as f:
# #             f.write(estimator_html_repr(model))

# #         # --- UPLOAD ARTEFAK DAN MODEL KE MLFLOW LOKAL ---
# #         # Kirim semua file di dalam folder temp ke root artefak MLflow
# #         mlflow.log_artifacts(temp_artifacts_dir)

# #         # Log model dengan input_example agar mempermudah deployment/testing nanti
# #         mlflow.sklearn.log_model(
# #             sk_model=model,
# #             artifact_path="model",
# #             input_example=input_example
# #         )

# #         print(f"✅ Model selesai dilatih. Akurasi Validasi: {acc:.4f}")
# #         print("✅ Semua artefak (JSON, PNG, HTML) sukses dikirim ke MLflow Lokal!")

# # if __name__ == "__main__":
# #     main()

# # modelling.py

# # import argparse
# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import mlflow
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# from sklearn.utils import estimator_html_repr

# def main(data_path):
#     # === 1. Inisialisasi Autolog ===
#     # Mengarahkan tracking ke server lokal yang berjalan di port 5000
#     mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
#     # Set nama eksperimen di server lokal
#     mlflow.set_experiment("Job_Salary_Classification_Local")
    
#     # Otomatis mencatat params, metrics bawaan, input_example, dan folder 'model' saat .fit() dipanggil
#     mlflow.autolog(log_input_examples=True, log_model_signatures=True)

#     # === 2. Load data ===
#     df = pd.read_csv(data_path)

#     X = pd.get_dummies(df.drop(columns=["salary_bin"]))
#     y = LabelEncoder().fit_transform(df["salary_bin"])

#     # === 3. Split ===
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     # === 4. Train model (Di dalam Run MLflow) ===
#     with mlflow.start_run(run_name="Random_Forest_Autolog_Base"):
#         print("\n🏋️ Melatih Model...")
#         model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
#         model.fit(X_train, y_train) # <--- Folder 'model' OTOMATIS dibuat di sini oleh autolog

#         # === 5. Evaluate ===
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average="weighted")
#         precision = precision_score(y_test, y_pred, average="weighted")
#         recall = recall_score(y_test, y_pred, average="weighted")

#         # Catat metrik tambahan kustom jika belum tercakup oleh autolog bawaan
#         mlflow.log_metrics({
#             "custom_f1_score": f1,
#             "custom_precision": precision,
#             "custom_recall": recall
#         })

#         # === 6. Pembuatan Artefak Kustom (Sejajar di Root Artifact) ===
        
#         # A. Buat & Log metric_info.json
#         metric_info = {
#             "accuracy": float(acc),
#             "f1_score": float(f1),
#             "precision": float(precision),
#             "recall": float(recall)
#         }
#         with open("metric_info.json", "w") as f:
#             json.dump(metric_info, f, indent=4)
#         mlflow.log_artifact("metric_info.json")

#         # B. Buat & Log training_confusion_matrix.png
#         plt.figure(figsize=(6, 4))
#         cm = confusion_matrix(y_test, y_pred)
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title("Confusion Matrix - Base Model")
#         plt.tight_layout()
#         plt.savefig("training_confusion_matrix.png")
#         plt.close()
#         mlflow.log_artifact("training_confusion_matrix.png")

#         # C. Buat & Log estimator.html
#         with open("estimator.html", "w", encoding="utf-8") as f:
#             f.write(estimator_html_repr(model))
#         mlflow.log_artifact("estimator.html")

#         print("✅ Model & Artefak Kustom sukses dicatat via Autolog.")
#         print(f"🔢 Akurasi: {acc:.4f} | F1-score: {f1:.4f}")

# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--data_path", type=str, required=True)
#     # args = parser.parse_args()
#     path_dataset_lokal = os.path.join("JobDataset_preprocessing", "job_dataset_preprocessed.csv")
#     main(path_dataset_lokal)

# modelling.py

import os
import json
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
    # === 1. Inisialisasi Autolog & Tracking URI ===
    # Mengarahkan tracking ke server lokal yang berjalan di port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    # Set nama eksperimen di server lokal
    mlflow.set_experiment("Job_Salary_Classification_Local")
    
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

        # === 6. Pembuatan Artefak Kustom (Sejajar di Root Artifact) ===
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

        print("\n✅ Model & Semua Artefak Kustom sukses dicatat via Autolog ke MLflow Lokal!")
        print(f"🔢 Akurasi: {acc:.4f} | F1-score: {f1:.4f}")

if __name__ == "__main__":
    # Langsung arahkan ke path dataset lokal Anda agar bisa langsung klik tombol Run/Play di VS Code
    path_dataset_lokal = os.path.join("JobDataset_preprocessing", "Job_dataset_preprocessed.csv")
    main(path_dataset_lokal)