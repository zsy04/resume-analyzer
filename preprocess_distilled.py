import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_data, preprocess_data, build_tfidf_features, build_skill_feature_matrix,
    encode_labels, SKILL_CATEGORY_CN, CATEGORY_CN, SALARY_RANGES,
    extract_skills, extract_experience_years, clean_text,
)
from model import (
    train_classifier, evaluate_classifier, train_salary_regressor,
    evaluate_regressor, get_skill_importance, prepare_training_data,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Resume_dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

LOG_FILE = os.path.join(CACHE_DIR, "preprocess_log.txt")


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    total_start = time.time()
    log("=" * 50)
    log("  智能简历分析系统 - 精简版模型训练")
    log("=" * 50)

    parquet_path = os.path.join(CACHE_DIR, "processed_data.parquet")
    if os.path.exists(parquet_path):
        log("\n加载已处理的数据...")
        df = pd.read_parquet(parquet_path)
        log(f"  完成! 共 {len(df)} 条简历")
    else:
        log("\n[1/6] 加载原始数据...")
        t = time.time()
        df = load_data(DATA_PATH)
        log(f"  完成! 耗时 {time.time()-t:.1f}s, 共 {len(df)} 条简历")

        log("\n[2/6] 数据预处理（技能提取、薪资标注）...")
        t = time.time()
        df = preprocess_data(df)
        log(f"  完成! 耗时 {time.time()-t:.1f}s")

        df_save = df.drop(columns=["skill_dict"])
        df_save.to_parquet(parquet_path, index=False)

    log("\n[3/6] 构建特征矩阵...")
    t = time.time()
    tfidf_vectorizer, tfidf_matrix = build_tfidf_features(df["cleaned_text"])
    skill_matrix, skill_names, skill_to_idx = build_skill_feature_matrix(df)
    label_encoder, encoded_labels = encode_labels(df["category"])
    log(f"  完成! 耗时 {time.time()-t:.1f}s")

    log("\n[4/6] 训练精简版分类模型...")
    t = time.time()
    from sklearn.ensemble import RandomForestClassifier
    X_tfidf, y_class, X_salary, y_salary = prepare_training_data(
        df, tfidf_vectorizer, tfidf_matrix, skill_matrix, label_encoder
    )
    X_train_t, X_test_t, y_train_c, y_test_c = train_test_split(
        X_tfidf, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train_t, y_train_c)
    y_pred = classifier.predict(X_test_t)
    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test_c, y_pred)
    class_metrics = {"accuracy": acc, "report": classification_report(y_test_c, y_pred, output_dict=True)}
    log(f"  完成! 耗时 {time.time()-t:.1f}s, 准确率: {acc:.2%}")

    log("\n[5/6] 训练精简版薪资预测模型...")
    t = time.time()
    from sklearn.ensemble import GradientBoostingRegressor
    scaler = StandardScaler()
    X_salary_scaled = scaler.fit_transform(X_salary)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_salary_scaled, y_salary, test_size=0.2, random_state=42
    )
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    regressor.fit(X_train_s, y_train_s)
    y_pred_s = regressor.predict(X_test_s)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test_s, y_pred_s)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_s, y_pred_s)
    r2 = r2_score(y_test_s, y_pred_s)
    salary_metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    skill_importance = []
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        n_skills = len(skill_names)
        skill_importances = importances[:n_skills]
        skill_importance = sorted(
            zip(skill_names, skill_importances),
            key=lambda x: x[1],
            reverse=True,
        )
    log(f"  完成! 耗时 {time.time()-t:.1f}s, R2: {r2:.4f}")

    log("\n[6/6] 保存缓存文件...")
    t = time.time()

    joblib.dump(tfidf_vectorizer, os.path.join(CACHE_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(CACHE_DIR, "label_encoder.joblib"))
    joblib.dump(classifier, os.path.join(CACHE_DIR, "classifier.joblib"))
    joblib.dump(regressor, os.path.join(CACHE_DIR, "regressor.joblib"))
    joblib.dump(scaler, os.path.join(CACHE_DIR, "scaler.joblib"))
    joblib.dump(skill_names, os.path.join(CACHE_DIR, "skill_names.joblib"))
    joblib.dump(skill_to_idx, os.path.join(CACHE_DIR, "skill_to_idx.joblib"))
    joblib.dump(skill_importance, os.path.join(CACHE_DIR, "skill_importance.joblib"))
    joblib.dump(class_metrics, os.path.join(CACHE_DIR, "class_metrics.joblib"))
    joblib.dump(salary_metrics, os.path.join(CACHE_DIR, "salary_metrics.joblib"))

    log(f"  完成! 耗时 {time.time()-t:.1f}s")

    total_time = time.time() - total_start
    log(f"\n{'='*50}")
    log(f"  全部完成! 总耗时 {total_time:.1f}s")
    log(f"{'='*50}")

    log("\n模型指标:")
    log(f"  分类准确率: {acc:.2%}")
    log(f"  薪资预测 R2: {r2:.4f}")
    log(f"  薪资预测 MAE: {mae:.2f}")

    import glob
    total_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(CACHE_DIR, "*.joblib")))
    total_size += os.path.getsize(parquet_path)
    log(f"\n缓存总大小: {total_size / 1024 / 1024:.2f} MB")

    for f in glob.glob(os.path.join(CACHE_DIR, "*.joblib")):
        size = os.path.getsize(f)
        log(f"  {os.path.basename(f)}: {size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
