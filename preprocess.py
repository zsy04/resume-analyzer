import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_data, preprocess_data, build_tfidf_features, build_skill_feature_matrix,
    encode_labels, get_feature_stats, SKILL_CATEGORY_CN, CATEGORY_CN, SALARY_RANGES,
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
    log("  智能简历分析系统 - 数据预处理与模型训练")
    log("=" * 50)

    log("\n[1/6] 加载原始数据...")
    t = time.time()
    df = load_data(DATA_PATH)
    log(f"  完成! 耗时 {time.time()-t:.1f}s, 共 {len(df)} 条简历")

    log("\n[2/6] 数据预处理（技能提取、薪资标注）...")
    t = time.time()
    df = preprocess_data(df)
    log(f"  完成! 耗时 {time.time()-t:.1f}s")

    log("\n[3/6] 构建特征矩阵...")
    t = time.time()
    tfidf_vectorizer, tfidf_matrix = build_tfidf_features(df["cleaned_text"])
    skill_matrix, skill_names, skill_to_idx = build_skill_feature_matrix(df)
    label_encoder, encoded_labels = encode_labels(df["category"])
    log(f"  完成! 耗时 {time.time()-t:.1f}s")

    log("\n[4/6] 训练分类模型...")
    t = time.time()
    X_tfidf, y_class, X_salary, y_salary = prepare_training_data(
        df, tfidf_vectorizer, tfidf_matrix, skill_matrix, label_encoder
    )
    X_train_t, X_test_t, y_train_c, y_test_c = train_test_split(
        X_tfidf, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    classifier = train_classifier(X_train_t, y_train_c, model_type="rf")
    class_metrics, _ = evaluate_classifier(classifier, X_test_t, y_test_c, label_encoder)
    log(f"  完成! 耗时 {time.time()-t:.1f}s, 准确率: {class_metrics['accuracy']:.2%}")

    log("\n[5/6] 训练薪资预测模型...")
    t = time.time()
    scaler = StandardScaler()
    X_salary_scaled = scaler.fit_transform(X_salary)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_salary_scaled, y_salary, test_size=0.2, random_state=42
    )
    regressor = train_salary_regressor(X_train_s, y_train_s, model_type="gbr")
    salary_metrics, _ = evaluate_regressor(regressor, X_test_s, y_test_s)
    skill_importance = get_skill_importance(regressor, skill_names)
    log(f"  完成! 耗时 {time.time()-t:.1f}s, R2: {salary_metrics['R2']:.4f}")

    log("\n[6/6] 保存缓存文件...")
    t = time.time()

    df_save = df.drop(columns=["skill_dict"])
    df_save.to_parquet(os.path.join(CACHE_DIR, "processed_data.parquet"), index=False)

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

    total = time.time() - total_start
    log(f"\n{'='*50}")
    log(f"  全部完成! 总耗时 {total:.1f}s")
    log(f"  缓存文件已保存至: {CACHE_DIR}")
    log(f"{'='*50}")


if __name__ == "__main__":
    main()
