import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
import os


def train_classifier(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "svm":
        model = LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test, label_encoder=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if label_encoder:
        target_names = label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    else:
        report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}, y_pred


def train_salary_regressor(X_train, y_train, model_type="gbr"):
    if model_type == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42,
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}, y_pred


def predict_category(model, tfidf_vectorizer, label_encoder, text, skills=None):
    if skills and len(skills) > 0:
        skill_based = predict_category_by_skills(skills, label_encoder)
        if skill_based:
            return skill_based

    text_features = tfidf_vectorizer.transform([text])
    pred = model.predict(text_features)[0]
    category = label_encoder.inverse_transform([pred])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_features)[0]
        confidence = max(proba)
    else:
        confidence = 0.0

    return category, confidence


CATEGORY_SKILL_SIGNATURES = {
    "Java Developers/Architects Resumes": {
        "primary": {"java", "spring", "spring boot", "hibernate", "mybatis", "maven", "jsp"},
        "secondary": {"spring mvc", "spring cloud", "struts", "jsf", "gradle", "tomcat", "weblogic", "websphere", "jboss"},
    },
    "Web Developer Resumes": {
        "primary": {"react", "angular", "vue", "vue.js", "javascript", "typescript", "html", "css", "node.js", "express"},
        "secondary": {"bootstrap", "jquery", "sass", "less", "tailwind", "webpack", "vite", "next.js", "redux", "vuex"},
    },
    "SQL Developers Resumes": {
        "primary": {"sql", "pl/sql", "t-sql", "sql server", "mysql", "postgresql", "oracle"},
        "secondary": {"ssis", "ssrs", "ssas", "stored procedures", "nosql", "db2", "sybase"},
    },
    "Business Analyst (BA) Resumes": {
        "primary": {"business analysis", "requirements", "stakeholder", "jira", "confluence"},
        "secondary": {"user stories", "use cases", "process flow", "visio", "agile", "scrum", "ba"},
    },
    "Network and Systems Administrators Resumes": {
        "primary": {"network", "linux", "windows server", "tcp/ip", "dns", "dhcp", "firewall"},
        "secondary": {"active directory", "vpn", "cisco", "vmware", "powershell", "bash", "nagios"},
    },
    "Datawarehousing, ETL, Informatica Resumes": {
        "primary": {"etl", "informatica", "data warehouse", "datawarehousing", "ssis", "talend", "pandas", "scikit-learn", "xgboost", "airflow"},
        "secondary": {"hive", "hadoop", "spark", "teradata", "snowflake", "datastage", "ab initio", "numpy", "jupyter", "matplotlib"},
    },
    "Business Intelligence, Business Object Resumes": {
        "primary": {"business intelligence", "bi", "tableau", "power bi", "businessobjects", "crystal reports"},
        "secondary": {"qlik", "looker", "cognos", "microstrategy", "dashboards", "reporting", "olap"},
    },
    "Project Manager Resumes": {
        "primary": {"project management", "pmp", "prince2", "agile", "scrum master"},
        "secondary": {"jira", "confluence", "ms project", "risk management", "stakeholder management", "itil"},
    },
    "Recruiter Resumes": {
        "primary": {"recruiting", "recruitment", "talent acquisition", "hiring", "sourcing"},
        "secondary": {"linkedin", "applicant tracking", "onboarding", "hr", "human resources", "interviewing"},
    },
}


def predict_category_by_skills(skills, label_encoder):
    skills_lower = set(s.lower() for s in skills)
    valid_categories = set(label_encoder.classes_)

    best_category = None
    best_score = -1

    for category, sig in CATEGORY_SKILL_SIGNATURES.items():
        if category not in valid_categories:
            continue
        primary = set(s.lower() for s in sig["primary"])
        secondary = set(s.lower() for s in sig["secondary"])

        primary_hits = len(skills_lower & primary)
        secondary_hits = len(skills_lower & secondary)

        score = primary_hits * 3 + secondary_hits * 1

        if score > best_score:
            best_score = score
            best_category = category

    if best_category and best_score >= 3:
        confidence = min(best_score / 15.0, 0.95)
        return best_category, confidence

    return None


def predict_salary(regressor, scaler, skill_matrix_row, experience_years, skill_count):
    features = np.concatenate([
        skill_matrix_row.flatten(),
        [experience_years, skill_count],
    ])
    features = features.reshape(1, -1)
    if scaler:
        features = scaler.transform(features)
    salary = regressor.predict(features)[0]
    return round(salary)


def get_skill_importance(regressor, skill_names):
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        n_skills = len(skill_names)
        skill_importances = importances[:n_skills]
        ranked = sorted(
            zip(skill_names, skill_importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked
    return []


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)


def prepare_training_data(df, tfidf_vectorizer, tfidf_matrix, skill_matrix, label_encoder):
    X_tfidf = tfidf_matrix
    y_class = label_encoder.transform(df["category"])

    additional_features = df[["experience_years", "skill_count"]].values
    X_salary = np.hstack([skill_matrix, additional_features])
    y_salary = df["salary_mid"].values

    return X_tfidf, y_class, X_salary, y_salary
