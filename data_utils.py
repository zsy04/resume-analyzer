import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


SKILL_DICTIONARY = {
    "programming_languages": [
        "Java", "Python", "C#", "C\\+\\+", "JavaScript", "TypeScript", "Ruby",
        "PHP", "Go", "Rust", "Swift", "Objective-C", "Kotlin", "Scala", "Perl",
        "R", "MATLAB", "Shell", "Bash", "PowerShell", "PL/SQL", "T-SQL",
        "HTML", "CSS", "SQL", "NoSQL", "Groovy", "Lua", "Dart", "VBA",
        "C", "Visual Basic", "ABAP", "COBOL", "Fortran", "Haskell", "Erlang",
        "Clojure", "F#", "Julia",
    ],
    "frameworks": [
        "Spring", "Spring Boot", "Spring MVC", "Hibernate", "Struts", "JSF",
        "Django", "Flask", "FastAPI", "React", "Angular", "Vue", "Vue.js",
        "Node.js", "Express", "Express.js", "Rails", "Ruby on Rails",
        "Laravel", "Symfony", ".NET", ".NET Core", "ASP.NET", "ASP.NET MVC",
        "WPF", "WinForms", "Entity Framework", "Blazor",
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
        "Spark", "Hadoop", "Storm", "Flink", "XGBoost", "LightGBM",
        "Bootstrap", "jQuery", "Tailwind", "SASS", "LESS",
        "MyBatis", "iBatis", "Play Framework", "Grails", "GWT",
        "Cordova", "Ionic", "Xamarin", "Flutter",
        "Cocoa Touch", "Core Data", "UIKit",
        "Airflow", "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI",
        "LangChain", "OpenCV", "NLTK", "Spacy",
    ],
    "databases": [
        "Oracle", "MySQL", "SQL Server", "PostgreSQL", "MongoDB", "Redis",
        "Cassandra", "CouchDB", "SQLite", "DB2", "Sybase", "Informix",
        "Elasticsearch", "Solr", "Neo4j", "DynamoDB", "Couchbase",
        "MariaDB", "HBase", "Hive", "Teradata", "Snowflake",
        "MS Access", "FileMaker",
    ],
    "devops_tools": [
        "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab",
        "SVN", "CVS", "Maven", "Gradle", "Ant", "NPM", "Yarn",
        "AWS", "Azure", "GCP", "Google Cloud", "Heroku",
        "Terraform", "Ansible", "Puppet", "Chef", "Vagrant",
        "Nagios", "Prometheus", "Grafana", "ELK", "Splunk",
        "Jira", "Confluence", "Bamboo", "TeamCity",
        "Nexus", "Artifactory", "SonarQube",
    ],
    "web_servers": [
        "Tomcat", "WebLogic", "WebSphere", "JBoss", "Nginx", "Apache",
        "IIS", "Jetty", "WildFly", "GlassFish",
    ],
    "methodologies": [
        "Agile", "Scrum", "Kanban", "Waterfall", "DevOps", "CI/CD",
        "TDD", "BDD", "SDLC", "ITIL", "Lean", "Six Sigma",
        "Microservices", "REST", "SOAP", "SOA", "MVC", "MVVM",
        "OOD", "OOP", "OOAD", "UML", "Design Patterns",
    ],
    "soft_skills": [
        "Communication", "Leadership", "Team Player", "Problem Solving",
        "Analytical", "Time Management", "Project Management",
        "Stakeholder Management", "Client Facing", "Mentoring",
    ],
}

SKILL_CATEGORY_CN = {
    "programming_languages": "编程语言",
    "frameworks": "框架与库",
    "databases": "数据库",
    "devops_tools": "DevOps与工具",
    "web_servers": "应用服务器",
    "methodologies": "方法论与架构",
    "soft_skills": "软技能",
}

CATEGORY_CN = {
    "Java Developers/Architects Resumes": "Java开发/架构",
    "Web Developer Resumes": "Web开发",
    "SQL Developers Resumes": "SQL开发",
    "Business Analyst (BA) Resumes": "业务分析师",
    "Network and Systems Administrators Resumes": "网络与系统管理",
    "Datawarehousing, ETL, Informatica Resumes": "数据仓库/ETL",
    "Business Intelligence, Business Object Resumes": "商业智能",
    "Project Manager Resumes": "项目经理",
    "Recruiter Resumes": "招聘专员",
}

SALARY_RANGES = {
    "Java Developers/Architects Resumes": {"min": 15000, "mid": 28000, "max": 50000},
    "Web Developer Resumes": {"min": 10000, "mid": 22000, "max": 40000},
    "SQL Developers Resumes": {"min": 12000, "mid": 23000, "max": 42000},
    "Business Analyst (BA) Resumes": {"min": 13000, "mid": 25000, "max": 45000},
    "Network and Systems Administrators Resumes": {"min": 10000, "mid": 20000, "max": 38000},
    "Datawarehousing, ETL, Informatica Resumes": {"min": 14000, "mid": 26000, "max": 48000},
    "Business Intelligence, Business Object Resumes": {"min": 13000, "mid": 24000, "max": 44000},
    "Project Manager Resumes": {"min": 18000, "mid": 32000, "max": 55000},
    "Recruiter Resumes": {"min": 8000, "mid": 15000, "max": 28000},
}

HIGH_VALUE_SKILLS = {
    "Kubernetes", "Docker", "AWS", "Azure", "GCP", "Terraform",
    "Spark", "Kafka", "Machine Learning", "Deep Learning",
    "Microservices", "DevOps", "CI/CD", "Spring Boot",
    "React", "Angular", "TensorFlow", "PyTorch",
    "Kubernetes", "Redis", "Elasticsearch", "Snowflake",
}

EXPERIENCE_MULTIPLIER = {
    0: 0.6,
    1: 0.7,
    2: 0.75,
    3: 0.8,
    5: 0.9,
    7: 1.0,
    10: 1.15,
    15: 1.25,
}


def load_data(filepath="data/Resume_dataset.csv"):
    df = pd.read_csv(filepath)
    return df


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s\.\-\+\#\/]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text):
    text_lower = text.lower()
    text_original = text
    found_skills = {}

    for category, skills in SKILL_DICTIONARY.items():
        found = []
        for skill in skills:
            pattern = r'\b' + skill + r'\b'
            if re.search(pattern, text_original, re.IGNORECASE):
                found.append(skill.replace("\\+", "+").replace("\\#", "#"))
        if found:
            found_skills[category] = found

    all_skills = []
    for skills in found_skills.values():
        all_skills.extend(skills)
    return found_skills, list(set(all_skills))


def extract_experience_years(text):
    patterns = [
        r"(\d+)\+?\s*[-–to]+\s*year[s]?\s*(?:of\s*)?(?:experience|exp)",
        r"(\d+)\+?\s*year[s]?\s*(?:of\s*)?(?:experience|exp)",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*year[s]?",
        r"over\s+(\d+)\s*\+?\s*year[s]?",
        r"around\s+(\d+)\s*\+?\s*year[s]?",
        r"(?<!\d)(\d{1,2})\s*年(?:以上)?(?:的)?(?:工作)?(?:开发)?(?:项目)?经验",
        r"拥有\s*(\d{1,2})\s*年",
        r"(?<!\d)(\d{1,2})\s*年(?:工作)?经历",
        r"工作\s*(?:经验|经历)\s*[:：]?\s*(\d{1,2})\s*年",
        r"(?<!\d)(\d{1,2})\+?\s*年(?:行业)?经验",
        r"(?<!\d)(\d{1,2})\s*年[^年0-9]*?(?:经验|经历|开发|工作)",
        r"(?<!\d)(\d{1,2})\s*年前",
        r"(?:经验|经历|工作)\s*[:：]?\s*(\d{1,2})\s*年",
    ]
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            val = int(m)
            if 1 <= val <= 50:
                years.append(val)

    if years:
        return max(years)
    return 0


def generate_salary_label(category, skills, experience_years):
    base = SALARY_RANGES.get(category, {"min": 10000, "mid": 20000, "max": 35000})

    high_value_count = sum(1 for s in skills if s in HIGH_VALUE_SKILLS)
    skill_bonus = min(high_value_count * 0.03, 0.2)

    exp_mult = 0.7
    for threshold, mult in sorted(EXPERIENCE_MULTIPLIER.items()):
        if experience_years >= threshold:
            exp_mult = mult

    total_skills = len(skills)
    quantity_bonus = min(total_skills * 0.005, 0.1)

    mid_salary = base["mid"] * exp_mult * (1 + skill_bonus + quantity_bonus)
    min_salary = mid_salary * 0.75
    max_salary = mid_salary * 1.35

    noise = np.random.normal(0, 0.05)
    mid_salary *= (1 + noise)

    return round(min_salary), round(mid_salary), round(max_salary)


def preprocess_data(df):
    df = df.copy()
    df["cleaned_text"] = df["Text"].apply(clean_text)
    df["skill_dict"] = df["cleaned_text"].apply(extract_skills)
    df["all_skills"] = df["skill_dict"].apply(lambda x: x[1])
    df["skill_categories"] = df["skill_dict"].apply(lambda x: x[0])
    df["skill_count"] = df["all_skills"].apply(len)
    df["experience_years"] = df["cleaned_text"].apply(extract_experience_years)

    salary_data = df.apply(
        lambda row: generate_salary_label(
            row["category"], row["all_skills"], row["experience_years"]
        ),
        axis=1,
    )
    df["salary_min"] = salary_data.apply(lambda x: x[0])
    df["salary_mid"] = salary_data.apply(lambda x: x[1])
    df["salary_max"] = salary_data.apply(lambda x: x[2])

    df["category_cn"] = df["category"].map(CATEGORY_CN)

    return df


def build_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def build_skill_feature_matrix(df):
    all_skill_names = sorted(set(s for skills in df["all_skills"] for s in skills))
    skill_to_idx = {skill: idx for idx, skill in enumerate(all_skill_names)}

    matrix = np.zeros((len(df), len(all_skill_names)))
    for i, skills in enumerate(df["all_skills"]):
        for skill in skills:
            if skill in skill_to_idx:
                matrix[i, skill_to_idx[skill]] = 1

    return matrix, all_skill_names, skill_to_idx


def encode_labels(categories):
    le = LabelEncoder()
    encoded = le.fit_transform(categories)
    return le, encoded


def get_feature_stats(df):
    stats = {
        "total_resumes": len(df),
        "categories": df["category_cn"].value_counts().to_dict(),
        "avg_skill_count": df["skill_count"].mean(),
        "avg_experience": df["experience_years"].mean(),
        "avg_salary": df["salary_mid"].mean(),
    }

    all_skills = []
    for skills in df["all_skills"]:
        all_skills.extend(skills)
    skill_freq = pd.Series(all_skills).value_counts().head(30).to_dict()
    stats["top_skills"] = skill_freq

    return stats
