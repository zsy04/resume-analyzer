import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import subprocess
import sys

from data_utils import (
    get_feature_stats, extract_skills, extract_experience_years,
    clean_text, SKILL_CATEGORY_CN, CATEGORY_CN, SALARY_RANGES, HIGH_VALUE_SKILLS,
)
from model import predict_category, predict_salary
from resume_advisor import generate_recommendations, get_resume_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")

st.set_page_config(page_title="智能简历分析系统", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeIn 0.6s ease-out; }

    .stApp {
        background: linear-gradient(180deg, #EDF2FA 0%, #F7FAFF 50%, #EDF2FA 100%) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #D6E4F5 0%, #E8F0FA 100%) !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #4A90D9 0%, #357ABD 50%, #2E6BA6 100%);
        padding: 1.2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(74, 144, 217, 0.25);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(74, 144, 217, 0.35);
    }
    .metric-card h2 { margin: 0; font-size: 1.8em; color: white !important; }
    .metric-card p { margin: 0; font-size: 0.9em; opacity: 0.92; color: white !important; }

    .recommendation-item {
        background: rgba(74, 144, 217, 0.06);
        border-left: 4px solid #4A90D9;
        padding: 10px 16px;
        margin: 6px 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.92em;
        color: #1A2B4A;
    }
    .recommendation-item.high {
        border-left-color: #E74C3C;
        background: rgba(231, 76, 60, 0.06);
    }
    .recommendation-item.medium {
        border-left-color: #F39C12;
        background: rgba(243, 156, 18, 0.06);
    }
    .recommendation-item.low {
        border-left-color: #27AE60;
        background: rgba(39, 174, 96, 0.06);
    }

    .skill-tag {
        display: inline-block;
        background: rgba(74, 144, 217, 0.1);
        border: 1px solid rgba(74, 144, 217, 0.25);
        border-radius: 20px;
        padding: 4px 14px;
        margin: 3px;
        font-size: 0.85em;
        color: #2E6BA6;
    }
    .skill-tag.high-value {
        background: rgba(231, 76, 60, 0.1);
        border-color: rgba(231, 76, 60, 0.3);
        color: #C0392B;
    }

    .score-circle {
        width: 120px; height: 120px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 2em; font-weight: bold; color: white;
        margin: 0 auto;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4A90D9, #357ABD);
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(74, 144, 217, 0.4);
    }

    h1, h2, h3 {
        color: #1A2B4A !important;
    }
    h1 {
        background: linear-gradient(135deg, #2E6BA6, #4A90D9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(74, 144, 217, 0.06);
        border-radius: 8px 8px 0 0;
        color: #2E6BA6;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(74, 144, 217, 0.15) !important;
        color: #1A2B4A !important;
        border-bottom: 3px solid #4A90D9 !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1A2B4A !important;
        -webkit-text-fill-color: #1A2B4A !important;
    }

    .stMarkdown, .stText {
        color: #1A2B4A;
    }

    .stAlert {
        border-radius: 10px;
    }

    .stSpinner > div {
        border-color: #4A90D9 transparent transparent transparent;
    }

    .stProgress > div > div > div {
        background-color: #4A90D9;
    }

    div[data-testid="stVerticalBlock"] > div:has(> div.metric-card) {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


DATA_PATH = os.path.join(BASE_DIR, "data", "Resume_dataset.csv")


def download_dataset():
    if os.path.exists(DATA_PATH):
        return True
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    st.info("正在下载数据集...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="ZSY04/resume-dataset",
            filename="Resume_dataset.csv",
            repo_type="dataset",
            local_dir=os.path.dirname(DATA_PATH),
        )
        return True
    except Exception:
        try:
            import urllib.request
            url = "https://huggingface.co/datasets/ZSY04/resume-dataset/resolve/main/Resume_dataset.csv"
            urllib.request.urlretrieve(url, DATA_PATH)
            return True
        except Exception as e:
            st.error(f"数据集下载失败: {e}")
            return False


def ensure_cache():
    parquet_path = os.path.join(CACHE_DIR, "processed_data.parquet")
    if os.path.exists(parquet_path):
        return True
    if not download_dataset():
        st.stop()
    st.warning("首次运行需要预处理数据并训练模型，请稍候...")
    preprocess_script = os.path.join(BASE_DIR, "preprocess.py")
    python_exe = sys.executable
    result = subprocess.run(
        [python_exe, preprocess_script],
        capture_output=True, text=True, cwd=BASE_DIR,
    )
    if result.returncode != 0:
        st.error(f"预处理失败: {result.stderr[-500:]}")
        st.stop()
    return True


@st.cache_resource
def load_cached_data():
    ensure_cache()
    df = pd.read_parquet(os.path.join(CACHE_DIR, "processed_data.parquet"))
    return df


@st.cache_resource
def load_cached_models():
    ensure_cache()
    return {
        "tfidf_vectorizer": joblib.load(os.path.join(CACHE_DIR, "tfidf_vectorizer.joblib")),
        "label_encoder": joblib.load(os.path.join(CACHE_DIR, "label_encoder.joblib")),
        "classifier": joblib.load(os.path.join(CACHE_DIR, "classifier.joblib")),
        "regressor": joblib.load(os.path.join(CACHE_DIR, "regressor.joblib")),
        "scaler": joblib.load(os.path.join(CACHE_DIR, "scaler.joblib")),
        "skill_names": joblib.load(os.path.join(CACHE_DIR, "skill_names.joblib")),
        "skill_to_idx": joblib.load(os.path.join(CACHE_DIR, "skill_to_idx.joblib")),
        "skill_importance": joblib.load(os.path.join(CACHE_DIR, "skill_importance.joblib")),
        "class_metrics": joblib.load(os.path.join(CACHE_DIR, "class_metrics.joblib")),
        "salary_metrics": joblib.load(os.path.join(CACHE_DIR, "salary_metrics.joblib")),
    }


EXAMPLE_RESUMES = {
    "java": """张三 | Java高级开发工程师
联系方式：zhangsan@email.com | 138-0000-1234

个人简介：
拥有8年Java后端开发经验，专注于分布式系统架构设计与高并发场景优化。

工作经历：
1. 某互联网大厂 | 高级Java开发工程师 | 2019-至今
   - 负责核心交易系统微服务架构设计，日均处理订单量500万+
   - 使用Spring Boot + Spring Cloud构建微服务集群，QPS提升300%
   - 基于Redis集群实现分布式缓存方案，缓存命中率达99.5%
   - 引入Kafka消息队列实现异步处理，系统响应时间降低60%
   - 使用Docker + Kubernetes进行容器化部署与运维

2. 某科技公司 | Java开发工程师 | 2016-2019
   - 参与ERP系统开发，使用Spring MVC + MyBatis技术栈
   - 负责MySQL数据库设计与优化，慢查询优化后性能提升200%
   - 编写单元测试与集成测试，代码覆盖率达85%

技能：
编程语言：Java, Python, SQL
框架：Spring Boot, Spring Cloud, MyBatis, Hibernate
数据库：MySQL, Redis, MongoDB
工具：Git, Maven, Docker, Kubernetes, Jenkins
消息队列：Kafka, RabbitMQ
操作系统：Linux

教育背景：
某大学 | 计算机科学与技术 | 硕士 | 2014-2016""",

    "frontend": """李四 | 前端开发工程师
联系方式：lisi@email.com | 139-0000-5678

个人简介：
3年前端开发经验，热衷于构建优秀的用户交互体验，熟悉现代前端技术栈。

工作经历：
1. 某创业公司 | 前端开发工程师 | 2022-至今
   - 使用React + TypeScript重构公司主站，页面加载速度提升40%
   - 基于Ant Design搭建后台管理系统，支持权限管理与数据可视化
   - 使用Redux Toolkit进行状态管理，优化组件间数据流
   - 实现响应式布局，适配移动端与PC端

2. 某外包公司 | 初级前端开发 | 2021-2022
   - 使用Vue.js开发企业官网与营销活动页面
   - 使用Webpack进行项目构建优化，打包体积减少35%
   - 编写Sass/LESS样式，配合UI设计师还原设计稿

技能：
编程语言：JavaScript, TypeScript, HTML5, CSS3
框架：React, Vue.js, Next.js
状态管理：Redux, Vuex
UI库：Ant Design, Element UI
工具：Webpack, Vite, Git, npm
其他：RESTful API, Axios, ECharts

教育背景：
某大学 | 软件工程 | 本科 | 2017-2021""",

    "data": """王五 | 数据分析师
联系方式：wangwu@email.com | 137-0000-9012

个人简介：
5年数据分析与挖掘经验，擅长从海量数据中提取业务洞察，驱动增长决策。

工作经历：
1. 某电商平台 | 高级数据分析师 | 2020-至今
   - 搭建用户行为分析体系，覆盖DAU/留存/转化等核心指标
   - 使用Python + Pandas处理TB级日志数据，产出自动化日报
   - 基于SQL从Hive数据仓库提取数据，构建用户画像标签体系
   - 使用Tableau搭建数据看板，支持业务团队自助式数据分析
   - 通过A/B测试优化推荐算法，点击率提升15%

2. 某金融公司 | 数据分析师 | 2018-2020
   - 使用Excel和SQL进行日常数据报表制作与分析
   - 建立信用风险评估模型，坏账率降低8%
   - 使用R语言进行统计建模与假设检验

技能：
编程语言：Python, SQL, R
数据处理：Pandas, NumPy, Spark
可视化：Tableau, Matplotlib, ECharts
机器学习：Scikit-learn, XGBoost
数据库：MySQL, Hive, PostgreSQL
工具：Jupyter Notebook, Git, Airflow
分析方法：A/B测试, 回归分析, 聚类分析

教育背景：
某大学 | 统计学 | 硕士 | 2016-2018""",
}


def main():
    st.title("🧠 智能简历分析系统")
    st.markdown("---")

    with st.spinner("正在加载预计算数据..."):
        df = load_cached_data()
        models = load_cached_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 数据概览", "🔍 简历分析", "💰 薪资预测", "💡 修改建议"
    ])

    with tab1:
        st.header("📊 数据集概览")
        stats = get_feature_stats(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><h2>{stats["total_resumes"]}</h2><p>简历总数</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h2>{len(stats["categories"])}</h2><p>职位类别</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h2>{stats["avg_skill_count"]:.1f}</h2><p>平均技能数</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h2>{stats["avg_salary"]:.0f}</h2><p>平均薪资(元/月)</p></div>', unsafe_allow_html=True)

        st.markdown("### 📋 类别分布")
        cat_df = pd.DataFrame(
            list(stats["categories"].items()),
            columns=["类别", "数量"]
        )
        fig_cat = px.bar(cat_df, x="类别", y="数量", color="类别",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_cat.update_layout(showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cat, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🔥 热门技能 TOP 5")
            skill_df = pd.DataFrame(
                list(stats["top_skills"].items())[:5],
                columns=["技能", "出现次数"]
            )
            fig_skill = px.bar(skill_df, x="出现次数", y="技能", orientation="h",
                              color="出现次数", color_continuous_scale="Blues")
            fig_skill.update_layout(height=500, yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_skill, use_container_width=True)

        with col_b:
            st.markdown("### 📈 技能重要性 TOP 5")
            if models["skill_importance"]:
                imp_df = pd.DataFrame(models["skill_importance"][:5], columns=["技能", "重要性"])
                fig_imp = px.bar(imp_df, x="重要性", y="技能", orientation="h",
                                color="重要性", color_continuous_scale="Blues")
                fig_imp.update_layout(height=500, yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("### 🤖 模型性能")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**分类模型**")
            st.write(f"准确率: **{models['class_metrics']['accuracy']:.2%}**")
        with col_m2:
            st.markdown("**薪资预测模型**")
            sm = models["salary_metrics"]
            st.write(f"R²: **{sm['R2']:.4f}** | RMSE: **{sm['RMSE']:.0f}** | MAE: **{sm['MAE']:.0f}**")

    with tab2:
        st.header("🔍 简历智能分析")

        input_method = st.radio("选择输入方式", ["使用示例简历", "粘贴简历文本", "从数据集选择示例"], horizontal=True)

        resume_text = ""
        if input_method == "使用示例简历":
            example_key = st.selectbox("选择示例", list(EXAMPLE_RESUMES.keys()),
                                       format_func=lambda k: {"java": "Java高级开发工程师", "frontend": "前端开发工程师", "data": "数据分析师"}[k],
                                       key="tab2_example")
            resume_text = EXAMPLE_RESUMES[example_key]
            with st.expander("📋 查看示例简历内容"):
                st.text(resume_text)
        elif input_method == "粘贴简历文本":
            resume_text = st.text_area("请粘贴简历内容", height=250,
                                       placeholder="在此粘贴您的简历全文...")
        else:
            sample_idx = st.selectbox("选择示例简历",
                                      range(len(df)),
                                      format_func=lambda i: f"[{df.iloc[i]['category_cn']}] {df.iloc[i]['job_title']}")
            resume_text = df.iloc[sample_idx]["Text"]

        if st.button("🚀 开始分析", key="analyze_btn") and resume_text:
            with st.spinner("正在分析简历..."):
                cleaned = clean_text(resume_text)
                skill_dict, all_skills = extract_skills(cleaned)
                experience_years = extract_experience_years(cleaned)

                category, confidence = predict_category(
                    models["classifier"], models["tfidf_vectorizer"],
                    models["label_encoder"], cleaned, skills=all_skills
                )
                category_cn = CATEGORY_CN.get(category, category)

                st.markdown("---")
                st.markdown("### 📋 分析结果")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><h2>{category_cn}</h2><p>职位分类</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h2>{confidence:.1%}</h2><p>分类置信度</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h2>{experience_years}年</h2><p>工作经验</p></div>', unsafe_allow_html=True)

                st.markdown("### 🏷️ 提取的技能标签")
                for cat, skills in skill_dict.items():
                    cat_cn = SKILL_CATEGORY_CN.get(cat, cat)
                    st.markdown(f"**{cat_cn}**")
                    skill_tags = ""
                    for s in skills:
                        cls = 'high-value' if s in HIGH_VALUE_SKILLS else ''
                        skill_tags += f'<span class="skill-tag {cls}">{s}</span>'
                    st.markdown(skill_tags, unsafe_allow_html=True)

                st.markdown(f"**共提取 {len(all_skills)} 个技能**")

                if all_skills:
                    st.markdown("### 🕸️ 技能雷达图")
                    cat_counts = {SKILL_CATEGORY_CN.get(k, k): len(v) for k, v in skill_dict.items()}
                    if cat_counts:
                        fig_radar = go.Figure(data=go.Scatterpolar(
                            r=list(cat_counts.values()),
                            theta=list(cat_counts.keys()),
                            fill="toself",
                            line_color="#4A90D9",
                            fillcolor="rgba(74, 144, 217, 0.2)",
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True)),
                            showlegend=False,
                            height=400,
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.header("💰 薪资预测")

        input_method_2 = st.radio("选择输入方式", ["使用示例简历", "粘贴简历文本", "从数据集选择示例"], horizontal=True, key="salary_input")

        salary_text = ""
        if input_method_2 == "使用示例简历":
            example_key_2 = st.selectbox("选择示例", list(EXAMPLE_RESUMES.keys()),
                                         format_func=lambda k: {"java": "Java高级开发工程师", "frontend": "前端开发工程师", "data": "数据分析师"}[k],
                                         key="tab3_example")
            salary_text = EXAMPLE_RESUMES[example_key_2]
            with st.expander("📋 查看示例简历内容"):
                st.text(salary_text)
        elif input_method_2 == "粘贴简历文本":
            salary_text = st.text_area("请粘贴简历内容", height=200,
                                       placeholder="在此粘贴您的简历全文...", key="salary_text")
        else:
            sample_idx_2 = st.selectbox("选择示例简历",
                                        range(len(df)),
                                        format_func=lambda i: f"[{df.iloc[i]['category_cn']}] {df.iloc[i]['job_title']}",
                                        key="salary_sample")
            salary_text = df.iloc[sample_idx_2]["Text"]

        if st.button("💰 预测薪资", key="salary_btn") and salary_text:
            with st.spinner("正在预测薪资..."):
                cleaned = clean_text(salary_text)
                skill_dict, all_skills = extract_skills(cleaned)
                experience_years = extract_experience_years(cleaned)

                category, confidence = predict_category(
                    models["classifier"], models["tfidf_vectorizer"],
                    models["label_encoder"], cleaned, skills=all_skills
                )
                category_cn = CATEGORY_CN.get(category, category)

                skill_vec = np.zeros(len(models["skill_names"]))
                for skill in all_skills:
                    if skill in models["skill_to_idx"]:
                        skill_vec[models["skill_to_idx"][skill]] = 1

                predicted_salary = predict_salary(
                    models["regressor"], models["scaler"],
                    skill_vec, experience_years, len(all_skills)
                )

                salary_range = SALARY_RANGES.get(category, {"min": 10000, "mid": 20000, "max": 35000})

                st.markdown("---")
                st.markdown(f"### 💵 薪资预测结果 — {category_cn}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#27AE60,#2ECC71)"><h2>¥{predicted_salary:,}</h2><p>预测月薪</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#F39C12,#E67E22)"><h2>¥{salary_range["min"]:,} - ¥{salary_range["max"]:,}</h2><p>行业薪资范围</p></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#4A90D9,#357ABD)"><h2>{experience_years}年 / {len(all_skills)}技能</h2><p>经验/技能数</p></div>', unsafe_allow_html=True)

                st.markdown("### 📊 技能对薪资的贡献度")
                if models["skill_importance"]:
                    user_skill_imp = [
                        (s, imp) for s, imp in models["skill_importance"]
                        if s in all_skills
                    ]
                    if user_skill_imp:
                        imp_df = pd.DataFrame(user_skill_imp[:5], columns=["技能", "贡献度"])
                        fig_imp = px.bar(imp_df, x="贡献度", y="技能", orientation="h",
                                        color="贡献度", color_continuous_scale="Blues")
                        fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("### 📈 同类别薪资分布")
                cat_df = df[df["category"] == category]
                fig_dist = px.histogram(cat_df, x="salary_mid", nbins=30,
                                       title=f"{category_cn} 薪资分布",
                                       color_discrete_sequence=["#4A90D9"])
                fig_dist.add_vline(x=predicted_salary, line_dash="dash", line_color="#E74C3C",
                                   annotation_text=f"预测: ¥{predicted_salary:,}")
                fig_dist.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)

    with tab4:
        st.header("💡 简历修改建议")

        input_method_3 = st.radio("选择输入方式", ["使用示例简历", "粘贴简历文本", "从数据集选择示例"], horizontal=True, key="adv_input")

        adv_text = ""
        if input_method_3 == "使用示例简历":
            example_key_3 = st.selectbox("选择示例", list(EXAMPLE_RESUMES.keys()),
                                         format_func=lambda k: {"java": "Java高级开发工程师", "frontend": "前端开发工程师", "data": "数据分析师"}[k],
                                         key="tab4_example")
            adv_text = EXAMPLE_RESUMES[example_key_3]
            with st.expander("📋 查看示例简历内容"):
                st.text(adv_text)
        elif input_method_3 == "粘贴简历文本":
            adv_text = st.text_area("请粘贴简历内容", height=200,
                                    placeholder="在此粘贴您的简历全文...", key="adv_text")
        else:
            sample_idx_3 = st.selectbox("选择示例简历",
                                        range(len(df)),
                                        format_func=lambda i: f"[{df.iloc[i]['category_cn']}] {df.iloc[i]['job_title']}",
                                        key="adv_sample")
            adv_text = df.iloc[sample_idx_3]["Text"]

        if st.button("💡 生成修改建议", key="adv_btn") and adv_text:
            with st.spinner("正在生成建议..."):
                cleaned = clean_text(adv_text)
                skill_dict, all_skills = extract_skills(cleaned)
                experience_years = extract_experience_years(cleaned)

                category, confidence = predict_category(
                    models["classifier"], models["tfidf_vectorizer"],
                    models["label_encoder"], cleaned, skills=all_skills
                )
                category_cn = CATEGORY_CN.get(category, category)

                score, score_details = get_resume_score(
                    adv_text, all_skills, skill_dict, category, experience_years, df
                )

                recommendations = generate_recommendations(
                    adv_text, all_skills, skill_dict, category, experience_years, df
                )

                st.markdown("---")

                col_score, col_details = st.columns([1, 2])
                with col_score:
                    st.markdown("### 📝 简历评分")
                    if score >= 80:
                        color = "#27AE60"
                    elif score >= 60:
                        color = "#F39C12"
                    else:
                        color = "#E74C3C"

                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div class="score-circle" style="background: conic-gradient({color} {score}%, #D6E4F5 {score}%);">
                            <span>{score}</span>
                        </div>
                        <p style="margin-top:10px; color:#6B7C93;">满分 100 分</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_details:
                    st.markdown("### 📊 评分明细")
                    for dim, val in score_details.items():
                        pct = val / 30 * 100 if dim == "技能丰富度" else val / 20 * 100 if dim == "高价值技能" else val / 20 * 100 if dim == "技能完整度" else val / 15 * 100
                        st.progress(min(int(pct), 100), text=f"{dim}: {val:.0f}分")

                st.markdown("---")
                st.markdown(f"### 💡 修改建议 — 目标岗位: {category_cn}")

                high_count = sum(1 for r in recommendations if r.get("priority") == "high")
                medium_count = sum(1 for r in recommendations if r.get("priority") == "medium")
                low_count = sum(1 for r in recommendations if r.get("priority") == "low")

                col_h, col_m, col_l = st.columns(3)
                with col_h:
                    st.markdown(f"🔴 高优先级: **{high_count}** 条")
                with col_m:
                    st.markdown(f"🟡 中优先级: **{medium_count}** 条")
                with col_l:
                    st.markdown(f"🟢 低优先级: **{low_count}** 条")

                st.markdown("---")
                for rec in recommendations:
                    priority = rec.get("priority", "low")
                    message = rec.get("message", "")
                    st.markdown(f'<div class="recommendation-item {priority}">{message}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🎯 推荐学习路径")
                missing_skill_recs = [r for r in recommendations if r["type"] == "missing_skill"]
                salary_boost_recs = [r for r in recommendations if r["type"] == "salary_boost"]

                if salary_boost_recs:
                    st.markdown("**🔥 高薪技能提升（学习后可显著提升薪资）**")
                    for rec in salary_boost_recs[:5]:
                        st.markdown(f'- {rec["message"]}')

                if missing_skill_recs:
                    st.markdown("**📚 补充行业通用技能**")
                    for rec in missing_skill_recs[:5]:
                        st.markdown(f'- {rec["message"]}')


if __name__ == "__main__":
    main()
