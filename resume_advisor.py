import numpy as np
from data_utils import (
    SKILL_DICTIONARY, SKILL_CATEGORY_CN, CATEGORY_CN,
    HIGH_VALUE_SKILLS, extract_skills, extract_experience_years, clean_text,
)


def get_missing_skills(user_skills, category, df):
    category_resumes = df[df["category"] == category]
    if len(category_resumes) == 0:
        return [], {}

    category_skills = []
    for skills in category_resumes["all_skills"]:
        category_skills.extend(skills)

    from collections import Counter
    skill_freq = Counter(category_skills)

    total = len(category_resumes)
    skill_prevalence = {s: c / total for s, c in skill_freq.items()}

    user_set = set(s.lower() for s in user_skills)
    missing = {}
    for skill, prevalence in skill_prevalence.items():
        if skill.lower() not in user_set and prevalence >= 0.1:
            missing[skill] = prevalence

    sorted_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)
    return sorted_missing, skill_prevalence


def get_high_salary_skills(category, df, top_n=15):
    category_resumes = df[df["category"] == category].copy()
    if len(category_resumes) == 0:
        return []

    median_salary = category_resumes["salary_mid"].median()
    high_salary = category_resumes[category_resumes["salary_mid"] >= median_salary]

    high_skills = []
    for skills in high_salary["all_skills"]:
        high_skills.extend(skills)

    from collections import Counter
    skill_freq = Counter(high_skills)
    total = len(high_salary) if len(high_salary) > 0 else 1

    ranked = [(s, c / total) for s, c in skill_freq.most_common(top_n)]
    return ranked


def check_resume_structure(text):
    suggestions = []
    text_lower = text.lower()

    structure_checks = [
        ("summary", [
            "summary", "profile", "objective", "about me",
            "概述", "简介", "个人简介", "个人总结", "自我评价", "自我介绍",
            "求职意向", "职业目标", "个人描述", "关于我",
        ], "缺少个人简介/职业概述部分"),
        ("skills", [
            "skills", "technologies", "technical skills", "tech stack", "competencies",
            "技能", "技术栈", "专业技能", "核心技能", "技术能力", "掌握技术", "技术特长",
        ], "缺少技能列表部分"),
        ("experience", [
            "experience", "professional experience", "work history", "employment",
            "工作经验", "工作经历", "职业经历", "从业经历", "工作背景", "从业经验",
        ], "缺少工作经验部分"),
        ("education", [
            "education", "academic", "degree", "qualification",
            "学历", "教育", "教育背景", "教育经历", "学术背景", "毕业院校",
        ], "缺少教育背景部分"),
        ("projects", [
            "project", "projects", "key projects", "project experience",
            "项目", "项目经验", "项目经历", "项目经验", "主要项目", "核心项目",
        ], "缺少项目经验部分"),
        ("certifications", [
            "certification", "certified", "certificate", "license",
            "证书", "认证", "资格证书", "专业认证", "资质证书",
        ], "缺少证书/认证部分（可选但有加分）"),
    ]

    for section, keywords, message in structure_checks:
        found = any(kw in text_lower for kw in keywords)
        if not found:
            suggestions.append({"type": "structure", "section": section, "message": message})

    return suggestions


def check_skill_balance(skill_dict):
    suggestions = []
    categories_present = set(skill_dict.keys())

    important_categories = ["programming_languages", "frameworks", "databases"]
    category_names = {
        "programming_languages": "编程语言",
        "frameworks": "框架与库",
        "databases": "数据库",
        "devops_tools": "DevOps工具",
        "methodologies": "方法论",
    }

    for cat in important_categories:
        if cat not in categories_present:
            suggestions.append({
                "type": "skill_gap",
                "category": cat,
                "category_cn": category_names.get(cat, cat),
                "message": f"缺少{category_names.get(cat, cat)}相关技能描述",
            })

    for cat, skills in skill_dict.items():
        if len(skills) < 2 and cat in important_categories:
            suggestions.append({
                "type": "skill_shallow",
                "category": cat,
                "category_cn": category_names.get(cat, cat),
                "message": f"{category_names.get(cat, cat)}技能描述过少（仅{len(skills)}个），建议补充",
            })

    return suggestions


def check_experience_description(text, experience_years):
    suggestions = []

    action_verbs = [
        "developed", "designed", "implemented", "managed", "led",
        "created", "built", "optimized", "improved", "delivered",
        "architected", "configured", "deployed", "integrated", "maintained",
    ]
    cn_action_verbs = [
        "负责", "主导", "设计", "开发", "实现", "搭建", "优化", "管理",
        "带领", "构建", "部署", "维护", "推动", "完成", "提升", "降低",
        "重构", "改造", "引入", "制定", "解决", "支撑", "保障",
    ]
    text_lower = text.lower()
    found_verbs = [v for v in action_verbs if v in text_lower]
    found_cn_verbs = [v for v in cn_action_verbs if v in text]

    if len(found_verbs) + len(found_cn_verbs) < 3 and experience_years > 2:
        suggestions.append({
            "type": "experience_quality",
            "message": "工作经验描述缺少有力的行为动词（如 负责/主导/设计/开发/优化），建议使用更专业的表述",
        })

    quantifiable = any(char.isdigit() for char in text)
    if not quantifiable:
        suggestions.append({
            "type": "experience_quality",
            "message": "工作经验中缺少量化成果（如团队规模、项目预算、性能提升百分比），建议补充数据支撑",
        })

    return suggestions


def generate_recommendations(text, user_skills, skill_dict, category, experience_years, df):
    recommendations = []

    missing_skills, skill_prevalence = get_missing_skills(user_skills, category, df)
    if missing_skills:
        top_missing = missing_skills[:8]
        for skill, prevalence in top_missing:
            value_tag = "🔥 高价值" if skill in HIGH_VALUE_SKILLS else ""
            recommendations.append({
                "type": "missing_skill",
                "priority": "high" if prevalence >= 0.3 else "medium",
                "skill": skill,
                "prevalence": f"{prevalence:.0%}",
                "message": f"建议补充技能: {skill}（同类岗位{prevalence:.0%}的简历包含此技能）{value_tag}",
            })

    high_salary_skills = get_high_salary_skills(category, df)
    user_set = set(s.lower() for s in user_skills)
    recommended_to_learn = [
        (s, p) for s, p in high_salary_skills
        if s.lower() not in user_set and s in HIGH_VALUE_SKILLS
    ][:5]
    for skill, prevalence in recommended_to_learn:
        recommendations.append({
            "type": "salary_boost",
            "priority": "high",
            "skill": skill,
            "prevalence": f"{prevalence:.0%}",
            "message": f"学习此技能可提升薪资: {skill}（高薪简历中{prevalence:.0%}包含此技能）🔥",
        })

    structure_suggestions = check_resume_structure(text)
    recommendations.extend(structure_suggestions)

    balance_suggestions = check_skill_balance(skill_dict)
    recommendations.extend(balance_suggestions)

    experience_suggestions = check_experience_description(text, experience_years)
    recommendations.extend(experience_suggestions)

    if experience_years == 0:
        recommendations.append({
            "type": "experience",
            "priority": "high",
            "message": "未在简历中检测到工作年限信息，建议明确标注工作年限",
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

    return recommendations


def get_resume_score(text, user_skills, skill_dict, category, experience_years, df):
    score = 0
    max_score = 100
    details = {}

    skill_score = min(len(user_skills) * 3, 24)
    score += skill_score
    details["技能丰富度"] = skill_score

    high_value = sum(1 for s in user_skills if s in HIGH_VALUE_SKILLS)
    hv_base = min(high_value * 5, 20)
    rare_bonus = 0
    if df is not None and len(df) > 0:
        cat_df = df[df["category"] == category] if category in df["category"].values else df
        from collections import Counter
        all_cat_skills = []
        for skills in cat_df["all_skills"]:
            all_cat_skills.extend(skills)
        skill_freq = Counter(all_cat_skills)
        total = len(cat_df)
        for s in user_skills:
            if s in HIGH_VALUE_SKILLS:
                prevalence = skill_freq.get(s, 0) / total if total > 0 else 1
                if prevalence < 0.15:
                    rare_bonus += 3
                elif prevalence < 0.3:
                    rare_bonus += 1.5
    rare_bonus = min(rare_bonus, 10)
    hv_total = min(hv_base + rare_bonus, 30)
    score += hv_total
    details["高价值技能"] = hv_total

    missing, _ = get_missing_skills(user_skills, category, df)
    completeness = max(0, 20 - len(missing))
    score += completeness
    details["技能完整度"] = completeness

    structure_suggestions = check_resume_structure(text)
    structure_score = max(0, 16 - len(structure_suggestions) * 2)
    score += structure_score
    details["简历结构"] = structure_score

    exp_score = min(experience_years * 2, 10)
    score += exp_score
    details["经验年限"] = exp_score

    return min(score, max_score), details
