import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path)
    tech_cols = ["Primary Languages", "Frameworks & Libraries", "Databases", "Cloud Infrastructure"]
    for col in tech_cols:
        df[col] = df[col].fillna("").str.lower().str.split(",")
    df["SkillSet"] = df[tech_cols].sum(axis=1)
    df["SkillSet"] = df["SkillSet"].apply(lambda x: [s.strip() for s in x if s.strip()])
    return df

def jaccard_similarity(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0
