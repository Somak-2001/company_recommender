import pandas as pd
from .utils import preprocess_data, jaccard_similarity

class SkillRecommender:
    def __init__(self, csv_path):
        self.df = preprocess_data(csv_path)

    def recommend(self, user_skills, top_n=10):
        user_skills = [s.strip().lower() for s in user_skills]
        self.df["Similarity"] = self.df["SkillSet"].apply(lambda x: jaccard_similarity(x, user_skills))
        results = self.df[self.df["Similarity"] > 0].sort_values(
            by=["Similarity", "P75 Salary (USD)"], ascending=False
        ).head(top_n)
        return results[["Company", "Common Job Roles", "P25 Salary (USD)", "P50 Salary (USD)", "P75 Salary (USD)", "Similarity"]]
