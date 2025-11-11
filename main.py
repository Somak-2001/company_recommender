from src.recommender import SkillRecommender

if __name__ == "__main__":
    model = SkillRecommender("data/company_tech_stack_dataset_enhanced.csv")
    user_skills = ["Java", "Spring Boot", "PostgreSQL", "GCP"]
    recommendations = model.recommend(user_skills)
    print("\nTop 10 Recommended Companies:\n")
    print(recommendations.to_string(index=False))
    print(type(recommendations))
