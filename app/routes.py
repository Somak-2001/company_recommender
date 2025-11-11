import os
import sys
# Ensure project root is on sys.path so absolute imports like `src.recommender` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from src.recommender import SkillRecommender

app = Flask(__name__)
model = SkillRecommender("data/company_tech_stack_dataset_enhanced.csv")

@app.route("/", methods=["GET"])
def home():
    return {"message": "Skill-based Company Recommender API is running!"}

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    skills = data.get("skills", [])
    if not skills:
        return jsonify({"error": "No skills provided"}), 400

    results = model.recommend(skills)
    return jsonify(results.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
