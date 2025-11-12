from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from difflib import SequenceMatcher

# Load dataset
df = pd.read_csv("data/company_tech_stack_dataset_enhanced.csv")

# Create FastAPI app
app = FastAPI(
    title="Skill-Based Company Recommender API",
    description="Give your skills, get top company recommendations based on salary and stack similarity.",
    version="1.0"
)

# ✅ Allow everything (all origins, headers, and methods)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins
    allow_credentials=True,   # Allow cookies and credentials
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],      # Allow all headers
)

# Define request model
class SkillRequest(BaseModel):
    skills: list[str]

# Function to calculate similarity between user skills and company stack
def compute_similarity(row, skills):
    text = ' '.join([
        str(row.get("Primary Languages", "")),
        str(row.get("Frameworks & Libraries", "")),
        str(row.get("Databases", "")),
        str(row.get("Cloud Infrastructure", ""))
    ]).lower()
    user_text = ' '.join(skills).lower()
    return SequenceMatcher(None, text, user_text).ratio()

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Skill-Based Company Recommender API!"}

# Main recommendation endpoint
@app.post("/recommend")
def recommend_companies(request: SkillRequest):
    df["Similarity"] = df.apply(lambda row: compute_similarity(row, request.skills), axis=1)
    top10 = df.sort_values(by=["Similarity", "P50 Salary (USD)"], ascending=False).head(10)

    result = top10[[
        "Company", "Common Job Roles",
        "P25 Salary (USD)", "P50 Salary (USD)", "P75 Salary (USD)", "Similarity"
    ]].to_dict(orient="records")

    return {"input_skills": request.skills, "recommendations": result}
