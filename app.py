import os
import urllib.request
import joblib
import pandas as pd
import gradio as gr
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# Download the trained model files from the first project
BASE = "https://github.com/rafaqatkhattak5/loan-approval-predictor/raw/main/"
for f in ["loan_model.pkl", "preprocessor.pkl"]:
    if not os.path.exists(f):
        urllib.request.urlretrieve(BASE + f, f)

model = joblib.load("loan_model.pkl")
prep = joblib.load("preprocessor.pkl")

def predict_loan(age: float, gender: str, marital_status: str, education: str,
                 employment_type: str, city: str, years_experience: float,
                 monthly_income: float, loan_amount: float, credit_score: float) -> dict:
    """Predict loan approval chance for one applicant.
    gender: Male or Female. marital_status: Single, Married or Divorced.
    education: Matric, Intermediate, Bachelors or Masters.
    employment_type: Salaried, Self-Employed, Business or Unemployed.
    city: Karak, Kohat, Peshawar, Lahore or Islamabad.
    Money values are in rupees. credit_score is 300 to 850."""
    df = pd.DataFrame([{
        "Age": age, "Gender": gender, "Marital_Status": marital_status,
        "Education": education, "Employment_Type": employment_type, "City": city,
        "Years_Experience": years_experience, "Monthly_Income": monthly_income,
        "Loan_Amount": loan_amount, "Credit_Score": credit_score
    }])
    prob = float(model.predict_proba(prep.transform(df))[0][1])
    return {"approved": prob >= 0.5, "probability_percent": round(prob * 100, 1)}

def max_approved_loan(age: float, gender: str, marital_status: str, education: str,
                      employment_type: str, city: str, years_experience: float,
                      monthly_income: float, credit_score: float) -> dict:
    """Find the highest loan amount (in rupees) this applicant can get approved.
    Same value options as predict_loan. Returns 0 if no tested amount is approved."""
    amounts = list(range(50000, 3000001, 50000))
    rows = [{
        "Age": age, "Gender": gender, "Marital_Status": marital_status,
        "Education": education, "Employment_Type": employment_type, "City": city,
        "Years_Experience": years_experience, "Monthly_Income": monthly_income,
        "Loan_Amount": a, "Credit_Score": credit_score
    } for a in amounts]
    probs = model.predict_proba(prep.transform(pd.DataFrame(rows)))[:, 1]
    approved = [a for a, p in zip(amounts, probs) if p >= 0.5]
    return {"max_loan_amount": max(approved) if approved else 0,
            "note": "Search limit is 3,000,000 rupees, so the real maximum may be higher."}

# The key is read from the environment, never written in the code
if not os.environ.get("GOOGLE_API_KEY"):
    from getpass import getpass
    os.environ["GOOGLE_API_KEY"] = getpass("Gemini API key: ")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_version="v1beta")

system_prompt = """You are a loan assistant for a practice AI project.
Use predict_loan to check one loan, and max_approved_loan to find the highest loan an applicant can get.
Required fields: age, gender, marital status, education, employment type, city,
years of experience, monthly income, credit score (and loan amount for predict_loan).
If any field is missing, ask the user for it. Never guess values.
Explain the result in simple words. Reply in the user's language (English, Urdu or Pashto).
Say once that this is a practice model on synthetic data, not real banking advice."""

agent = create_agent(model=llm, tools=[predict_loan, max_approved_loan],
                     system_prompt=system_prompt, checkpointer=InMemorySaver())

def chat(text, thread):
    result = agent.invoke({"messages": [{"role": "user", "content": text}]},
                          {"configurable": {"thread_id": thread}})
    return result["messages"][-1].content

def respond(message, history, request: gr.Request):
    return chat(message, request.session_hash)

demo = gr.ChatInterface(
    fn=respond,
    title="🏦 Loan AI Agent",
    description="Ask about your loan chances. Practice model on synthetic data, not real banking advice.",
)

if __name__ == "__main__":
    demo.launch()
