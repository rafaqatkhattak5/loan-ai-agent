# Loan AI Agent

An AI agent that answers loan questions in plain language (English, Urdu, Pashto). It uses Gemini to understand the user and calls my trained Random Forest model as a tool.

## Tools
- predict_loan: approval chance for one applicant
- max_approved_loan: highest loan amount that gets approved

## Tech
Python, LangChain, Gemini, Gradio, scikit-learn

## Run
1. pip install -r requirements.txt
2. Set the environment variable GOOGLE_API_KEY
3. python app.py

Model: https://github.com/rafaqatkhattak5/loan-approval-predictor
Practice project on synthetic data, not real banking advice.
