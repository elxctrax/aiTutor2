from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# load model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased', tokenizer='distilbert-base-uncased')

@app.get("/")
def read_root():
    return {"message": "AI Tutor API is running"}

@app.post("/ask")
def ask_question(question: str, context: str):
    result = qa_pipeline(question=question, context=context)
    return {"answer": result['answer'], "confidence": result['score']}