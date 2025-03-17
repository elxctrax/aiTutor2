from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    context: str

# load model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased', tokenizer='distilbert-base-uncased')

@app.get("/")
def read_root():
    return {"message": "AI Tutor API is running"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    result = qa_pipeline(question=request.question, context=request.context)
    return {"answer": result['answer'], "confidence": result['score']}