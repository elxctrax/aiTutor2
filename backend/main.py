from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import DistilBertForQuestionAnswering
from transformers import pipeline

def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer) # pipeline
    return qa_pipeline

def main():
    qa_pipeline = load_model() # load the model and the pipeline

    context = "Artificial Intelligence is a field of computer science that aims to create intelligent machines." # quick test for Q/A
    question = "What is AI?"

    result = qa_pipeline(context=context, question=question) # define result

    print(f"Answer: {result['answer']} (Score: {result['score']:.4f})")


if __name__ == '__main__':
    main()
