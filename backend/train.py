from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, TrainingArguments, Trainer

# load data
def load_data():
    dataset = load_dataset("json", data_files={"train": "train_data.json"}) # call train_data.json file
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess(examples):
        inputs = tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512)
        return inputs
    
    dataset = dataset.map(preprocess, batched=True)
    return dataset, tokenizer

# loading the pretrained model
def train():
    dataset, tokenizer = load_data()
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased") 


    # define all arguements 
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    train()