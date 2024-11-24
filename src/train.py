import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def train_model(model_name, train_path, test_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(example):
        return tokenizer(example["comment"], truncation=True, padding=True)

    # Load dataset
    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("Model training complete.")
    # Save the fine-tuned model and tokenizer
    trainer.save_model("./models/fraud_detector")
    tokenizer.save_pretrained("./models/fraud_detector")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Pretrained model name (e.g., aubmindlab/bert-base-arabertv2)")
    parser.add_argument("--train", required=True, help="Path to training dataset")
    parser.add_argument("--test", required=True, help="Path to testing dataset")
    parser.add_argument("--output", required=True, help="Output directory for the model")
    args = parser.parse_args()
    train_model(args.model, args.train, args.test, args.output)
