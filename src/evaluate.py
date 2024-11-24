from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset

def evaluate_model(model_path, test_path):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load the test dataset
    dataset = load_dataset("json", data_files={"test": test_path})
    tokenized_test = dataset["test"].map(lambda x: tokenizer(x["comment"], truncation=True, padding=True), batched=True)

    # Initialize Trainer
    trainer = Trainer(model=model)

    # Get predictions
    predictions = trainer.predict(tokenized_test)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # Calculate metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test", required=True, help="Path to test dataset")
    args = parser.parse_args()
    evaluate_model(args.model, args.test)