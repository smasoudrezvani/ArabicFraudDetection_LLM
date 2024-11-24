import os
import json
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def create_directory(path):
    """
    Create a directory if it doesn't exist.
    Args:
        path (str): The path to the directory to be created.
    """
    os.makedirs(path, exist_ok=True)
    logging.info(f"Directory created at: {path}")


def save_json(data, file_path):
    """
    Save a dictionary as a JSON file.
    Args:
        data (dict): Data to save.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data saved to JSON file: {file_path}")


def load_json(file_path):
    """
    Load a dictionary from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict: Loaded JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Data loaded from JSON file: {file_path}")
    return data


def compute_metrics(predictions, labels):
    """
    Compute evaluation metrics: accuracy, precision, recall, and F1 score.
    Args:
        predictions (list): Predicted labels.
        labels (list): True labels.
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    logging.info(f"Metrics computed: {metrics}")
    return metrics


def setup_logging(log_file="logfile.log"):
    """
    Set up logging to log both to console and a file.
    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")
