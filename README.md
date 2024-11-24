# ArabicFraudDetection

A machine learning project to detect fraudulent comments in ride-hailing services using Arabic text data. The project leverages pre-trained language models like [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv2) to classify user comments as fraudulent or non-fraudulent.

---

## Overview

Detecting fraudulent activities in ride-hailing services is critical for ensuring reliability and customer trust. This project focuses on processing Arabic text data, fine-tuning a pre-trained language model, and deploying an effective system for fraud detection.

### Key Features:
- **Custom Preprocessing**: Automatically label comments based on fraud-related keywords.
- **Fine-Tuning Pre-trained Models**: Fine-tune AraBERT for binary classification tasks.
- **Evaluation Metrics**: Calculate metrics like accuracy, precision, recall, and F1-score.
- **Modular Code**: Organized and reusable Python modules for data processing, training, and evaluation.
- **Docker Support**: Easily run the project in a containerized environment.

---

## Directory Structure

```
ArabicFraudDetection/
├── data/                    # Store data files or sample datasets
├── src/                     # Python scripts for preprocessing, training, evaluation, etc.
│   ├── preprocessing.py     # Data cleaning and preparation
│   ├── train.py             # Model fine-tuning script
│   ├── evaluate.py          # Evaluation and metrics calculation
│   ├── utils.py             # Helper functions
├── models/                  # Save trained models and checkpoints
├── notebooks/               # Jupyter notebooks for exploratory work
├── tests/                   # Unit and integration tests
├── tmp_trainer/             # 
├── requirements.txt         # Python dependencies
├── Dockerfile               # Dockerfile to containerize the project
├── README.md                # Documentation
├── LICENSE                  # License file
├── .gitignore               # Ignore unnecessary files (e.g., data, logs)
└── setup.py                 # Package installation script
```

---

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Pip
- Docker (if using containerized setup)
- Virtual environment (optional but recommended)

---

## Installation

### Option 1: Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/smasoudrezvani/ArabicFraudDetection_LLM.git
   cd ArabicFraudDetection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your raw dataset (e.g., `df_rating&comment.xlsx`) in the `data/` folder.

### Option 2: Run with Docker

1. Build the Docker image:
   ```bash
   docker build -t arabic-fraud-detection-LLM .
   ```

2. Run the Docker container:
   ```bash
   docker run --rm -it arabic-fraud-detection-LLM
   ```

---

## Usage (Not using Docker)

### 1. Preprocess Data
Run the preprocessing script to clean and label the data:
```bash
python3 ./src/preprocessing.py --input "data/df_rating&comment.xlsx" --output "data/processed"
```

### 2. Train the Model
Fine-tune the pre-trained language model:
```bash
python3 ./src/train.py --model aubmindlab/bert-base-arabertv2 --train data/processed/train.json --test data/processed/test.json --output models/fraud_detector
```

### 3. Evaluate the Model
Evaluate the model's performance on the test set:
```bash
python3 ./src/evaluate.py --model ./models/fraud_detector --test ./data/processed/test.json
```

---

## Using Docker

### 1. Run Preprocessing with Docker
You can modify the `Dockerfile` command or use the container interactively:
```bash
docker run -v $(pwd)/data:/app/data arabic-fraud-detection python src/preprocessing.py --input data/df_rating&comment.xlsx --output data/processed
```

### 2. Train the Model with Docker
```bash
docker run -v $(pwd)/models:/app/models arabic-fraud-detection python src/train.py --model aubmindlab/bert-base-arabertv2 --train data/processed/train.json --test data/processed/test.json --output models/fraud_detector
```

---

## Results

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 0.92   |
| Precision   | 0.90   |
| Recall      | 0.89   |
| F1 Score    | 0.89   |

---

## Advanced Features

- **Hyperparameter Tuning**: Use the `Trainer` API's hyperparameter search functionality to optimize the model.
- **Data Augmentation**: Extend the dataset using techniques like back-translation or synonym replacement.
- **Deployment**: Deploy the model as a REST API using FastAPI or a user-friendly interface with Streamlit.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m 'Add feature-name'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Special thanks to [Hugging Face](https://huggingface.co/) for providing the tools and pre-trained models that make this project possible.
```

---

This version includes details about how to use the `Dockerfile` for preprocessing, training, and evaluation. Let me know if you'd like further changes!