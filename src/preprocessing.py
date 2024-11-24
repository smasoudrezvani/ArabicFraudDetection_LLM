import pandas as pd
import os
from sklearn.model_selection import train_test_split
import datasets
# from datasets import dataset_utils

word_list = [
    "تركني", "لا يأتي", "لم أكن", "لم ياتي", "لم يحضر", "لم يصل",
    "لن ياخذني", "ما اجا", "ما اجاني", "ما اجة", "ما اجه", "ما اجى",
    "ما اخذني", "ما حضر", "ما وصلني", "مااجاني", "ماجاني", "ماوصل", "موصل"
]

def label_fraud(comment):
    if not isinstance(comment, str):
        return 0
    for word in word_list:
        if word in comment:
            return 1
    return 0


def preprocess_data(input_path, output_path):
    # Load the dataset
    df = pd.read_excel(input_path)
    df['label'] = df['comment'].apply(label_fraud)
    df = df.dropna(subset=['comment'])

    # Split into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save to JSON
    train.to_json(f"{output_path}/train.json", orient="records", lines=True)
    test.to_json(f"{output_path}/test.json", orient="records", lines=True)
    print(f"Data preprocessing complete. Files saved in: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataset file path")
    parser.add_argument("--output", required=True, help="Output directory for processed files")
    args = parser.parse_args()
    preprocess_data(args.input, args.output)

