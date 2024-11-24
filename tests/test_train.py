import unittest
from transformers import AutoTokenizer
from src.train import train_model

class TestTraining(unittest.TestCase):

    def test_tokenizer(self):
        model_name = "aubmindlab/bert-base-arabertv2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sample_text = "هذا تعليق تجريبي"

        # Tokenize and check keys
        tokens = tokenizer(sample_text, truncation=True, padding=True)
        self.assertTrue("input_ids" in tokens)
        self.assertTrue("attention_mask" in tokens)

    def test_train_model(self):
        # Mock inputs
        model_name = "aubmindlab/bert-base-arabertv2"
        train_path = "mock_train.json"
        test_path = "mock_test.json"
        output_dir = "mock_model"

        # Create dummy data
        import json
        train_data = [{"comment": "السائق لم يأتي", "label": 1}]
        test_data = [{"comment": "الرحلة كانت رائعة", "label": 0}]

        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(test_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Train model
        train_model(model_name, train_path, test_path, output_dir)

        # Check if model files are created
        import os
        self.assertTrue(os.path.exists(output_dir))

        # Clean up
        os.remove(train_path)
        os.remove(test_path)
        import shutil
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    unittest.main()
