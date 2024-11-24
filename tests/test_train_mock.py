import unittest
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.train import train_model

class TestTrainingWithMock(unittest.TestCase):

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.Trainer.train")
    def test_train_model_with_mocks(self, mock_train, mock_model, mock_tokenizer):
        # Mock tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.__call__.return_value = {
            "input_ids": [101, 2003, 2005],
            "attention_mask": [1, 1, 1],
        }

        # Mock model
        mock_model.return_value = MagicMock()

        # Mock training process
        mock_train.return_value = None

        # Define inputs
        model_name = "mock-model"
        train_path = "mock_train.json"
        test_path = "mock_test.json"
        output_dir = "mock_model_output"

        # Create dummy JSON data for train and test
        import json
        train_data = [{"comment": "السائق لم يأتي", "label": 1}]
        test_data = [{"comment": "الرحلة كانت رائعة", "label": 0}]

        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(test_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Call train_model with mocks
        train_model(model_name, train_path, test_path, output_dir)

        # Assertions
        mock_tokenizer.assert_called_once_with(model_name)
        mock_model.assert_called_once_with(model_name, num_labels=2)
        mock_train.assert_called_once()  # Ensure training was "called"

        # Cleanup
        import os
        os.remove(train_path)
        os.remove(test_path)

if __name__ == "__main__":
    unittest.main()
