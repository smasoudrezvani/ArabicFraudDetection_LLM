import unittest
import pandas as pd
from src.preprocessing import label_fraud, preprocess_data

class TestPreprocessing(unittest.TestCase):

    def test_label_fraud(self):
        # Test cases for fraud labeling
        fraud_comment = "السائق ما وصلني"
        non_fraud_comment = "الرحلة كانت رائعة"
        empty_comment = None

        self.assertEqual(label_fraud(fraud_comment), 1)
        self.assertEqual(label_fraud(non_fraud_comment), 0)
        self.assertEqual(label_fraud(empty_comment), 0)

    def test_preprocess_data(self):
        # Create a mock dataset
        data = {
            "comment": ["السائق ما وصلني", "الرحلة كانت رائعة", None],
            "other_column": [1, 2, 3],
        }
        df = pd.DataFrame(data)

        # Write to a temporary file
        input_path = "mock_data.xlsx"
        output_path = "mock_processed"

        df.to_excel(input_path, index=False)

        # Preprocess the data
        preprocess_data(input_path, output_path)

        # Read processed data
        train_df = pd.read_json(f"{output_path}/train.json", lines=True)
        test_df = pd.read_json(f"{output_path}/test.json", lines=True)

        # Verify the structure
        self.assertTrue("label" in train_df.columns)
        self.assertTrue("label" in test_df.columns)

        # Clean up
        import os
        os.remove(input_path)
        os.remove(f"{output_path}/train.json")
        os.remove(f"{output_path}/test.json")
        os.rmdir(output_path)

if __name__ == "__main__":
    unittest.main()
