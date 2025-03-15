from sklearn.model_selection import ParameterGrid
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.optim import AdamW

class HyperparameterTuner:
    """
    A class to perform hyperparameter tuning for a pre-trained transformer model using grid search.
    The best model is selected based on the macro F1-score.
    """

    def __init__(self, train_path, val_path, model_name, num_labels):
        """
        Initializes the HyperparameterTuner with paths to datasets, model name, and number of labels.

        Args:
            train_path (str): Path to the training dataset CSV file.
            val_path (str): Path to the validation dataset CSV file.
            model_name (str): Name of the pre-trained model to use.
            num_labels (int): Number of emotion labels in the dataset.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.model_name = model_name
        self.num_labels = num_labels
        self.train_dataset, self.val_dataset = self._load_and_prepare_datasets()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )

    def _load_and_prepare_datasets(self):
        """
        Loads and prepares the datasets from CSV files, converting emotion columns to lists of labels.

        Returns:
            tuple: A tuple containing the training and validation datasets.
        """
        df_train = pd.read_csv(self.train_path)
        df_val = pd.read_csv(self.val_path)
        emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        df_train[emotion_columns] = df_train[emotion_columns].astype('float32')
        df_val[emotion_columns] = df_val[emotion_columns].astype('float32')
        df_train['labels'] = df_train[emotion_columns].values.tolist()
        df_val['labels'] = df_val[emotion_columns].values.tolist()
        return Dataset.from_pandas(df_train[['text', 'labels']]), Dataset.from_pandas(df_val[['text', 'labels']])

    def _tokenize_dataset(self, dataset):
        """
        Tokenizes the dataset using the model's tokenizer.

        Args:
            dataset (Dataset): Dataset to tokenize.

        Returns:
            Dataset: Tokenized dataset.
        """
        return dataset.map(lambda x: self.tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)

    def _compute_metrics(self, pred):
        """
        Computes evaluation metrics (accuracy, precision, recall, F1-score).

        Args:
            pred: Model predictions.

        Returns:
            dict: Evaluation metrics.
        """
        labels = pred.label_ids
        preds = (pred.predictions >= 0.5).astype(int)  # Convert logits to binary predictions
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def _train_and_evaluate(self, params):
        """
        Trains and evaluates the model with the given hyperparameters.

        Args:
            params (dict): Hyperparameters for training.

        Returns:
            dict: Evaluation metrics.
        """
        # Update model's dropout probability
        self.model.config.hidden_dropout_prob = params['dropout_prob']

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./temp_output",
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=params['weight_decay'],
            logging_dir='./logs',
            logging_steps=500,
            report_to="none",
            save_strategy="no",
            load_best_model_at_end=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            optimizers=(AdamW(self.model.parameters(), lr=params['learning_rate']), None)
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        return eval_results

    def tune_hyperparameters(self):
        """
        Performs hyperparameter tuning using grid search and selects the best combination based on F1-score.

        Returns:
            DataFrame: A DataFrame containing the results of all hyperparameter combinations.
        """
        # Define the hyperparameter grid
        param_grid = {
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],  # Learning rate
            'weight_decay': [0.001, 0.01, 0.1],  # Weight decay
            'dropout_prob': [0.3, 0.5]  # Dropout probability
        }

        # Generate hyperparameter combinations
        grid = ParameterGrid(param_grid)

        # Tokenize datasets
        self.train_dataset = self._tokenize_dataset(self.train_dataset)
        self.val_dataset = self._tokenize_dataset(self.val_dataset)

        # Store results
        results = []

        # Iterate over hyperparameter combinations
        for params in grid:
            print(f"Testing hyperparameters: {params}")
            try:
                metrics = self._train_and_evaluate(params)
                results.append({
                    'params': params,
                    'metrics': metrics
                })
                print(f"Results for this combination: {metrics}")
            except Exception as e:
                print(f"Error with hyperparameters {params}: {e}")

        # Sort results by F1-score
        results_sorted = sorted(results, key=lambda x: x['metrics']['eval_f1'], reverse=True)

        # Display the best combination
        best_result = results_sorted[0]
        print(f"Best hyperparameters: {best_result['params']}")
        print(f"Metrics: {best_result['metrics']}")

        # Save results to a DataFrame
        results_df = pd.DataFrame(results)
        metrics_df = results_df['metrics'].apply(pd.Series)
        result_df = pd.concat([results_df['params'], metrics_df], axis=1)
        result_df = result_df.sort_values(by='eval_f1', ascending=False)

        return result_df

# Example usage
if __name__ == "__main__":
    tuner = HyperparameterTuner(
        train_path='/path/to/train.csv',
        val_path='/path/to/val.csv',
        model_name='FacebookAI/xlm-roberta-base',
        num_labels=6
    )
    results_df = tuner.tune_hyperparameters()
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("Hyperparameter tuning results saved to 'hyperparameter_tuning_results.csv'.")