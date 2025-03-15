from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score
import os

class EmotionDetectionModelEvaluator:
    """
    A class to evaluate multiple pre-trained transformer models for emotion detection in multilingual texts.
    This class handles loading datasets, tokenizing text, training models, and evaluating their performance.
    """

    def __init__(self, train_path, val_path, num_labels):
        """
        Initializes the EmotionDetectionModelEvaluator with paths to training and validation datasets.

        Args:
            train_path (str): Path to the training dataset CSV file.
            val_path (str): Path to the validation dataset CSV file.
            num_labels (int): Number of emotion labels in the dataset.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.num_labels = num_labels
        self.train_dataset, self.val_dataset = self._load_and_prepare_datasets()

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

    def _compute_metrics(self, p: EvalPrediction):
        """
        Computes evaluation metrics (F1 and precision) for the model predictions.

        Args:
            p (EvalPrediction): Contains predictions and true labels.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return self._labels_metrics(predictions=preds, labels=p.label_ids)

    def _labels_metrics(self, predictions, labels, threshold=0.4):
        """
        Calculates evaluation metrics for binary multi-label classification.

        Args:
            predictions (np.array): Predicted logits or probabilities.
            labels (np.array): True labels.
            threshold (float): Threshold for converting probabilities to binary labels.

        Returns:
            dict: Evaluation metrics including F1 and precision (macro and micro).
        """
        sigmoid = nn.Sigmoid()
        probs = sigmoid(torch.tensor(predictions))
        y_pred = (probs >= threshold).float().numpy()
        y_true = labels
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro')
        }

    def _train_and_evaluate(self, train_dataset, val_dataset, model, model_name):
        """
        Trains and evaluates a model on the provided datasets.

        Args:
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            model: Pre-trained model to be fine-tuned.
            model_name (str): Name of the model for identification.

        Returns:
            dict: Evaluation metrics for the model.
        """
        training_args = TrainingArguments(
            output_dir="./temp_output",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy='epoch',
            save_strategy='no',
            load_best_model_at_end=False,
            logging_dir=None,
            logging_steps=500
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        trainer.train()
        metrics = trainer.evaluate()
        metrics['model_name'] = model_name
        return metrics

    def evaluate_models(self, models_to_test):
        """
        Evaluates a list of models and selects the best one based on F1 macro score.

        Args:
            models_to_test (list): List of model names to be evaluated.

        Returns:
            DataFrame: A DataFrame containing the evaluation results for all models.
        """
        best_model = None
        best_f1 = 0
        results = []
        for model_name in models_to_test:
            try:
                print(f"Testing model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=self.num_labels)
                tokenized_train = self.train_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
                tokenized_val = self.val_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
                metrics = self._train_and_evaluate(tokenized_train, tokenized_val, model, model_name)
                results.append(metrics)
                if metrics['eval_f1_macro'] > best_f1:
                    best_f1 = metrics['eval_f1_macro']
                    best_model = model
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
            finally:
                del model
                torch.cuda.empty_cache()
        sorted_results = sorted(results, key=lambda x: x['eval_f1_macro'], reverse=True)
        return pd.DataFrame(sorted_results)

# Example usage
if __name__ == "__main__":
    evaluator = EmotionDetectionModelEvaluator(
        train_path='/path/to/train.csv',
        val_path='/path/to/val.csv',
        num_labels=6
    )
    models = ['FacebookAI/xlm-roberta-base', 'google-bert/bert-base-multilingual-cased', 'distilbert/distilbert-base-multilingual-cased', 'microsoft/mdeberta-v3-base', 'ai4bharat/IndicBERTv2-MLM-only']
    results_df = evaluator.evaluate_models(models)
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("Evaluation results saved to 'model_evaluation_results.csv'.")