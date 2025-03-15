from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score
from torch.optim import AdamW
from itertools import combinations

class CustomLossEvaluator:
    """
    A class to evaluate custom loss function combinations for a pre-trained transformer model.
    The best combination is selected based on the macro F1-score.
    """

    def __init__(self, train_path, val_path, model_name, num_labels):
        """
        Initializes the CustomLossEvaluator with paths to datasets, model name, and number of labels.

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
            ignore_mismatched_sizes=True,
            num_labels=self.num_labels
        )
        self.loss_functions = self._define_loss_functions()

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

    def _define_loss_functions(self):
        """
        Defines the custom loss functions to be evaluated.

        Returns:
            dict: A dictionary of loss functions.
        """
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction

            def forward(self, inputs, targets):
                BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
                pt = torch.exp(-BCE_loss)  # Estimated probability
                F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
                if self.reduction == 'mean':
                    return F_loss.mean()
                elif self.reduction == 'sum':
                    return F_loss.sum()
                return F_loss

        class LabelSmoothingLoss(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing

            def forward(self, inputs, targets):
                num_classes = inputs.size(1)
                smooth_targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
                log_probs = torch.log_softmax(inputs, dim=-1)
                return -(smooth_targets * log_probs).sum(dim=-1).mean()

        return {
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
            "MSELoss": nn.MSELoss(),
            "CrossEntropyLoss": nn.CrossEntropyLoss(),
            "FocalLoss": FocalLoss(alpha=0.25, gamma=2.0),
            "LabelSmoothingLoss": LabelSmoothingLoss(smoothing=0.1)
        }

    def _compute_metrics(self, p: EvalPrediction):
        """
        Computes evaluation metrics (F1 and precision, macro and micro).

        Args:
            p (EvalPrediction): Contains predictions and true labels.

        Returns:
            dict: Evaluation metrics.
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        sigmoid = nn.Sigmoid()
        probs = sigmoid(torch.tensor(preds))
        y_pred = (probs >= 0.4).float().numpy()
        y_true = p.label_ids
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro')
        }

    def _train_and_evaluate(self, loss_fns, weight_decay, lr, dropout_prob):
        """
        Trains and evaluates the model with the given loss functions and hyperparameters.

        Args:
            loss_fns (list): List of loss functions to use.
            weight_decay (float): Weight decay for the optimizer.
            lr (float): Learning rate for the optimizer.
            dropout_prob (float): Dropout probability for the model.

        Returns:
            dict: Evaluation metrics.
        """
        # Update model's dropout probability
        self.model.config.hidden_dropout_prob = dropout_prob

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./temp_output",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy='epoch',
            save_strategy='no',
            load_best_model_at_end=False,
            logging_dir=None,
            logging_steps=500,
            weight_decay=weight_decay  # Use the provided weight_decay
        )

        # Trainer
        trainer = self.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
            loss_fns=loss_fns,
            optimizers=(AdamW(self.model.parameters(), lr=lr), None)  # Use the provided learning rate
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        metrics = trainer.evaluate()
        return metrics

    class CustomTrainer(Trainer):
        """
        Custom Trainer class to handle combined loss functions.
        """
        def __init__(self, *args, loss_fns=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fns = loss_fns

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            total_loss = 0
            for loss_fn in self.loss_fns:
                total_loss += loss_fn(logits, labels.float())
            avg_loss = total_loss / len(self.loss_fns)
            return (avg_loss, outputs) if return_outputs else avg_loss

    def evaluate_loss_combinations(self, weight_decay, lr, dropout_prob):
        """
        Evaluates all combinations of loss functions with the given hyperparameters.

        Args:
            weight_decay (float): Weight decay for the optimizer.
            lr (float): Learning rate for the optimizer.
            dropout_prob (float): Dropout probability for the model.

        Returns:
            DataFrame: A DataFrame containing the results of all loss function combinations.
        """
        # Tokenize datasets
        self.train_dataset = self._tokenize_dataset(self.train_dataset)
        self.val_dataset = self._tokenize_dataset(self.val_dataset)

        # Generate all combinations of loss functions
        all_combinations = []
        for r in range(1, len(self.loss_functions) + 1):
            all_combinations.extend(combinations(self.loss_functions.keys(), r))

        # Store results
        results = []

        # Evaluate each combination
        for combination in all_combinations:
            print(f"Testing combination: {combination}")
            loss_fns = [self.loss_functions[fn] for fn in combination]
            metrics = self._train_and_evaluate(loss_fns, weight_decay, lr, dropout_prob)
            results.append({
                "combination": combination,
                "f1_macro": metrics.get("eval_f1_macro", 0.0),
                "precision_macro": metrics.get("eval_precision_macro", 0.0),
                "f1_micro": metrics.get("eval_f1_micro", 0.0),
                "precision_micro": metrics.get("eval_precision_micro", 0.0)
            })

        # Sort results by F1-score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='f1_macro', ascending=False)
        return results_df

# Example usage
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = CustomLossEvaluator(
        train_path='/path/to/train.csv',
        val_path='/path/to/val.csv',
        model_name='FacebookAI/xlm-roberta-base',
        num_labels=6
    )

    # Define hyperparameters
    weight_decay = 0.01
    lr = 2e-5
    dropout_prob = 0.5

    # Evaluate loss combinations with the specified hyperparameters
    results_df = evaluator.evaluate_loss_combinations(weight_decay, lr, dropout_prob)
    results_df.to_csv('loss_function_combinations_results.csv', index=False)
    print("Loss function combinations results saved to 'loss_function_combinations_results.csv'.")