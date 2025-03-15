from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score
from torch.optim import AdamW

class FinalModelTrainer:
    """
    A class to train and evaluate the final model using the best hyperparameters and loss functions.
    The model is trained on the combined training and validation datasets and evaluated on the test set.
    """

    def __init__(self, train_path, val_path, test_path, model_name, num_labels, weight_decay, lr, dropout_prob, loss_fns):
        """
        Initializes the FinalModelTrainer with dataset paths, model name, number of labels, and hyperparameters.

        Args:
            train_path (str): Path to the training dataset CSV file.
            val_path (str): Path to the validation dataset CSV file.
            test_path (str): Path to the test dataset CSV file.
            model_name (str): Name of the pre-trained model to use.
            num_labels (int): Number of emotion labels in the dataset.
            weight_decay (float): Weight decay for the optimizer.
            lr (float): Learning rate for the optimizer.
            dropout_prob (float): Dropout probability for the model.
            loss_fns (list): List of loss functions to use.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_name = model_name
        self.num_labels = num_labels
        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout_prob = dropout_prob
        self.loss_fns = loss_fns
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_and_prepare_datasets()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            num_labels=self.num_labels
        )
        self.model.config.hidden_dropout_prob = self.dropout_prob

    def _load_and_prepare_datasets(self):
        """
        Loads and prepares the datasets from CSV files, converting emotion columns to lists of labels.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets.
        """
        df_train = pd.read_csv(self.train_path)
        df_val = pd.read_csv(self.val_path)
        df_test = pd.read_csv(self.test_path)

        # Convert emotion columns to float32
        emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        df_train[emotion_columns] = df_train[emotion_columns].astype('float32')
        df_val[emotion_columns] = df_val[emotion_columns].astype('float32')
        df_test[emotion_columns] = df_test[emotion_columns].astype('float32')

        # Create labels column
        df_train['labels'] = df_train[emotion_columns].values.tolist()
        df_val['labels'] = df_val[emotion_columns].values.tolist()

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(df_train[['text', 'labels']])
        val_dataset = Dataset.from_pandas(df_val[['text', 'labels']])
        test_dataset = Dataset.from_pandas(df_test[['id', 'text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']])

        return train_dataset, val_dataset, test_dataset

    def _tokenize_dataset(self, dataset):
        """
        Tokenizes the dataset using the model's tokenizer.

        Args:
            dataset (Dataset): Dataset to tokenize.

        Returns:
            Dataset: Tokenized dataset.
        """
        return dataset.map(lambda x: self.tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)

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

    @staticmethod
    def _define_loss_functions():
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

    def train_and_evaluate_final(self):
        """
        Trains the final model on the combined training and validation datasets and evaluates it on the test set.
        Only saves the model from the last epoch.
        """
        # Combine training and validation datasets
        combined_dataset = Dataset.from_pandas(pd.concat([
            self.train_dataset.to_pandas(),
            self.val_dataset.to_pandas()
        ]))

        # Tokenize datasets
        combined_dataset = self._tokenize_dataset(combined_dataset)
        test_dataset = self._tokenize_dataset(self.test_dataset)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./final_output",  # Output directory for results
            num_train_epochs=10,  # Train for 10 epochs
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="no",  # Disable evaluation during training
            save_strategy="no",  # Do not save checkpoints during training
            save_total_limit=1,  # Only keep the final model
            logging_dir=None,  # Disable logging
            logging_steps=500,  # Reduce intermediate logs
            weight_decay=self.weight_decay  # Use the provided weight decay
        )

        # Trainer
        trainer = self.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=combined_dataset,
            eval_dataset=None,  # No validation set
            compute_metrics=self._compute_metrics,
            loss_fns=self.loss_fns,
            optimizers=(AdamW(self.model.parameters(), lr=self.lr), None)
        )

        # Train the model
        trainer.train()

        # Save the final model (last epoch)
        trainer.save_model("./final_output")

        # Evaluate on the test set
        predictions = trainer.predict(test_dataset)
        test_preds = predictions.predictions
        test_labels = predictions.label_ids

        # Capture evaluation metrics
        metrics = predictions.metrics

        # Save test predictions to a CSV file
        sigmoid = nn.Sigmoid()
        probs = sigmoid(torch.tensor(test_preds))
        y_preds = (probs >= 0.4).float().numpy()

        df_test_predictions = pd.DataFrame(y_preds, columns=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'])
        df_test_predictions['id'] = self.test_dataset['id']
        df_test_predictions['text'] = self.test_dataset['text']
        df_test_predictions = df_test_predictions[['id', 'text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']]
        df_test_predictions.to_csv("pred_YOR.csv", index=False)

        # Save training information to a CSV file
        training_data = {
            "model_name": self.model_name,
            "best_combination": [str(fn) for fn in self.loss_fns],
            "f1_macro": metrics.get("eval_f1_macro", 0.0),
            "precision_macro": metrics.get("eval_precision_macro", 0.0),
            "f1_micro": metrics.get("eval_f1_micro", 0.0),
            "precision_micro": metrics.get("eval_precision_micro", 0.0),
            "epoch_metrics": metrics
        }

        df_training_info = pd.DataFrame([training_data])
        df_training_info.to_csv("final_training_info_YOR.csv", index=False)

        return metrics

    class CustomTrainer(Trainer):
        """
        Custom Trainer class to handle combined loss functions.
        """
        def __init__(self, *args, loss_fns=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_fns = loss_fns

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Computes the combined loss using the provided loss functions.

            Args:
                model: The model to compute the loss for.
                inputs: Input data.
                return_outputs (bool): Whether to return model outputs.
                **kwargs: Additional arguments (e.g., num_items_in_batch).

            Returns:
                The computed loss.
            """
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            total_loss = 0
            for loss_fn in self.loss_fns:
                total_loss += loss_fn(logits, labels.float())
            avg_loss = total_loss / len(self.loss_fns)
            return (avg_loss, outputs) if return_outputs else avg_loss

# Example usage
if __name__ == "__main__":
    # Define paths and hyperparameters
    train_path = '/path/to/train.csv'
    val_path = '/path/to/val.csv'
    test_path = '/path/to/test.csv'
    model_name = 'FacebookAI/xlm-roberta-base'
    num_labels = 6
    weight_decay = 0.01
    lr = 2e-5
    dropout_prob = 0.5

    # Define loss functions
    loss_fns = [
        nn.BCEWithLogitsLoss(),
        FinalModelTrainer._define_loss_functions()["FocalLoss"],
        FinalModelTrainer._define_loss_functions()["LabelSmoothingLoss"]
    ]

    # Initialize and train the final model
    trainer = FinalModelTrainer(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model_name=model_name,
        num_labels=num_labels,
        weight_decay=weight_decay,
        lr=lr,
        dropout_prob=dropout_prob,
        loss_fns=loss_fns
    )

    metrics = trainer.train_and_evaluate_final()
    print("Final training complete. Metrics:")
    print(metrics)