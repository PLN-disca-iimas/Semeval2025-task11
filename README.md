# LATE-GIL-NLP at SemEval-2025 Task 11: Multi-Language Emotion Detection and Intensity Classification (Track A & B)

This repository contains the base code used by the **LATE-GIL-NLP** team for participation in **SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection**. The goal of this task is to detect emotions in multilingual texts and classify their intensity, addressing challenges such as class imbalance and linguistic diversity.

The provided code serves as templates that can be adapted to different languages and configurations. These codes cover the following key functionalities:

1. **Model Selection** (`MODEL_SELECTION.py`)
2. **Hyperparameter Tuning** (`HYPERPARAMETER_TUNING.py`)
3. **Custom Loss Functions Combination** (`CUSTOM_LOSS_FUNCTIONS_COMBINATION.py`)
4. **Final Model Training** (`FINAL_MODEL_TRAINING.py`)

---

## Repository Structure

The repository is organized as follows:

SemEval-2025-Task-11/Track_A_CODES/
MODEL_SELECTION.py # Base code for model selection
HYPERPARAMETER_TUNING.py # Base code for hyperparameter tuning
CUSTOM_LOSS_FUNCTIONS_COMBINATION.py # Base code for custom loss functions combination
FINAL_MODEL_TRAINING.py # Base code for final model training
README.md 
requirements.txt # Required dependencies to run the code


---

## Description of Base Code

### 1. **Model Selection** (`MODEL_SELECTION.py`)

This script evaluates multiple pre-trained models available on Hugging Face that support the target languages. Models are filtered based on language compatibility, and the top 5 most downloaded models for each language are selected for initial testing. Each model is fine-tuned for 1 epoch on the training set to identify the best-performing model for each language.

#### Features:
- Filtering models by language compatibility.
- Evaluation of pre-trained models.
- Selection of the best model based on performance metrics.

---

### 2. **Hyperparameter Tuning** (`HYPERPARAMETER_TUNING.py`)

This script performs a grid search to optimize hyperparameters using the best model identified in the model selection stage. The hyperparameters being optimized include:

- Learning rate (`learning_rate`): `[1e-5, 2e-5, 3e-5, 5e-5]`
- Weight decay (`weight_decay`): `[0.001, 0.01, 0.1]`
- Dropout probability (`dropout_prob`): `[0.3, 0.5]`

#### Features:
- Grid search for hyperparameter optimization.
- Training with the `AdamW` optimizer.
- Selection of the best hyperparameter combination based on evaluation metrics.

---

### 3. **Custom Loss Functions Combination** (`CUSTOM_LOSS_FUNCTIONS_COMBINATION.py`)

This script implements and combines custom loss functions to improve performance on imbalanced datasets. The loss functions evaluated include:

- `BCEWithLogitsLoss`
- `MSELoss`
- `CrossEntropyLoss`
- `FocalLoss` (with `alpha=0.25`, `gamma=2.0`)
- `LabelSmoothingLoss` (with `smoothing=0.1`)

#### Features:
- Implementation of custom loss functions.
- Combination of multiple loss functions to improve performance on imbalanced data.
- Evaluation of loss function combinations.

---

### 4. **Final Model Training** (`FINAL_MODEL_TRAINING.py`)

This script trains the final model using the best model, optimal hyperparameters, and loss function combination identified in the previous stages. The model is trained on the combined training and validation sets and evaluated on the test set.

#### Features:
- Training the final model with the best hyperparameters and loss functions.
- Saving the final model for production use.
- Evaluating the model on the test set.

---

## Requirements

To run the code, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

---

## Results

The results obtained in SemEval-2025 Task 11 demonstrate strong performance in low-resource languages, such as Tigrinya (2nd place), Igbo (3rd place), and Oromo (4th place). For more details on the results, refer to the paper.

---

## Contributions
This repository was developed by the LATE-GIL-NLP team as part of their participation in SemEval-2025 Task 11. If you would like to contribute or have any questions, feel free to open an issue or submit a pull request.

---

## References

Paper description: To be published

GitHub task description: https://github.com/emotion-analysis-project/SemEval2025-task11?tab=readme-ov-file

---

## Cite as:

@inproceedings{vázquez-osorio-etal-2025-late-gil-nlp,
  title = "{LATE-GIL-NLP at SemEval-2025 Task 11: Multi-Language Emotion Detection and Intensity Classification Using Transformer Models with Optimized Loss Functions for Imbalanced Data}",
  author = "Vázquez-osorio, Jesús and Gómez-adorno, Helena and Sierra, Gerardo and Sierra-casiano, Vladimir and Canchola-hernández, Diana and Tovar-cortés, José and Solís-vilchis, Roberto and Salazar, Gabriel",
  booktitle = "Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics"
}
