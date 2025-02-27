import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Ensure Matplotlib Works in Headless Mode ---
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

# --- Load Fine-Tuned Model ---
MODEL_PATH = "path/to/your/fine-tuned-model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

# --- Load Validation Dataset ---
with open("path/to/your/validation_dataset.json", "r") as f:
    validation_data = json.load(f)

true_labels = []
pred_labels_with_log_level = []
pred_labels_without_log_level = []
results = []

# Define valid outputs
valid_outputs = {"INFO", "WARNING", "ERROR", "CRITICAL"}

# --- Model Inference with Retry ---
def get_valid_prediction(prompt, max_retries=3):
    """Runs inference and ensures a valid response is returned."""
    for _ in range(max_retries):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=10, num_return_sequences=1)

        # Decode and filter prediction
        predicted_text = tokenizer.decode(output[0], skip_special_tokens=True).strip().upper()
        predicted_label = next((word for word in predicted_text.split() if word in valid_outputs), "UNKNOWN")

        if predicted_label in valid_outputs:
            return predicted_label  # Valid response found

    return "UNKNOWN"  # Return "UNKNOWN" if retries fail

# --- Processing Validation Data ---
for record in validation_data:
    instruction = record["instruction"]
    input_data = record["input"].copy()  # Copy input data
    expected_output = record["output"]

    # Construct input prompts
    prompt_with_log_level = f"{instruction}\n{json.dumps(input_data, indent=2)}"
    input_data.pop("log_level", None)  # Remove log_level
    prompt_without_log_level = f"{instruction}\n{json.dumps(input_data, indent=2)}"

    # Get valid predictions
    predicted_label_with_log_level = get_valid_prediction(prompt_with_log_level)
    predicted_label_without_log_level = get_valid_prediction(prompt_without_log_level)

    # Store results
    true_labels.append(expected_output)
    pred_labels_with_log_level.append(predicted_label_with_log_level)
    pred_labels_without_log_level.append(predicted_label_without_log_level)

    results.append({
        "actual_output": expected_output,
        "predicted_with_log_level": predicted_label_with_log_level,
        "predicted_without_log_level": predicted_label_without_log_level
    })

# --- Save Predictions to CSV ---
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

# --- Evaluation Metrics for Both Cases ---
def evaluate_predictions(true_labels, pred_labels, case_name):
    """Evaluates model performance and saves confusion matrix as an image."""
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nAccuracy ({case_name}): {accuracy:.4f}")

    print(f"Classification Report ({case_name}):")
    print(classification_report(true_labels, pred_labels, labels=["INFO", "WARNING", "ERROR", "CRITICAL"]))

    # Confusion Matrix as Table (Text Output)
    cm = confusion_matrix(true_labels, pred_labels, labels=["INFO", "WARNING", "ERROR", "CRITICAL"])
    print(f"\nConfusion Matrix ({case_name}):\n")
    print(pd.DataFrame(cm, index=["INFO", "WARNING", "ERROR", "CRITICAL"], 
                           columns=["INFO", "WARNING", "ERROR", "CRITICAL"]))

    # Save Confusion Matrix Heatmap as Image
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["INFO", "WARNING", "ERROR", "CRITICAL"], 
                yticklabels=["INFO", "WARNING", "ERROR", "CRITICAL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({case_name})")

    img_filename = f"confusion_matrix_{case_name.replace(' ', '_').lower()}.png"
    plt.savefig(img_filename)
    print(f"Confusion Matrix saved as {img_filename}")

    # --- Additional Evaluation Metrics ---
    # F1 Score
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted")
    print(f"Macro F1 Score ({case_name}): {macro_f1:.4f}")
    print(f"Weighted F1 Score ({case_name}): {weighted_f1:.4f}")

    # Precision & Recall
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    print(f"Precision ({case_name}): {precision:.4f}")
    print(f"Recall ({case_name}): {recall:.4f}")

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    print(f"Matthews Correlation Coefficient ({case_name}): {mcc:.4f}")

    # Cohen’s Kappa Score
    kappa = cohen_kappa_score(true_labels, pred_labels)
    print(f"Cohen’s Kappa Score ({case_name}): {kappa:.4f}")

    # AUC-ROC for Binary Classification ("CRITICAL" vs. Others)
    true_binary = [1 if label == "CRITICAL" else 0 for label in true_labels]
    pred_binary = [1 if label == "CRITICAL" else 0 for label in pred_labels]
    auc_score = roc_auc_score(true_binary, pred_binary)
    print(f"AUC-ROC Score ({case_name}): {auc_score:.4f}")

# Evaluate both cases
evaluate_predictions(true_labels, pred_labels_with_log_level, "With Log Level")
evaluate_predictions(true_labels, pred_labels_without_log_level, "Without Log Level")
