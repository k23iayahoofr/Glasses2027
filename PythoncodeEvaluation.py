"""
AV-ASD Dataset Evaluation Script
Adapted for: https://github.com/ShijianDeng/AV-ASD/tree/main/dataset/csvs

IMPORTANT: The AV-ASD csvs folder contains video URLs, NOT model predictions.
You must first run your classifier on the 928 extracted video clips to obtain
predicted_situation and confidence values before running the evaluation below.

This script provides:
  1. A loader that reads the AV-ASD CSV structure (video URLs)
  2. A simulated prediction generator (replace with your real model output)
  3. The evaluation metrics from your paper (F1, FAR, ECE)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# AV-ASD BEHAVIOR CATEGORIES (10 classes)
# ==========================================
AV_ASD_SITUATIONS = [
    'Absence or Avoidance of Eye Contact',
    'Aggressive Behavior',
    'Hyper- or Hyporeactivity to Sensory Input',
    'Non-Responsiveness to Verbal Interaction',
    'Non-Typical Language',
    'Object Lining-Up',
    'Self-Hitting or Self-Injurious Behavior',
    'Self-Spinning or Spinning Objects',
    'Upper Limb Stereotypies',
    'Background'
]

# ==========================================
# 1. LOAD AV-ASD CSV STRUCTURE
# ==========================================
def load_av_asd_csv(csv_path):
    """
    Loads a CSV from the AV-ASD dataset.
    The CSV contains video URLs, not features or predictions.

    Expected columns (based on AV-ASD repo):
        - video_id
        - url (YouTube or Facebook link)
        - label (behavior category)
        - start_time / end_time (for clip extraction)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded AV-ASD CSV: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")
    return df


# ==========================================
# 2. SIMULATE MODEL PREDICTIONS
#    (Replace with your real inference output)
# ==========================================
def simulate_predictions_from_urls(df, accuracy=0.80, random_state=42):
    """
    Given the AV-ASD CSV (video URLs + labels), simulate what your
    classifier WOULD output after running on the actual video clips.

    Replace this function with your real model inference:
        - Load each video clip
        - Run your audiovisual classifier
        - Record predicted_situation and confidence
    """
    np.random.seed(random_state)

    # Detect the label column (AV-ASD uses 'label' or 'class')
    label_col = None
    for c in ['label', 'class', 'category', 'behavior']:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("No label column found in CSV. Expected one of: label, class, category, behavior")

    y_true = df[label_col].values

    # Simulate predictions
    y_pred = []
    confidences = []
    for true_label in y_true:
        if np.random.rand() < accuracy:
            pred = true_label
            conf = np.random.uniform(0.65, 0.99)
        else:
            candidates = [s for s in AV_ASD_SITUATIONS if s != true_label]
            pred = np.random.choice(candidates)
            conf = np.random.uniform(0.30, 0.75)
        y_pred.append(pred)
        confidences.append(conf)

    # Build evaluation dataframe
    eval_df = pd.DataFrame({
        'video_id': df.get('video_id', range(len(df))),
        'true_situation': y_true,
        'predicted_situation': y_pred,
        'confidence': confidences,
        'clip_duration_hours': 1.0 / 3600.0,  # 1 second per clip -> hours
    })
    return eval_df


# ==========================================
# 3. METRIC FUNCTIONS
# ==========================================
def calculate_f1_per_situation(df):
    """Precision, Recall, F1 per behavior category."""
    labels = [s for s in AV_ASD_SITUATIONS if s in df['true_situation'].unique()]
    precision, recall, f1, support = precision_recall_fscore_support(
        df['true_situation'],
        df['predicted_situation'],
        average=None,
        labels=labels,
        zero_division=0
    )
    metrics_df = pd.DataFrame({
        'Situation': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    return metrics_df


def calculate_false_alert_rate(df, target_situation='Aggressive Behavior',
                               confidence_threshold=0.7):
    """
    FAR = N_FA / T_h   (Eq. 1 in your paper)
    A "cue" is delivered when predicted_situation == target AND confidence >= threshold.
    A false alert is a delivered cue where the TRUE situation is NOT the target.
    """
    cues_delivered = df[
        (df['predicted_situation'] == target_situation) &
        (df['confidence'] >= confidence_threshold)
    ]
    false_alerts = cues_delivered[cues_delivered['true_situation'] != target_situation]
    N_FA = len(false_alerts)
    T_h = df['clip_duration_hours'].sum()
    far = N_FA / T_h if T_h > 0 else 0.0
    return far, N_FA


def calculate_ece(df, n_bins=10):
    """
    Expected Calibration Error (Eq. 2 in your paper).
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    df = df.copy()
    df['correct'] = (df['true_situation'] == df['predicted_situation']).astype(int)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = df[(df['confidence'] > lo) & (df['confidence'] <= hi)]
        if len(in_bin) > 0:
            acc_in_bin = in_bin['correct'].mean()
            conf_in_bin = in_bin['confidence'].mean()
            ece += (len(in_bin) / len(df)) * abs(acc_in_bin - conf_in_bin)
    return ece


# ==========================================
# 4. MAIN EVALUATION PIPELINE
# ==========================================
def run_evaluation(csv_path=None):
    print("=" * 65)
    print("AV-ASD EVALUATION: Audiovisual Smart Glasses Behavior Support")
    print("=" * 65)

    # ---- 1. Load data ----
    print("\n[1] Loading dataset...")
    if csv_path and os.path.exists(csv_path):
        raw_df = load_av_asd_csv(csv_path)
        eval_df = simulate_predictions_from_urls(raw_df, accuracy=0.80)
        print(f"    Simulated predictions for {len(eval_df)} clips.")
        print("    >>> REPLACE simulate_predictions_from_urls() with your real")
        print("        model inference on the extracted AV-ASD video clips. <<<")
    else:
        print("    No CSV path provided. Falling back to fully simulated data.")
        eval_df = simulate_predictions_from_urls(
            pd.DataFrame({'video_id': range(928),
                          'label': np.random.choice(AV_ASD_SITUATIONS, 928)}),
            accuracy=0.80
        )

    # ---- 2. Technical Performance (E1) ----
    print("\n[2] E1 Technical Performance: Situation Recognition")
    metrics_df = calculate_f1_per_situation(eval_df)
    print("\n    Per-Situation Performance:")
    print(metrics_df.to_string(index=False))
    macro_f1 = metrics_df['F1-Score'].mean()
    print(f"\n    Macro F1-Score: {macro_f1:.4f}")

    # ---- 3. False Alert Rate ----
    print("\n[3] E1 Technical Performance: False Alert Rate")
    for target in ['Aggressive Behavior', 'Self-Hitting or Self-Injurious Behavior']:
        far, n_fa = calculate_false_alert_rate(eval_df, target_situation=target)
        print(f"    Target: '{target}'")
        print(f"      False Alerts: {n_fa}  |  FAR: {far:.2f} alerts/hour")

    # ---- 4. Calibration ----
    print("\n[4] E1 Technical Performance: Calibration")
    ece = calculate_ece(eval_df, n_bins=10)
    print(f"    Expected Calibration Error (ECE): {ece:.4f}")
    print("    (Lower is better; 0 = perfectly calibrated)")

    # ---- 5. Confusion Matrix + Plots ----
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # (a) F1 per situation
        sns.barplot(x='F1-Score', y='Situation', data=metrics_df,
                    palette='viridis', ax=axes[0])
        axes[0].set_title('F1-Score by Behavior Category')
        axes[0].set_xlim(0, 1)

        # (b) Calibration curve
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accs = []
        for i in range(10):
            in_bin = eval_df[(eval_df['confidence'] > bins[i]) &
                             (eval_df['confidence'] <= bins[i + 1])]
            accs.append(in_bin['correct'].mean()
                        if len(in_bin) > 0 else np.nan)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect')
        axes[1].plot(bin_centers, accs, 'o-', label='Model')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Calibration Curve (ECE={ece:.3f})')
        axes[1].legend()

        # (c) Confusion matrix
        labels = [s for s in AV_ASD_SITUATIONS if s in eval_df['true_situation'].unique()]
        cm = confusion_matrix(eval_df['true_situation'],
                              eval_df['predicted_situation'],
                              labels=labels, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[2], cbar=False)
        axes[2].set_title('Normalized Confusion Matrix')
        axes[2].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.savefig('av_asd_evaluation_results.png', dpi=150)
        print("\n[5] Visualization saved: av_asd_evaluation_results.png")
    except Exception as e:
        print(f"\n[5] Visualization skipped: {e}")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    # Point this to the CSV you downloaded from:
    # https://github.com/ShijianDeng/AV-ASD/tree/main/dataset/csvs
    # Example (adjust to the actual filename, e.g. 'data.csv'):
    CSV_PATH = "AV-ASD/dataset/csvs/data.csv"

    run_evaluation(csv_path=CSV_PATH)