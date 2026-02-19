import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def custom_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plots a confusion matrix with counts and percentages relative to true labels.
    """
    # Calculate basic confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages relative to True Labels (row totals)
    # np.newaxis is used to divide each row by its sum
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create labels with count and row percentage
    labels = [f"{count}\n({percent:.2%})" for count, percent in zip(cm.flatten(), cm_normalized.flatten())]
    labels = np.array(labels).reshape(2, 2)
    
    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Stayed (0)', 'Churned (1)'], 
                yticklabels=['Stayed (0)', 'Churned (1)'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

def custom_roc_curve(y_true, y_probs, title='ROC Curve'):
    """
    Plots the ROC Curve using true labels and predicted probabilities.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()    

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def display_model_performance(y_true, y_pred, y_proba, model_name="Model"):
    """
    Prints major classification metrics in a structured way.
    """
    print(f"--- {model_name} Results ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))