# src/model_evaluator.py
"""
Evaluation module: produces all result figures and metrics.
Called after model_trainer.py to analyze the trained model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report,
                             roc_curve, auc)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing   import label_binarize
from pathlib import Path

LABEL_NAMES = ['Normal', 'Inner Race', 'Outer Race', 'Ball Element']
FIGURES_DIR = Path('../reports/figures')

class ModelEvaluator:
    def __init__(self, model_path: str = 'models/final_model.pkl',
                 scaler_path: str = 'models/scaler.pkl'):
        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def plot_confusion_matrix(self, y_true, y_pred,
                              normalize: bool = True,
                              save: bool = True) -> None:
        """
        Plot confusion matrix.
        Each row = true class.
        Each column = predicted class.
        """
        cm = confusion_matrix(
            y_true, y_pred,
            normalize='true' if normalize else None
        )
        
        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(
            confusion_matrix = cm,
            display_labels   = LABEL_NAMES
        )
        disp.plot(
            ax          = ax,
            cmap        = 'Blues',
            colorbar    = True,
            values_format = '.2%' if normalize else 'd'
        )
        ax.set_title(
            'Confusion Matrix — Random Forest Classifier\n'
            'Rows = True Class | Columns = Predicted Class\n'
            'Diagonal = correctly classified (recall)',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()
        
        if save:
            fig.savefig(FIGURES_DIR / 'day8_confusion_matrix.png',
                        dpi=200, bbox_inches='tight')
        plt.show()
        
        # Print interpretation
        print("\nConfusion Matrix Interpretation:")
        for i, label in enumerate(LABEL_NAMES):
            recall = cm[i, i]
            errors = [(LABEL_NAMES[j], cm[i,j])
                      for j in range(len(LABEL_NAMES)) if j != i and cm[i,j] > 0.01]
            print(f"  {label:15s}: {recall:.1%} correctly identified", end='')
            if errors:
                error_str = ', '.join([f"{pct:.1%} confused as {lbl}"
                                       for lbl, pct in errors])
                print(f"  | Errors: {error_str}")
            else:
                print()

    def plot_feature_importance(self, feature_names: list,
                                top_n: int = 15,
                                save: bool = True) -> None:
        """
        Random Forest feature importance plot.
        Importance = how much each feature reduces impurity
        across all trees, averaged over the forest.
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance only available for tree-based models")
            return
            
        importances = self.model.feature_importances_
        indices     = np.argsort(importances)[::-1][:top_n]
        colors_bar = ['#e74c3c'] * 3 + ['#3498db'] * (top_n - 3)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            range(top_n),
            importances[indices][::-1],
            color   = colors_bar[::-1],
            alpha   = 0.85,
            edgecolor = 'black',
            linewidth = 0.5
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(
            [feature_names[i] for i in indices[::-1]],
            fontsize=10
        )
        ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)',
                      fontsize=11)
        ax.set_title(
            f'Top {top_n} Feature Importances — Random Forest\n'
            'Red bars = Top 3 most diagnostic features',
            fontsize=12, fontweight='bold'
        )
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save:
            fig.savefig(FIGURES_DIR / 'day8_feature_importance.png',
                        dpi=200, bbox_inches='tight')
        plt.show()
        
        print(f"\nTop 5 most important features:")
        for rank, idx in enumerate(indices[:5], 1):
            print(f"  {rank}. {feature_names[idx]:22s}: "
                  f"{importances[idx]:.4f} ({importances[idx]*100:.1f}%)")

    def plot_learning_curve(self, X_scaled, y, save: bool = True) -> None:
        """
        Learning curve: model accuracy vs training set size.
        """
        print("Computing learning curve (this takes ~1-2 minutes)...")
        train_sizes, train_scores, val_scores = learning_curve(
            self.model,
            X_scaled, y,
            cv           = StratifiedKFold(5, shuffle=True, random_state=42),
            n_jobs       = -1,
            train_sizes  = np.linspace(0.1, 1.0, 8),
            scoring      = 'accuracy'
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_sizes, train_mean,
                color='#e74c3c', marker='o',
                label='Training accuracy', linewidth=2)
        ax.fill_between(train_sizes,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.12, color='#e74c3c')
        ax.plot(train_sizes, val_mean,
                color='#2ecc71', marker='s',
                label='Validation accuracy', linewidth=2)
        ax.fill_between(train_sizes,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.12, color='#2ecc71')
        ax.set_xlabel('Training Set Size (windows)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_ylim(0.6, 1.05)
        ax.set_title(
            'Learning Curve — Random Forest\n'
            'Convergence shows model has sufficient training data',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            fig.savefig(FIGURES_DIR / 'day8_learning_curve.png',
                        dpi=150, bbox_inches='tight')
        plt.show()
        
        gap = train_mean[-1] - val_mean[-1]
        print(f"\nFinal training accuracy:   {train_mean[-1]:.4f}")
        print(f"Final validation accuracy: {val_mean[-1]:.4f}")
        print(f"Generalization gap:        {gap:.4f}")
        
        if gap < 0.05:
            print("→ Minimal overfitting ✅")
        elif gap < 0.10:
            print("→ Mild overfitting — acceptable")
        else:
            print("→ Significant overfitting — consider pruning or more data")