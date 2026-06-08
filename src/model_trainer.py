# src/model_trainer.py
"""
Model training module for MHIS fault classification.
Trains and compares 5 classifiers using stratified k-fold
cross-validation. Saves the best model and preprocessing scaler.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold,
                                     cross_validate)
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier)
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (accuracy_score,
                                     classification_report,
                                     confusion_matrix)

LABEL_NAMES   = ['Normal', 'Inner Race', 'Outer Race', 'Ball Element']

class ModelTrainer:
    """
    Loads feature dataset, trains multiple classifiers,
    evaluates with cross-validation, saves best model.
    """
    CV_FOLDS    = 5
    TEST_SIZE   = 0.2
    RANDOM_SEED = 42

    def __init__(self, data_path: str):
        self.df = pd.read_parquet(data_path)
        
        # Identify feature columns (exclude metadata)
        meta_cols = ['label', 'fault_type', 'severity', 'load_hp']
        self.feature_cols = [c for c in self.df.columns
                             if c not in meta_cols]
        
        self.X_raw = self.df[self.feature_cols].values
        self.y     = self.df['label'].values
        
        # Scale features — required for SVM and KNN
        # Random Forest does not need scaling but it does not hurt
        self.scaler   = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X_raw)
        
        print(f"Dataset loaded: {len(self.df)} windows, "
              f"{len(self.feature_cols)} features, "
              f"{len(np.unique(self.y))} classes")

    def get_models(self) -> dict:
        """
        Returns dict of model_name → sklearn estimator.
        All models use SCALED features.
        """
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators  = 200,
                max_features  = 'sqrt',    # each tree sees sqrt(23) ≈ 4 features
                min_samples_leaf = 2,
                random_state  = self.RANDOM_SEED,
                n_jobs        = -1          # use all CPU cores
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators  = 150,
                learning_rate = 0.1,
                max_depth     = 5,
                random_state  = self.RANDOM_SEED
            ),
            'SVM (RBF)': SVC(
                kernel      = 'rbf',
                C           = 10,
                gamma       = 'scale',
                probability = True,         # needed for predict_proba
                random_state= self.RANDOM_SEED
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors = 7,
                metric      = 'euclidean',
                n_jobs      = -1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth    = 15,
                random_state = self.RANDOM_SEED
            )
        }

    def cross_validate_all(self) -> pd.DataFrame:
        """
        Run stratified 5-fold cross-validation on all 5 models.
        Returns DataFrame with accuracy and F1 scores per model.
        This is the main comparison table for your report.
        """
        skf = StratifiedKFold(
            n_splits = self.CV_FOLDS,
            shuffle  = True,
            random_state = self.RANDOM_SEED
        )
        
        results = []
        print(f"\nRunning {self.CV_FOLDS}-fold cross-validation...")
        print(f"{'Model':22s} | {'Acc Mean':>10} | "
              f"{'Acc Std':>8} | {'F1 Mean':>10} | {'F1 Std':>8}")
        print("─" * 70)
        
        for name, model in self.get_models().items():
            cv_results = cross_validate(
                model,
                self.X_scaled,
                self.y,
                cv      = skf,
                scoring = {
                    'accuracy': 'accuracy',
                    'f1_macro': 'f1_macro'
                },
                n_jobs = -1
            )
            
            acc_mean = cv_results['test_accuracy'].mean()
            acc_std  = cv_results['test_accuracy'].std()
            f1_mean  = cv_results['test_f1_macro'].mean()
            f1_std   = cv_results['test_f1_macro'].std()
            
            print(f"  {name:20s} | {acc_mean:>9.4f}  | "
                  f"{acc_std:>7.4f}  | {f1_mean:>9.4f}  | {f1_std:>7.4f}")
            
            results.append({
                'Model':      name,
                'Acc_Mean':   round(acc_mean, 4),
                'Acc_Std':    round(acc_std, 4),
                'F1_Mean':    round(f1_mean, 4),
                'F1_Std':     round(f1_std, 4),
                'Acc_Min':    round(cv_results['test_accuracy'].min(), 4),
                'Acc_Max':    round(cv_results['test_accuracy'].max(), 4)
            })
            
        results_df = pd.DataFrame(results).sort_values(
            'F1_Mean', ascending=False
        )
        return results_df

    def train_final_model(self, model_name: str = 'Random Forest') -> tuple:
        """
        Train the selected model on 80% of data.
        Evaluate on the held-out 20% test set.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled,
            self.y,
            test_size    = self.TEST_SIZE,
            stratify     = self.y,       # maintain class proportions
            random_state = self.RANDOM_SEED
        )
        
        print(f"\nTraining final {model_name} model...")
        print(f"  Train set: {len(X_train)} windows")
        print(f"  Test set:  {len(X_test)} windows")
        
        model = self.get_models()[model_name]
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        
        print(f"\nFinal Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print("\nPer-class performance:")
        print(classification_report(
            y_test, y_pred,
            target_names = LABEL_NAMES,
            digits       = 4
        ))
        
        # Save model and scaler
        Path('models').mkdir(exist_ok=True)
        joblib.dump(model,       'models/final_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("\nModel saved to models/final_model.pkl")
        print("Scaler saved to models/scaler.pkl")
        
        return model, self.scaler, X_test, y_test, y_pred

    def get_feature_names(self) -> list:
        return self.feature_cols