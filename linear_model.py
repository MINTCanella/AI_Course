# file: linear_model.py
"""
Linear regression model for salary prediction with cross-validation.
Optimized for hh.ru data.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedLinearSalaryPredictor:
    """
    Improved linear regression model for salary prediction with regularization.

    This class implements linear regression models with automatic
    hyperparameter tuning using cross-validation.
    """

    def __init__(self, resources_dir: str = 'resources'):
        """
        Initialize the predictor.

        Args:
            resources_dir: Directory to save/load model resources
        """
        self.resources_dir = Path(resources_dir)
        self.resources_dir.mkdir(exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.selected_feature_indices = None
        self.selected_feature_names = None
        self.metrics = {}
        self.best_alpha = None
        self.top_correlated_features = None

    def load_data(self, x_path: str, y_path: str,
                  feature_names_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load processed data from .npy files.

        Args:
            x_path: Path to x_data.npy
            y_path: Path to y_data.npy
            feature_names_path: Path to feature_names.txt

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Loading data from %s and %s", x_path, y_path)

        X = np.load(x_path)
        y = np.load(y_path)

        logger.info("Data loaded: X shape=%s, y shape=%s", X.shape, y.shape)

        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.feature_names = [line.strip().split(': ', 1)[1] for line in lines]
        else:
            possible_paths = [
                Path(x_path).parent / 'feature_names.txt',
                Path(x_path).parent / 'feature_names.json',
                Path('feature_names.txt')
            ]

            for path in possible_paths:
                if path.exists():
                    if path.suffix == '.json':
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.feature_names = data.get('feature_names', [])
                    else:
                        with open(path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            self.feature_names = [line.strip().split(': ', 1)[1] for line in lines]
                    break
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if len(self.feature_names) != X.shape[1]:
            logger.warning(
                "Feature names count (%s) doesn't match X columns (%s)",
                len(self.feature_names), X.shape[1]
            )
            if len(self.feature_names) > X.shape[1]:
                self.feature_names = self.feature_names[:X.shape[1]]
            else:
                for i in range(len(self.feature_names), X.shape[1]):
                    self.feature_names.append(f'feature_{i}')

        logger.info("Loaded %s feature names", len(self.feature_names))

        self._analyze_feature_correlations(X, y)

        return X, y, self.feature_names

    def _analyze_feature_correlations(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Analyze correlations between features and target.

        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("=== Feature Correlation Analysis ===")

        correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                    correlations.append((abs(corr), i, corr, feature_name))

        correlations.sort(reverse=True)

        logger.info("Top features by correlation with target:")
        for i, (abs_corr, idx, corr, name) in enumerate(correlations[:15]):
            direction = "↑" if corr > 0 else "↓"
            logger.info(
                "  %2d. %-30s: %+.4f (abs: %.4f) %s",
                i + 1, name, corr, abs_corr, direction
            )

        self.top_correlated_features = correlations[:20]

    def select_features_by_correlation(self, X: np.ndarray, y: np.ndarray, k: int = 15) -> np.ndarray:
        """
        Select top k features based on correlation with target.

        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select

        Returns:
            Indices of selected features
        """
        correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    correlations.append((abs(corr), i))

        correlations.sort(reverse=True)

        self.selected_feature_indices = [i for _, i in correlations[:k]]
        self.selected_feature_names = [self.feature_names[i] for i in self.selected_feature_indices]

        logger.info("Selected top %s features:", k)
        for i, idx in enumerate(self.selected_feature_indices):
            name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
            logger.info("  %s. %s", i + 1, name)

        return self.selected_feature_indices

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info("Train set: %s samples", X_train.shape[0])
        logger.info("Test set: %s samples", X_test.shape[0])

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    model_type: str = 'ridge', n_features: int = 15) -> Any:
        """
        Train linear regression model with cross-validation.

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of linear model ('ridge', 'lasso', 'elasticnet')
            n_features: Number of top features to use

        Returns:
            Trained model
        """
        logger.info("Training %s regression with CV...", model_type)

        if n_features < X_train.shape[1]:
            logger.info("Selecting top %s features...", n_features)
            selected_indices = self.select_features_by_correlation(X_train, y_train, k=n_features)
            X_train_selected = X_train[:, selected_indices]
        else:
            X_train_selected = X_train
            self.selected_feature_names = self.feature_names

        self.scaler = StandardScaler()

        if model_type == 'ridge':
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            model = RidgeCV(alphas=alphas, cv=5, scoring='r2')
            logger.info("RidgeCV will test alphas: %s", alphas)
        elif model_type == 'lasso':
            model = LassoCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1)
            logger.info("LassoCV will automatically select alpha")
        elif model_type == 'elasticnet':
            model = ElasticNetCV(cv=5, random_state=42, max_iter=5000, n_jobs=-1)
            logger.info("ElasticNetCV will automatically select parameters")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        pipeline = SkPipeline([
            ('scaler', self.scaler),
            ('model', model)
        ])

        logger.info("Training model with cross-validation...")
        pipeline.fit(X_train_selected, y_train)
        self.model = pipeline

        if model_type == 'ridge':
            self.best_alpha = pipeline.named_steps['model'].alpha_
            logger.info("Best alpha selected by RidgeCV: %s", self.best_alpha)
        elif model_type == 'lasso':
            self.best_alpha = pipeline.named_steps['model'].alpha_
            logger.info("Best alpha selected by LassoCV: %s", self.best_alpha)
        elif model_type == 'elasticnet':
            model_step = pipeline.named_steps['model']
            self.best_alpha = model_step.alpha_
            self.best_l1_ratio = model_step.l1_ratio_
            logger.info("Best parameters: alpha=%s, l1_ratio=%s", self.best_alpha, self.best_l1_ratio)

        cv_scores = cross_val_score(pipeline, X_train_selected, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)

        logger.info("5-fold Cross-validation R² scores: %s", cv_scores)
        logger.info("Mean CV R²: %.4f (±%.4f)", cv_scores.mean(), cv_scores.std())

        logger.info("Model trained successfully with CV")

        return self.model

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test target (in log scale)

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        logger.info("=" * 60)
        logger.info("Evaluating model on test set...")

        if self.selected_feature_indices:
            X_test_selected = X_test[:, self.selected_feature_indices]
        else:
            X_test_selected = X_test

        y_pred_log = self.model.predict(X_test_selected)

        y_pred_rub = np.expm1(y_pred_log)
        y_test_rub = np.expm1(y_test)

        zero_mask_pred = y_pred_rub == 0
        zero_mask_test = y_test_rub == 0

        if zero_mask_pred.any() or zero_mask_test.any():
            logger.warning("Found zeros after inverse transformation:")
            logger.warning("  Predictions: %s zeros", zero_mask_pred.sum())
            logger.warning("  True values: %s zeros", zero_mask_test.sum())

            epsilon = 1e-10
            y_pred_rub = np.where(y_pred_rub == 0, epsilon, y_pred_rub)
            y_test_rub = np.where(y_test_rub == 0, epsilon, y_test_rub)

        metrics_log = {
            'mae_log': mean_absolute_error(y_test, y_pred_log),
            'mse_log': mean_squared_error(y_test, y_pred_log),
            'rmse_log': np.sqrt(mean_squared_error(y_test, y_pred_log)),
            'r2_log': r2_score(y_test, y_pred_log)
        }

        metrics_rub = {
            'mae_rub': mean_absolute_error(y_test_rub, y_pred_rub),
            'mse_rub': mean_squared_error(y_test_rub, y_pred_rub),
            'rmse_rub': np.sqrt(mean_squared_error(y_test_rub, y_pred_rub)),
            'r2_rub': r2_score(y_test_rub, y_pred_rub),
            'mape': np.mean(np.abs((y_test_rub - y_pred_rub) / np.maximum(y_test_rub, 1e-10))) * 100
        }

        within_20_pct = np.mean(np.abs((y_test_rub - y_pred_rub) / np.maximum(y_test_rub, 1e-10)) < 0.2) * 100
        within_30_pct = np.mean(np.abs((y_test_rub - y_pred_rub) / np.maximum(y_test_rub, 1e-10)) < 0.3) * 100

        self.metrics = {**metrics_log, **metrics_rub,
                        'within_20_pct': within_20_pct,
                        'within_30_pct': within_30_pct,
                        'best_alpha': self.best_alpha}

        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 60)

        logger.info("In log scale:")
        logger.info("  R²:   %.4f", metrics_log['r2_log'])
        logger.info("  MAE:  %.4f", metrics_log['mae_log'])
        logger.info("  RMSE: %.4f", metrics_log['rmse_log'])

        logger.info("In original scale (RUB):")
        logger.info("  R²:   %.4f", metrics_rub['r2_rub'])

        # Исправляем форматирование - убираем запятую в %f
        mae_rub_formatted = float(metrics_rub['mae_rub'])
        rmse_rub_formatted = float(metrics_rub['rmse_rub'])

        logger.info("  MAE:  %.4f RUB", mae_rub_formatted)
        logger.info("  RMSE: %.4f RUB", rmse_rub_formatted)
        logger.info("  MAPE: %.1f%%", metrics_rub['mape'])
        logger.info("  Within 20%% error: %.1f%%", within_20_pct)
        logger.info("  Within 30%% error: %.1f%%", within_30_pct)

        if hasattr(self, 'best_alpha') and self.best_alpha:
            logger.info("  Best regularization parameter: %s", self.best_alpha)

        logger.info("Example predictions (first 5 samples):")
        for i in range(min(5, len(y_test_rub))):
            actual = y_test_rub[i]
            predicted = y_pred_rub[i]
            error_pct = ((predicted - actual) / max(actual, 1e-10)) * 100
            logger.info(
                "  #%s: Actual=%.0f RUB, Predicted=%.0f RUB, Error=%.1f%%",
                i + 1, actual, predicted, error_pct
            )

        return self.metrics

    def analyze_coefficients(self) -> pd.DataFrame:
        """
        Analyze model coefficients.

        Returns:
            DataFrame with coefficients and their importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        model_step = self.model.named_steps['model']

        if hasattr(model_step, 'coef_'):
            coefficients = model_step.coef_
        else:
            logger.warning("Model doesn't have coefficients")
            return pd.DataFrame()

        if self.selected_feature_names:
            feature_names = self.selected_feature_names
        else:
            feature_names = self.feature_names

        if len(coefficients) != len(feature_names):
            logger.warning(
                "Coefficients count (%s) doesn't match feature names (%s)",
                len(coefficients), len(feature_names)
            )
            feature_names = [f'feature_{i}' for i in range(len(coefficients))]

        coef_df = pd.DataFrame({
            'feature': feature_names[:len(coefficients)],
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'importance_rank': range(1, len(coefficients) + 1)
        })

        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        coef_df['importance_rank'] = range(1, len(coef_df) + 1)

        logger.info("=" * 60)
        logger.info("TOP 15 MOST IMPORTANT FEATURES")
        logger.info("=" * 60)

        for i, (_, row) in enumerate(coef_df.head(15).iterrows()):
            impact = "Increases salary" if row['coefficient'] > 0 else "Decreases salary"
            logger.info(
                "  %2d. %-30s: %+.4f (%s)",
                i + 1, row['feature'], row['coefficient'], impact
            )

        return coef_df

    def save_model(self) -> Dict[str, str]:
        """
        Save trained model and metadata.

        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        logger.info("Saving model and metadata...")

        model_path = self.resources_dir / 'improved_linear_salary_model.joblib'
        joblib.dump(self.model, model_path)

        if self.scaler:
            scaler_path = self.resources_dir / 'scaler.joblib'
            joblib.dump(self.scaler, scaler_path)

        model_step = self.model.named_steps['model']
        model_type = type(model_step).__name__

        metadata = {
            'model_type': model_type,
            'feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names if self.selected_feature_names else self.feature_names,
            'selected_feature_indices': self.selected_feature_indices if self.selected_feature_indices else list(
                range(len(self.feature_names))),
            'n_features': len(self.feature_names),
            'metrics': self.metrics,
            'best_alpha': self.best_alpha if hasattr(self, 'best_alpha') else None,
            'top_correlated_features': [
                {'feature': name, 'correlation': corr}
                for _, _, corr, name in (self.top_correlated_features or [])
            ],
            'training_date': pd.Timestamp.now().isoformat(),
            'model_path': str(model_path.absolute()),
            'notes': 'Target variable is log1p(salary). Use expm1() to convert back to RUB.',
            'version': 'improved_2.0_with_CV'
        }

        metadata_path = self.resources_dir / 'improved_model_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        coef_df = self.analyze_coefficients()
        if not coef_df.empty:
            coef_path = self.resources_dir / 'coefficients.csv'
            coef_df.to_csv(coef_path, index=False, encoding='utf-8-sig')

        log_path = self.resources_dir / 'training_metrics.log'
        self._save_training_metrics_to_log(log_path)

        logger.info("Model saved to %s", model_path)
        logger.info("Metadata saved to %s", metadata_path)
        logger.info("Training metrics saved to %s", log_path)

        if not coef_df.empty:
            logger.info("Coefficients saved to %s", coef_path)

        return {
            'model': str(model_path),
            'metadata': str(metadata_path),
            'scaler': str(scaler_path) if self.scaler else None,
            'metrics_log': str(log_path)
        }

    def _save_training_metrics_to_log(self, log_path: Path) -> None:
        """
        Save training metrics to a separate log file.

        Args:
            log_path: Path to save the metrics log
        """
        # Используем простое сохранение в файл без логгера
        # чтобы избежать проблем с кодировкой символа ²
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("MODEL TRAINING METRICS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Training Date: {pd.Timestamp.now().isoformat()}\n")
                f.write("\n")

                if self.metrics:
                    f.write("TEST SET METRICS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"R-squared (RUB): {self.metrics.get('r2_rub', 0):.4f}\n")

                    # Преобразуем значения в float для корректного форматирования
                    mae_rub = float(self.metrics.get('mae_rub', 0))
                    rmse_rub = float(self.metrics.get('rmse_rub', 0))

                    f.write(f"MAE (RUB): {mae_rub:.4f}\n")
                    f.write(f"RMSE (RUB): {rmse_rub:.4f}\n")
                    f.write(f"MAPE: {self.metrics.get('mape', 0):.2f}%\n")
                    f.write(f"Within 20% error: {self.metrics.get('within_20_pct', 0):.2f}%\n")
                    f.write(f"Within 30% error: {self.metrics.get('within_30_pct', 0):.2f}%\n")
                    f.write("\n")

                if self.best_alpha:
                    f.write("MODEL PARAMETERS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Best alpha: {self.best_alpha}\n")
                    if hasattr(self, 'best_l1_ratio'):
                        f.write(f"Best l1_ratio: {self.best_l1_ratio}\n")
                    f.write("\n")

                if self.selected_feature_names:
                    f.write(f"SELECTED FEATURES ({len(self.selected_feature_names)} total):\n")
                    f.write("-" * 40 + "\n")
                    for i, feature in enumerate(self.selected_feature_names[:20], 1):
                        f.write(f"{i:3d}. {feature}\n")
                    if len(self.selected_feature_names) > 20:
                        f.write(f"... and {len(self.selected_feature_names) - 20} more\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")
        except Exception as e:
            logger.error("Error saving training metrics: %s", e)

    def load_model(self) -> Any:
        """
        Load trained model.

        Returns:
            Loaded model
        """
        model_path = self.resources_dir / 'improved_linear_salary_model.joblib'

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first."
            )

        self.model = joblib.load(model_path)

        metadata_path = self.resources_dir / 'improved_model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.selected_feature_names = metadata.get('selected_feature_names', [])
                self.selected_feature_indices = metadata.get('selected_feature_indices', [])
                self.metrics = metadata.get('metrics', {})
                self.best_alpha = metadata.get('best_alpha')

        logger.info("Model loaded from %s", model_path)

        return self.model

    def predict(self, X: np.ndarray, return_log: bool = False) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix
            return_log: If True, return predictions in log scale

        Returns:
            Predictions (in RUB by default)
        """
        if self.model is None:
            self.load_model()

        logger.info("Making predictions for %s samples...", X.shape[0])

        if self.selected_feature_indices and X.shape[1] > len(self.selected_feature_indices):
            X_selected = X[:, self.selected_feature_indices]
        else:
            X_selected = X

        y_pred_log = self.model.predict(X_selected)

        if return_log:
            return y_pred_log

        y_pred_rub = np.expm1(y_pred_log)
        y_pred_rub = np.clip(y_pred_rub, 10000, None)

        logger.info("Predictions generated:")
        logger.info(
            "   Range: [%.0f RUB, %.0f RUB]",
            y_pred_rub.min(), y_pred_rub.max()
        )
        logger.info("   Mean: %.0f RUB", y_pred_rub.mean())
        logger.info("   Median: %.0f RUB", np.median(y_pred_rub))

        return y_pred_rub


def train_improved_linear_model():
    """
    Main function for training improved linear model.

    This function handles command line arguments and orchestrates
    the training pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Train improved linear regression model for salary prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python linear_model.py --x-data files/x_data.npy --y-data files/y_data.npy
  python linear_model.py --x-data files/x_data.npy --y-data files/y_data.npy --model-type lasso
  python linear_model.py --x-data files/x_data.npy --y-data files/y_data.npy --n-features 15
        '''
    )

    parser.add_argument('--x-data', type=str, required=True,
                        help='Path to x_data.npy file')
    parser.add_argument('--y-data', type=str, required=True,
                        help='Path to y_data.npy file')
    parser.add_argument('--features', type=str,
                        help='Path to feature_names.txt file')
    parser.add_argument('--model-type', type=str, default='ridge',
                        choices=['ridge', 'lasso', 'elasticnet'],
                        help='Type of linear model with CV (default: ridge)')
    parser.add_argument('--n-features', type=int, default=15,
                        help='Number of top features to use (default: 15)')
    parser.add_argument('--resources-dir', type=str, default='resources',
                        help='Directory to save model resources (default: resources)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size proportion (default: 0.2)')

    args = parser.parse_args()

    predictor = ImprovedLinearSalaryPredictor(resources_dir=args.resources_dir)

    try:
        logger.info("=" * 60)
        logger.info("TRAINING IMPROVED LINEAR MODEL WITH CV")
        logger.info("=" * 60)

        X, y, feature_names = predictor.load_data(
            args.x_data, args.y_data, args.features
        )

        X_train, X_test, y_train, y_test = predictor.prepare_data(
            X, y, test_size=args.test_size
        )

        model = predictor.train_model(
            X_train, y_train,
            model_type=args.model_type,
            n_features=args.n_features
        )

        metrics = predictor.evaluate_model(X_test, y_test)

        predictor.analyze_coefficients()

        saved_paths = predictor.save_model()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        logger.info("Final Test R²: %.4f", metrics['r2_rub'])
        logger.info("Final Test MAE: %.4f RUB", float(metrics['mae_rub']))
        logger.info("Final Test MAPE: %.1f%%", metrics['mape'])
        logger.info("Within 20%% error: %.1f%%", metrics['within_20_pct'])

        if 'best_alpha' in metrics and metrics['best_alpha']:
            logger.info("Best regularization parameter: %s", metrics['best_alpha'])

        if metrics['r2_rub'] < 0.2:
            logger.error("VERY POOR PERFORMANCE (R² < 0.2)")
            logger.error("The model needs significant improvement.")
        elif metrics['r2_rub'] < 0.4:
            logger.warning("MODERATE PERFORMANCE (R² < 0.4)")
            logger.warning("The model has some predictive power but could be better.")
        elif metrics['r2_rub'] < 0.6:
            logger.info("GOOD PERFORMANCE (R² < 0.6)")
            logger.info("The model has solid predictive power.")
        else:
            logger.info("EXCELLENT PERFORMANCE (R² >= 0.6)")
            logger.info("The model has strong predictive power!")

    except Exception as e:
        logger.error("Error during training: %s", e)
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    train_improved_linear_model()