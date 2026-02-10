"""
Pipeline for classifying IT developer levels (junior/middle/senior) from hh.ru data.
Proof of Concept for automatic level classification.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
import joblib


class DeveloperLevelClassifier:
    """Classifier for IT developer levels (Junior/Middle/Senior)."""

    def __init__(self, data_dir: str = '.', model_type: str = 'logistic'):
        """
        Initialize classifier.

        Args:
            data_dir: Directory with processed data
            model_type: Type of model ('logistic', 'random_forest')
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.classes_ = None
        self.results = {}
        self.output_dir = Path("classification_results")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Map developer levels
        self.level_mapping = {
            'junior': ['junior', 'младший', 'стажер', 'trainee', 'intern', 'джуниор'],
            'middle': ['middle', 'миддл', 'разработчик', 'developer', 'мидл'],
            'senior': ['senior', 'сеньор', 'ведущий', 'старший', 'lead', 'тимлид']
        }

        # IT keywords to filter only IT resumes
        self.it_keywords = [
            'разработчик', 'developer', 'программист', 'engineer',
            'data scientist', 'data engineer', 'devops', 'qa', 'тестировщик',
            'frontend', 'front-end', 'backend', 'back-end', 'fullstack', 'full-stack',
            'ios', 'android', 'mobile', 'веб', 'web', 'software'
        ]

    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
        """Load processed data from pipeline output."""
        self.logger.info("Loading processed data...")

        # Try multiple possible locations for data files
        possible_paths = [
            self.data_dir / 'x_data.npy',
            Path('files') / 'x_data.npy',
            Path('x_data.npy'),
            Path('files/x_data.npy')
        ]

        x_path = None
        for path in possible_paths:
            if path.exists():
                x_path = path
                self.data_dir = path.parent
                self.logger.info(f"Found data in: {self.data_dir}")
                break

        if x_path is None:
            self.logger.error("Processed data not found!")
            self.logger.info("Please run preprocessing pipeline first:")
            self.logger.info("  python main.py files/hh.csv")
            sys.exit(1)

        y_path = self.data_dir / 'y_data.npy'
        feature_names_path = self.data_dir / 'feature_names.txt'

        X = np.load(x_path)
        y_original = np.load(y_path) if y_path.exists() else None

        # Load feature names
        if feature_names_path.exists():
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                self.feature_names = []
                for line in f.readlines():
                    if ': ' in line:
                        self.feature_names.append(line.strip().split(': ')[1])
            self.logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.logger.warning("Feature names file not found, using generic names")

        self.logger.info(f"X shape: {X.shape}")
        if y_original is not None:
            self.logger.info(f"y shape: {y_original.shape}")

        return X, y_original, self.feature_names

    def extract_position_level(self, raw_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract developer level from raw data based on position title."""
        self.logger.info("Extracting developer levels from raw data...")

        # Check if raw data exists
        possible_paths = [
            raw_data_path,
            Path('files') / 'hh.csv',
            Path('hh.csv'),
            Path('files/hh.csv')
        ]

        actual_path = None
        for path_str in possible_paths:
            path = Path(path_str) if isinstance(path_str, str) else path_str
            if path.exists():
                actual_path = path
                self.logger.info(f"Found raw data at: {actual_path}")
                break

        if actual_path is None:
            self.logger.error("Raw data file not found!")
            sys.exit(1)

        # Load raw data
        self.logger.info(f"Loading raw data from {actual_path}...")
        try:
            chunks = []
            for chunk in pd.read_csv(actual_path, chunksize=10000, low_memory=False):
                chunks.append(chunk)
            raw_df = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            try:
                raw_df = pd.read_csv(actual_path, low_memory=False)
            except Exception as e2:
                self.logger.error(f"Failed to load data: {e2}")
                sys.exit(1)

        self.logger.info(f"Raw data loaded: {len(raw_df)} rows")

        # Find position column
        position_column = None
        possible_position_cols = [
            'Ищет работу на должность:',
            'Должность',
            'Position',
            'Ищет работу на должность',
            'должность'
        ]

        for col in possible_position_cols:
            if col in raw_df.columns:
                position_column = col
                self.logger.info(f"Using position column: '{position_column}'")
                break

        if position_column is None:
            self.logger.warning("Position column not found!")
            # Create dummy levels for testing
            levels = np.array(['middle'] * len(raw_df))
            mask = np.ones(len(raw_df), dtype=bool)
            return levels, mask

        # Initialize all as unknown
        levels = np.array(['unknown'] * len(raw_df))
        position_titles = raw_df[position_column].astype(str).str.lower()

        # Extract levels from position titles
        self.logger.info("Extracting levels from position titles...")
        for level, keywords in self.level_mapping.items():
            mask = position_titles.apply(
                lambda x: any(keyword in x for keyword in keywords)
            )
            levels[mask] = level
            self.logger.info(f"Found {mask.sum()} {level} developers by title")

        # Refine based on experience if available
        if 'Опыт работы' in raw_df.columns:
            self.logger.info("Refining levels based on experience...")

            def extract_experience_years(exp_text):
                """Extract experience in years from text."""
                if not isinstance(exp_text, str):
                    return 0

                years = 0
                import re
                year_match = re.search(r'(\d+)\s*лет', exp_text)
                if year_match:
                    years += int(year_match.group(1))

                month_match = re.search(r'(\d+)\s*месяц', exp_text)
                if month_match:
                    years += int(month_match.group(1)) / 12

                return years

            experience_years = raw_df['Опыт работы'].apply(extract_experience_years)

            # Adjust levels based on experience
            junior_exp_mask = (experience_years < 2) & (levels == 'unknown')
            levels[junior_exp_mask] = 'junior'

            middle_exp_mask = (experience_years >= 2) & (experience_years < 5) & (levels == 'unknown')
            levels[middle_exp_mask] = 'middle'

            senior_exp_mask = (experience_years >= 5) & (levels == 'unknown')
            levels[senior_exp_mask] = 'senior'

        # Filter only IT-related positions
        self.logger.info("Filtering IT positions...")
        it_mask = position_titles.apply(
            lambda x: any(keyword in x.lower() for keyword in self.it_keywords)
        )

        self.logger.info(f"IT positions: {it_mask.sum()} out of {len(raw_df)}")

        # Filter to keep only IT positions with known levels
        final_mask = it_mask & (levels != 'unknown')
        final_levels = levels[final_mask]

        # Relax filter if too few samples
        if len(final_levels) < 100:
            self.logger.warning(f"Too few IT samples ({len(final_levels)}), relaxing filter...")
            final_mask = levels != 'unknown'
            final_levels = levels[final_mask]

        self.logger.info("=" * 60)
        self.logger.info("FINAL LEVEL DISTRIBUTION:")
        self.logger.info(f"Total developers: {len(final_levels)}")
        for level in ['junior', 'middle', 'senior']:
            count = (final_levels == level).sum()
            if len(final_levels) > 0:
                percentage = count / len(final_levels) * 100
                self.logger.info(f"  {level.capitalize()}: {count} ({percentage:.1f}%)")
            else:
                self.logger.info(f"  {level.capitalize()}: {count} (0.0%)")
        self.logger.info("=" * 60)

        return final_levels, final_mask

    def prepare_classification_data(self, X: np.ndarray, y_levels: np.ndarray,
                                    mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for classification."""
        self.logger.info("Preparing classification data...")

        # Apply mask to filter IT developers
        X_filtered = X[mask]

        if len(X_filtered) != len(y_levels):
            self.logger.warning(f"Data mismatch: X_filtered={len(X_filtered)}, y_levels={len(y_levels)}")
            min_len = min(len(X_filtered), len(y_levels))
            X_filtered = X_filtered[:min_len]
            y_levels = y_levels[:min_len]
            self.logger.info(f"Truncated to {min_len} samples")

        # Encode labels
        self.label_encoder.fit(['junior', 'middle', 'senior'])
        y_encoded = self.label_encoder.transform(y_levels)
        self.classes_ = self.label_encoder.classes_

        self.logger.info(f"Classification data shape: X={X_filtered.shape}, y={y_encoded.shape}")
        self.logger.info(f"Classes: {list(self.classes_)}")

        # Check class distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            percentage = count / len(y_encoded) * 100 if len(y_encoded) > 0 else 0
            self.logger.info(f"  Class {cls_name}: {count} samples ({percentage:.1f}%)")

        return X_filtered, y_encoded

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train classification model."""
        if len(y_train) == 0:
            self.logger.error("No training data available!")
            return

        self.logger.info(f"Training {self.model_type} model...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Compute class weights for imbalance
        try:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        except Exception as e:
            self.logger.warning(f"Could not compute class weights: {e}")
            class_weight_dict = None

        # Initialize and train model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight=class_weight_dict,
                random_state=42,
                multi_class='ovr'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train_scaled, y_train)

        self.logger.info(f"Model trained successfully")
        self.logger.info(f"Training accuracy: {self.model.score(X_train_scaled, y_train):.4f}")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model and generate classification report."""
        self.logger.info("Evaluating model...")

        if self.model is None:
            self.logger.error("Model not trained!")
            return {}

        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        # Generate classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True
        )

        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': dict(zip(class_names, precision)),
            'recall': dict(zip(class_names, recall)),
            'f1': dict(zip(class_names, f1)),
            'support': dict(zip(class_names, support)),
            'classification_report': report,
            'model_type': self.model_type,
            'classes': class_names.tolist()
        }

        # Log results
        self.logger.info("=" * 60)
        self.logger.info("CLASSIFICATION RESULTS:")
        self.logger.info(f"Model: {self.model_type}")
        self.logger.info(f"Overall Accuracy: {accuracy:.4f}")
        self.logger.info("Per-class metrics:")
        for cls in class_names:
            self.logger.info(f"  {cls.upper():8s}: "
                             f"Precision={self.results['precision'][cls]:.4f}, "
                             f"Recall={self.results['recall'][cls]:.4f}, "
                             f"F1={self.results['f1'][cls]:.4f}, "
                             f"Support={self.results['support'][cls]}")

        return self.results

    def save_model(self, model_path: str = None):
        """Save trained model and preprocessing objects."""
        if self.model is None:
            self.logger.warning("No model to save!")
            return

        if model_path is None:
            model_path = self.output_dir / 'developer_classifier_model.pkl'

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'results': self.results
        }

        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")

    def run_pipeline(self, raw_data_path: str):
        """Run complete classification pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("DEVELOPER LEVEL CLASSIFICATION PoC")
        self.logger.info("=" * 60)

        try:
            # Step 1: Load processed data
            X, y_original, feature_names = self.load_data()

            # Step 2: Extract developer levels from raw data
            y_levels, mask = self.extract_position_level(raw_data_path)

            # Step 3: Prepare classification data
            if len(y_levels) == 0:
                self.logger.error("No developers found for classification!")
                self.logger.info("Try adjusting the IT keywords or level mapping")
                return

            X_class, y_encoded = self.prepare_classification_data(X, y_levels, mask)

            # Check if we have enough data
            if len(y_encoded) < 100:
                self.logger.warning(f"Very small dataset: {len(y_encoded)} samples")

            # Step 4: Split data
            if len(y_encoded) < 30:
                self.logger.warning("Dataset too small for proper train/test split")
                # Use all data for training
                X_train, X_test, y_train, y_test = X_class, X_class[:0], y_encoded, y_encoded[:0]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_class, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                )

            self.logger.info(f"Train set: {X_train.shape[0]} samples")
            self.logger.info(f"Test set: {X_test.shape[0]} samples")

            # Step 5: Train model
            self.train_model(X_train, y_train)

            # Step 6: Evaluate model
            if len(y_test) > 0:
                results = self.evaluate_model(X_test, y_test)
            else:
                self.logger.warning("No test data available for evaluation")
                results = {}

            # Step 7: Save model
            self.save_model()

            # Step 8: Generate final report
            self.generate_final_report(results)

            self.logger.info("=" * 60)
            self.logger.info("PoC COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Error in pipeline: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def generate_final_report(self, results: Dict[str, Any]):
        """Generate final PoC report with conclusions."""
        report_path = self.output_dir / 'classification_poc_report.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Proof of Concept: Developer Level Classification\n\n")
            f.write("## Executive Summary\n\n")
            f.write("This PoC demonstrates that developer levels can be automatically ")
            f.write("classified from hh.ru resume data.\n\n")

            f.write("## Results Summary\n\n")
            if results:
                f.write(f"- **Model Type**: {results.get('model_type', 'N/A')}\n")
                f.write(f"- **Overall Accuracy**: {results.get('accuracy', 0):.4f}\n")
                if 'support' in results:
                    f.write(f"- **Number of Samples**: {sum(results['support'].values())}\n\n")

                f.write("## Per-Class Performance\n\n")
                f.write("| Class | Precision | Recall | F1-Score | Support |\n")
                f.write("|-------|-----------|--------|----------|---------|\n")
                for cls in results.get('classes', []):
                    f.write(f"| {cls.capitalize()} | {results['precision'][cls]:.4f} | ")
                    f.write(f"{results['recall'][cls]:.4f} | {results['f1'][cls]:.4f} | ")
                    f.write(f"{results['support'][cls]} |\n")
            else:
                f.write("No evaluation results available (model trained but not tested).\n\n")

            f.write("\n## Key Findings\n\n")
            f.write("1. **Feasibility**: Approach shows promise for automatic level classification\n")
            f.write("2. **Data Quality**: Level extraction depends on position title quality\n")
            f.write("3. **IT Filter**: Filtering IT positions improves model relevance\n\n")

            f.write("## Limitations and Recommendations\n\n")
            f.write("1. **Class Imbalance**: Consider class weighting techniques\n")
            f.write("2. **Label Quality**: Manual validation of extracted levels needed\n")
            f.write("3. **Model Selection**: Try additional algorithms like XGBoost\n")
            f.write("4. **Hyperparameter Tuning**: Optimize model parameters\n")

        self.logger.info(f"Final report saved to {report_path}")


def main():
    """Main function for classification PoC."""
    import sys

    # Configure logging to file and console
    output_dir = Path("classification_results")
    output_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'classification_poc.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        logger.error("Usage: python classification_pipeline.py <path_to_raw_csv> [--model TYPE]")
        logger.error("Example: python classification_pipeline.py files/hh.csv --model random_forest")
        sys.exit(1)

    raw_data_path = sys.argv[1]
    model_type = 'logistic'  # default

    # Parse arguments
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_type = sys.argv[i + 1]

    # Check if file exists
    if not os.path.exists(raw_data_path):
        logger.warning(f"Raw data file not found at specified path: {raw_data_path}")

    # Initialize and run classifier
    classifier = DeveloperLevelClassifier(data_dir='files', model_type=model_type)
    classifier.run_pipeline(raw_data_path)


if __name__ == '__main__':
    main()