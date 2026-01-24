"""
Data handlers for processing pipeline.
Uses Chain of Responsibility pattern.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from .chain import Handler

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader(Handler):
    """Load CSV file with large file support."""

    def handle(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}...")

        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=10000, low_memory=False):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df)} rows.")

        return self._next(df)


class DataCleaner(Handler):
    """Data cleaning: remove duplicates, extra spaces, special characters."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data from noise and duplicates.

        Args:
            df: Source DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")

        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)

        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicates.")

        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = self._clean_text_column(df[col])

        logger.info(f"After cleaning: {len(df)} rows.")
        return self._next(df)

    @staticmethod
    def _clean_text_column(series: pd.Series) -> pd.Series:
        """Clean text column from special characters and extra spaces.

        Args:
            series: Source Series

        Returns:
            Cleaned Series
        """
        cleaned = series.astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
        cleaned = cleaned.str.replace('\xa0', ' ', regex=False)
        cleaned = cleaned.str.replace('\ufeff', '', regex=False)

        return cleaned


class FeatureExtractor(Handler):
    """Extract and create features from raw data."""

    def __init__(self, salary_threshold: Optional[float] = None):
        """Initialize feature extractor.

        Args:
            salary_threshold: Threshold for salary outlier filtering (in RUB)
        """
        super().__init__()
        self.salary_threshold = salary_threshold

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and create new features from data.

        Args:
            df: Source DataFrame

        Returns:
            DataFrame with new features
        """
        logger.info("Extracting features...")

        if 'Пол, возраст' in df.columns:
            df['Возраст'] = df['Пол, возраст'].apply(self._extract_age)
            df['Пол'] = df['Пол, возраст'].apply(self._extract_gender)

        if 'ЗП' in df.columns:
            df['ЗП_число'] = df['ЗП'].apply(self._normalize_salary)
            salaries = df['ЗП_число']

            logger.info(f"Salaries before processing: min={salaries.min():.0f}, "
                       f"max={salaries.max():.0f}, median={salaries.median():.0f}")

            if self.salary_threshold:
                df = self._handle_salary_outliers(df)

        if 'Город' in df.columns:
            df['Город_очищенный'] = df['Город'].apply(self._extract_city)

        if 'Опыт работы' in df.columns:
            df['Опыт_месяцы'] = df['Опыт работы'].apply(self._extract_experience)

        if 'Авто' in df.columns:
            df['Есть_авто'] = df['Авто'].apply(
                lambda x: 1 if 'Имеется' in str(x) else 0
            )
            df = df.drop(columns=['Авто'])

        return self._next(df)

    def _handle_salary_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process salary outliers.

        Args:
            df: DataFrame with salaries

        Returns:
            DataFrame with processed salaries
        """
        salaries = df['ЗП_число']

        if self.salary_threshold:
            outliers_mask = salaries > self.salary_threshold
            outlier_count = outliers_mask.sum()

            if outlier_count > 0:
                logger.info(f"Found {outlier_count} salary outliers "
                           f"(>{self.salary_threshold:.0f} RUB)")

                median_salary = salaries[~outliers_mask].median()
                df.loc[outliers_mask, 'ЗП_число'] = median_salary
                logger.info(f"Outliers replaced with median: {median_salary:.0f} RUB")

        df['ЗП_лог'] = np.log1p(df['ЗП_число'])
        df = df.drop(columns=['ЗП_число'])
        df = df.rename(columns={'ЗП_лог': 'ЗП_число'})

        logger.info("Applied logarithmic salary transformation")
        return df

    @staticmethod
    def _extract_age(text: str) -> int:
        """Extract age from string.

        Args:
            text: Text like "Мужчина, 42 года, родился..."

        Returns:
            Age in years or -1 if not found
        """
        match = re.search(r'(\d+)\s*год', text)
        return int(match.group(1)) if match else -1

    @staticmethod
    def _extract_gender(text: str) -> str:
        """Extract gender from string.

        Args:
            text: Text with gender information

        Returns:
            'Мужчина', 'Женщина' or 'Не указано'
        """
        if 'Мужчина' in text:
            return 'Мужчина'
        elif 'Женщина' in text:
            return 'Женщина'
        return 'Не указано'

    @staticmethod
    def _normalize_salary(salary: str) -> float:
        """Normalize salary string to number.

        Args:
            salary: Salary string like "27 000 руб."

        Returns:
            Numeric salary value or 0.0
        """
        if pd.isna(salary):
            return 0.0

        salary_str = str(salary)
        salary_str = re.sub(r'[^\d\s.,]', '', salary_str)
        salary_str = re.sub(r'\s+', '', salary_str)
        salary_str = salary_str.replace(',', '.')

        try:
            return float(salary_str) if salary_str else 0.0
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _extract_city(city: str) -> str:
        """Extract main city from string.

        Args:
            city: String with city information

        Returns:
            City name or empty string
        """
        if not isinstance(city, str):
            return ''

        parts = city.split(',')
        return parts[0].strip() if parts else ''

    @staticmethod
    def _extract_experience(experience_text: str) -> int:
        """Extract work experience in months.

        Args:
            experience_text: Text with work experience

        Returns:
            Experience in months or 0
        """
        if not isinstance(experience_text, str):
            return 0

        total_months = 0
        year_match = re.search(r'(\d+)\s*лет', experience_text)

        if year_match:
            total_months += int(year_match.group(1)) * 12

        month_match = re.search(r'(\d+)\s*месяц', experience_text)
        if month_match:
            total_months += int(month_match.group(1))

        return total_months


class MulticollinearityHandler(Handler):
    """Handle feature multicollinearity."""

    def __init__(self, correlation_threshold: float = 0.95):
        """Initialize multicollinearity handler.

        Args:
            correlation_threshold: Correlation threshold for feature removal
        """
        super().__init__()
        self.correlation_threshold = correlation_threshold

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame without multicollinear features
        """
        logger.info("Processing multicollinearity...")

        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            logger.info("Too few numeric features for analysis")
            return self._next(df)

        logger.info(f"Analyzing {len(numeric_df.columns)} numeric features")

        corr_matrix = numeric_df.corr().abs()
        to_drop = self._find_correlated_features(corr_matrix)

        if to_drop:
            logger.info(f"Removed {len(to_drop)} features due to high correlation")

            for feature in to_drop:
                if feature in numeric_df.columns:
                    correlations = corr_matrix[feature]
                    high_corr = correlations[correlations > self.correlation_threshold]
                    high_corr = high_corr.drop(feature)

                    if not high_corr.empty:
                        correlated_with = high_corr.index[0]
                        corr_value = high_corr.iloc[0]
                        logger.debug(f"  {feature} (correlation {corr_value:.3f} with '{correlated_with}')")

            df = df.drop(columns=to_drop)
            logger.info(f"Remaining features: {df.shape[1]}")
        else:
            logger.info("No high correlations found")

        return self._next(df)

    def _find_correlated_features(self, corr_matrix: pd.DataFrame) -> List[str]:
        """Find highly correlated features for removal.

        Args:
            corr_matrix: Correlation matrix (absolute values)

        Returns:
            List of features to remove
        """
        corr_matrix = corr_matrix.copy()
        corr_values = corr_matrix.values.copy()
        np.fill_diagonal(corr_values, 0)
        corr_matrix = pd.DataFrame(corr_values,
                                   index=corr_matrix.index,
                                   columns=corr_matrix.columns)

        to_drop = []
        columns = list(corr_matrix.columns)

        for i, col1 in enumerate(columns):
            if col1 in to_drop:
                continue

            for col2 in columns[i + 1:]:
                if col2 in to_drop:
                    continue

                corr = corr_matrix.loc[col1, col2]

                if corr > self.correlation_threshold:
                    corr_count_col1 = (corr_matrix[col1] > self.correlation_threshold).sum()
                    corr_count_col2 = (corr_matrix[col2] > self.correlation_threshold).sum()

                    if corr_count_col1 >= corr_count_col2:
                        to_drop.append(col2)
                    else:
                        to_drop.append(col1)
                        break

        return to_drop


class MissingValueHandler(Handler):
    """Handle missing values."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in data.

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame without missing values
        """
        logger.info("Processing missing values...")

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            imputer = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            mode_values = df[col].mode()
            fill_value = mode_values[0] if not mode_values.empty else 'Не указано'
            df[col] = df[col].fillna(fill_value)

        return self._next(df)


class CategoricalEncoder(Handler):
    """Encode categorical features with multiple strategy support."""

    def __init__(self, max_categories_for_ohe: int = 15,
                 special_columns: dict = None,
                 drop_first: bool = True):
        """Initialize encoder.

        Args:
            max_categories_for_ohe: Maximum categories for One-Hot Encoding
            special_columns: Special handling for specific columns
                             Format: {'column_name': 'method'}
                             Methods: 'label', 'binary', 'frequency', 'onehot'
            drop_first: Whether to drop first column in One-Hot Encoding
        """
        super().__init__()
        self.max_categories_for_ohe = max_categories_for_ohe
        self.special_columns = special_columns or {}
        self.drop_first = drop_first
        self.encoders = {}
        self.column_order = []

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features.

        Args:
            df: DataFrame with categorical features

        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")

        categorical_columns = df.select_dtypes(include=['object']).columns
        logger.info(f"Found {len(categorical_columns)} categorical features")

        self.column_order = list(df.columns)

        for col in categorical_columns:
            if col not in df.columns:
                continue

            unique_count = df[col].nunique()
            logger.debug(f"'{col}': {unique_count} unique values")

            if col in self.special_columns:
                method = self.special_columns[col]
                df = self._apply_special_encoding(df, col, method)
                continue

            if unique_count == 1:
                df = df.drop(columns=[col])
                logger.debug(f"Removed feature '{col}' (only one unique value)")

            elif unique_count == 2:
                df = self._apply_binary_encoding(df, col)
                logger.debug("Applied binary encoding (0/1)")

            elif 3 <= unique_count <= self.max_categories_for_ohe:
                df = self._apply_one_hot_encoding(df, col)
                logger.debug(f"Applied One-Hot Encoding (drop_first={self.drop_first})")

            elif unique_count > self.max_categories_for_ohe:
                df = self._apply_frequency_encoding(df, col)
                logger.debug("Applied Frequency Encoding")

            self.column_order = list(df.columns)

        logger.info(f"After encoding: {df.shape[1]} features")
        return self._next(df)

    def _apply_special_encoding(self, df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
        """Apply special encoding for column.

        Args:
            df: Source DataFrame
            column: Column name to encode
            method: Encoding method ('label', 'binary', 'frequency', 'onehot')

        Returns:
            DataFrame with encoded column
        """
        if method == 'label':
            return self._apply_label_encoding(df, column)
        elif method == 'binary':
            return self._apply_binary_encoding(df, column)
        elif method == 'frequency':
            return self._apply_frequency_encoding(df, column)
        elif method == 'onehot':
            return self._apply_one_hot_encoding(df, column)
        else:
            logger.warning(f"Unknown encoding method '{method}' for column '{column}'")
            return df

    def _apply_label_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply Label Encoding to column.

        Args:
            df: Source DataFrame
            column: Column name to encode

        Returns:
            DataFrame with encoded column
        """
        try:
            le = LabelEncoder()
            encoded_values = le.fit_transform(df[column])
            df[column] = encoded_values.astype(np.float64)

            self.encoders[column] = {
                'type': 'label',
                'encoder': le,
                'mapping': dict(zip(le.classes_, range(len(le.classes_))))
            }

            logger.debug(f"Label Encoding: {self.encoders[column]['mapping']}")

        except Exception as e:
            logger.error(f"Label Encoding error for '{column}': {e}")
            df = self._apply_frequency_encoding(df, column)

        return df

    def _apply_binary_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply binary encoding (0/1) to column.

        Args:
            df: Source DataFrame
            column: Column name to encode

        Returns:
            DataFrame with encoded column
        """
        try:
            unique_values = df[column].unique()

            if len(unique_values) != 2:
                logger.warning(f"Column '{column}' has {len(unique_values)} values, expected 2")
                return self._apply_one_hot_encoding(df, column)

            mapping = {unique_values[0]: 0.0, unique_values[1]: 1.0}
            df[column] = df[column].map(mapping).astype(np.float64)

            self.encoders[column] = {'type': 'binary', 'mapping': mapping}
            logger.debug(f"Binary Encoding: {mapping}")

        except Exception as e:
            logger.error(f"Binary Encoding error for '{column}': {e}")
            df = self._apply_one_hot_encoding(df, column)

        return df

    def _apply_one_hot_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply One-Hot Encoding to column.

        Args:
            df: Source DataFrame
            column: Column name to encode

        Returns:
            DataFrame with encoded column
        """
        try:
            dummies = pd.get_dummies(
                df[column],
                prefix=column,
                dtype=np.float64,
                drop_first=self.drop_first
            )

            created_columns = list(dummies.columns)
            logger.debug(f"Created {len(created_columns)} one-hot features")

            df = pd.concat([df.drop(columns=[column]), dummies], axis=1)

            self.encoders[column] = {
                'type': 'onehot',
                'created_columns': created_columns,
                'drop_first': self.drop_first,
                'original_categories': df[column].unique() if column in df.columns else []
            }

        except Exception as e:
            logger.error(f"One-Hot Encoding error for '{column}': {e}")
            df = self._apply_frequency_encoding(df, column)

        return df

    def _apply_frequency_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply Frequency Encoding to column.

        Args:
            df: Source DataFrame
            column: Column name to encode

        Returns:
            DataFrame with encoded column
        """
        try:
            freq = df[column].value_counts(normalize=True)
            df[column] = df[column].map(freq).astype(np.float64)

            self.encoders[column] = {'type': 'frequency', 'mapping': freq.to_dict()}
            logger.debug(f"Frequency Encoding: {len(freq)} unique values")

        except Exception as e:
            logger.error(f"Frequency Encoding error for '{column}': {e}")
            df = df.drop(columns=[column])
            logger.debug(f"Removed feature '{column}' due to encoding error")

        return df

    def get_encoding_info(self) -> dict:
        """Return information about applied encodings.

        Returns:
            Dictionary with encoding information
        """
        return self.encoders

    def get_column_order(self) -> list:
        """Return current column order.

        Returns:
            List of column names in current order
        """
        return self.column_order


class FeatureMetadataSaver(Handler):
    """Save feature information (names, types) to separate file."""

    def __init__(self, output_dir: str = '.'):
        """Initialize metadata saver.

        Args:
            output_dir: Directory for saving files
        """
        super().__init__()
        self.output_dir = output_dir
        self.feature_names = []

    def handle(self, data: Any) -> Any:
        """Save feature metadata.

        Args:
            data: DataFrame with features or other data

        Returns:
            Original data without changes
        """
        if isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)
            logger.info("Saving feature metadata...")
            logger.info(f"Found {len(self.feature_names)} features")

            metadata = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'data_shape': data.shape,
                'data_types': {col: str(data[col].dtype) for col in data.columns}
            }

            metadata_path = f'{self.output_dir}/feature_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Metadata saved to {metadata_path}")

            txt_path = f'{self.output_dir}/feature_names.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                for i, name in enumerate(self.feature_names):
                    f.write(f"{i}: {name}\n")

            logger.info(f"Feature names saved to {txt_path}")

        return self._next(data)


class OutlierHandler(Handler):
    """Handle data outliers."""

    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        """Initialize outlier handler.

        Args:
            method: Processing method ('iqr', 'zscore', 'clip', 'remove')
            threshold: Threshold for outlier detection
        """
        super().__init__()
        self.method = method
        self.threshold = threshold

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric features.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with processed outliers
        """
        logger.info("Processing outliers...")

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            initial_outliers = self._count_outliers(df[col], self.method, self.threshold)

            if initial_outliers > 0:
                logger.info(f"{col}: {initial_outliers} outliers "
                           f"({initial_outliers/len(df)*100:.1f}%)")

                if self.method == 'clip':
                    df[col] = self._clip_outliers(df[col], self.method, self.threshold)
                    logger.debug("Outliers clipped")
                elif self.method == 'remove':
                    pass
                elif self.method in ['iqr', 'zscore']:
                    df[col] = self._winsorize_outliers(df[col], self.method, self.threshold)
                    logger.debug("Outliers replaced with boundary values")

        if self.method == 'remove':
            mask = pd.Series(True, index=df.index)
            for col in numeric_columns:
                outlier_mask = self._get_outlier_mask(df[col], self.method, self.threshold)
                mask &= ~outlier_mask

            rows_removed = len(df) - mask.sum()
            if rows_removed > 0:
                df = df[mask]
                logger.info(f"Removed {rows_removed} rows with outliers")

        return self._next(df)

    @staticmethod
    def _count_outliers(series: pd.Series, method: str, threshold: float) -> int:
        """Count outliers in series.

        Args:
            series: Value vector
            method: Outlier detection method
            threshold: Threshold

        Returns:
            Number of outliers
        """
        mask = OutlierHandler._get_outlier_mask(series, method, threshold)
        return mask.sum()

    @staticmethod
    def _get_outlier_mask(series: pd.Series, method: str, threshold: float) -> pd.Series:
        """Return outlier mask.

        Args:
            series: Value vector
            method: Outlier detection method
            threshold: Threshold

        Returns:
            Boolean Series mask
        """
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (series < lower_bound) | (series > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        return pd.Series(False, index=series.index)

    @staticmethod
    def _clip_outliers(series: pd.Series, method: str, threshold: float) -> pd.Series:
        """Clip outliers to boundary values.

        Args:
            series: Value vector
            method: Outlier detection method
            threshold: Threshold

        Returns:
            Series with clipped outliers
        """
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return series.clip(lower_bound, upper_bound)

        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            return series.clip(lower_bound, upper_bound)

        return series

    @staticmethod
    def _winsorize_outliers(series: pd.Series, method: str, threshold: float) -> pd.Series:
        """Replace outliers with boundary values (winsorization).

        Args:
            series: Value vector
            method: Outlier detection method
            threshold: Threshold

        Returns:
            Series with processed outliers
        """
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            result = series.copy()
            result[result < lower_bound] = lower_bound
            result[result > upper_bound] = upper_bound
            return result

        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

            result = series.copy()
            result[result < lower_bound] = lower_bound
            result[result > upper_bound] = upper_bound
            return result

        return series


class Normalizer(Handler):
    """Normalize numeric features."""

    def __init__(self, use_robust_scaler: bool = False, use_standard_scaler: bool = True):
        """Initialize normalizer.

        Args:
            use_robust_scaler: Use RobustScaler (better for data with outliers)
            use_standard_scaler: Apply StandardScaler after RobustScaler for std=1
        """
        super().__init__()
        self.use_robust_scaler = use_robust_scaler
        self.use_standard_scaler = use_standard_scaler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features.

        Args:
            df: DataFrame with numeric features

        Returns:
            Normalized DataFrame
        """
        logger.info("Normalizing numeric features...")

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 0:
            df[numeric_columns] = df[numeric_columns].apply(
                pd.to_numeric, errors='coerce'
            )
            df[numeric_columns] = df[numeric_columns].fillna(0)

            if self.use_robust_scaler:
                logger.info("Using RobustScaler (robust to outliers)")
                robust_scaler = RobustScaler()
                df[numeric_columns] = robust_scaler.fit_transform(df[numeric_columns])

            if self.use_standard_scaler:
                logger.info("Using StandardScaler (mean=0, std=1)")
                standard_scaler = StandardScaler()
                df[numeric_columns] = standard_scaler.fit_transform(df[numeric_columns])

            df[numeric_columns] = df[numeric_columns].astype(np.float64)

        return self._next(df)


class TargetSplitter(Handler):
    """Split into features (X) and target variable (y)."""

    def __init__(self, target_column: str = 'ЗП_число'):
        """Initialize splitter.

        Args:
            target_column: Name of target column
        """
        super().__init__()
        self.target_column = target_column
        self.feature_names = []

    def handle(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Split data into features and target variable.

        Args:
            df: Full DataFrame

        Returns:
            Tuple (X, y, feature_names) where X - features, y - target,
            feature_names - list of feature names

        Raises:
            ValueError: If target column not found
        """
        logger.info("Splitting into X and y...")

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        self.feature_names = [col for col in df.columns if col != self.target_column]

        logger.info(f"Found {len(self.feature_names)} features")
        for i, name in enumerate(self.feature_names[:20]):
            logger.debug(f"[{i}] {name}")

        if len(self.feature_names) > 20:
            logger.debug(f"... and {len(self.feature_names) - 20} more features")

        y = self._prepare_target(df[self.target_column])
        X_df = df.drop(columns=[self.target_column])
        X = self._prepare_features(X_df)

        self._log_split_info(X, y)

        return self._next((X, y, self.feature_names))

    @staticmethod
    def _prepare_target(target_series: pd.Series) -> np.ndarray:
        """Prepare target variable.

        Args:
            target_series: Series with target variable

        Returns:
            Numpy array with target variable
        """
        y = pd.to_numeric(target_series, errors='coerce')
        median_val = y.median()
        y = y.fillna(median_val)

        if y.skew() > 1.0:
            logger.info(f"Applying logarithmic transformation to target variable "
                       f"(skew={y.skew():.2f})")
            y = np.log1p(y)

        return y.values.astype(np.float64)

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix.

        Args:
            df: DataFrame with features

        Returns:
            Feature matrix as numpy array
        """
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        return df.values.astype(np.float64)

    @staticmethod
    def _log_split_info(X: np.ndarray, y: np.ndarray) -> None:
        """Log information about data split.

        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
        logger.debug(f"X min/max/mean: [{X.min():.4f}, {X.max():.4f}, {X.mean():.4f}]")
        logger.debug(f"y min/max/mean: [{y.min():.4f}, {y.max():.4f}, {y.mean():.4f}]")
        logger.debug(f"y std/skew/kurtosis: [{y.std():.4f}, {stats.skew(y):.4f}, "
                    f"{stats.kurtosis(y):.4f}]")


class NumpySaver(Handler):
    """Save data to .npy files with type checking."""

    def __init__(self, output_dir: str = '.'):
        """Initialize saver.

        Args:
            output_dir: Directory for saving files
        """
        super().__init__()
        self.output_dir = output_dir

    def handle(self, data: Tuple[np.ndarray, np.ndarray, list]) -> None:
        """Save data to .npy files with metadata.

        Args:
            data: Tuple (X, y, feature_names) to save
        """
        X, y, feature_names = data
        logger.info("Saving to .npy files...")

        X, y = self._ensure_correct_dtypes(X, y)

        if X.shape[1] != len(feature_names):
            logger.warning(f"X has {X.shape[1]} columns, "
                          f"feature_names contains {len(feature_names)} names")

            if len(feature_names) > X.shape[1]:
                feature_names = feature_names[:X.shape[1]]
                logger.info(f"Trimmed feature names to {len(feature_names)}")
            elif len(feature_names) < X.shape[1]:
                missing_count = X.shape[1] - len(feature_names)
                for i in range(missing_count):
                    feature_names.append(f"feature_{len(feature_names) + i}")
                logger.info(f"Added {missing_count} generic names")

        self._validate_data_quality(X, y)
        x_path, y_path = self._save_files(X, y)
        self._save_feature_names(feature_names)
        self._verify_saved_files(x_path, y_path)

        return self._next(None)

    @staticmethod
    def _ensure_correct_dtypes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure data has correct types.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple (X, y) with fixed types
        """
        if X.dtype != np.float64:
            logger.info(f"Converting X to float64 (was {X.dtype})")
            X = X.astype(np.float64)

        if y.dtype != np.float64:
            logger.info(f"Converting y to float64 (was {y.dtype})")
            y = y.astype(np.float64)

        return X, y

    @staticmethod
    def _validate_data_quality(X: np.ndarray, y: np.ndarray) -> None:
        """Check data quality before saving.

        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.debug("Checking data quality...")

        x_nan = np.isnan(X).sum()
        y_nan = np.isnan(y).sum()
        x_inf = np.isinf(X).sum()
        y_inf = np.isinf(y).sum()

        if x_nan > 0 or y_nan > 0 or x_inf > 0 or y_inf > 0:
            logger.warning("Problematic values detected")
            logger.debug(f"NaN: X={x_nan}, y={y_nan}")
            logger.debug(f"Inf: X={x_inf}, y={y_inf}")

        logger.debug(f"X statistics: mean={X.mean():.4f}, std={X.std():.4f}")
        logger.debug(f"y statistics: mean={y.mean():.4f}, std={y.std():.4f}, "
                    f"skew={stats.skew(y):.4f}")

        if abs(y.mean()) > 1.0 or y.std() > 10.0:
            logger.warning("y has unusual statistics")

        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X, rowvar=False)
            high_corr_count = 0

            for i in range(corr_matrix.shape[0]):
                for j in range(i+1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > 0.99:
                        high_corr_count += 1

            if high_corr_count > 0:
                logger.warning(f"Found {high_corr_count} feature pairs with correlation > 0.99")
                logger.debug("Possible dummy variable trap in One-Hot Encoding")

    def _save_files(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, str]:
        """Save arrays to files.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of paths to saved files
        """
        x_path = f"{self.output_dir}/x_data.npy"
        y_path = f"{self.output_dir}/y_data.npy"

        np.save(x_path, X)
        np.save(y_path, y)

        logger.info(f"Saved: {x_path} ({X.shape}, {X.dtype})")
        logger.info(f"Saved: {y_path} ({y.shape}, {y.dtype})")

        return x_path, y_path

    def _save_feature_names(self, feature_names: list) -> None:
        """Save feature names to separate file.

        Args:
            feature_names: List of feature names
        """
        feature_names_path = f"{self.output_dir}/feature_names.txt"

        with open(feature_names_path, 'w', encoding='utf-8') as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i}: {name}\n")

        logger.info(f"Feature names saved to {feature_names_path}")

        metadata = {
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'note': 'Saved with x_data.npy and y_data.npy'
        }

        json_path = f"{self.output_dir}/feature_names.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Feature metadata saved to {json_path}")

    def _verify_saved_files(self, x_path: str, y_path: str) -> None:
        """Verify correctness of saved files.

        Args:
            x_path: Path to X file
            y_path: Path to y file
        """
        logger.debug("Verifying saved files...")

        try:
            X_loaded = np.load(x_path)
            y_loaded = np.load(y_path)

            logger.info("Files loaded correctly")
            logger.debug(f"X: {X_loaded.shape}, {X_loaded.dtype}")
            logger.debug(f"y: {y_loaded.shape}, {y_loaded.dtype}")

            feature_names_path = f"{self.output_dir}/feature_names.txt"
            if Path(feature_names_path).exists():
                with open(feature_names_path, 'r', encoding='utf-8') as f:
                    feature_names_loaded = [line.strip() for line in f.readlines()]

                if X_loaded.shape[1] == len(feature_names_loaded):
                    logger.info("Feature names match X dimension")
                else:
                    logger.warning(f"Feature names count ({len(feature_names_loaded)}) "
                                  f"doesn't match X dimension ({X_loaded.shape[1]})")

        except ValueError as e:
            logger.error(f"Error loading files: {e}")
            logger.error("Files not in correct format. Check data processing.")
            raise