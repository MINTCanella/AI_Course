"""
Data handlers for processing pipeline.
Uses Chain of Responsibility pattern.
"""

import re
import json
import logging
from typing import Any, Tuple, Optional, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats

from .chain import Handler

logger = logging.getLogger(__name__)


class DataLoader(Handler):
    """Load CSV file with support for large files."""

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
        """Clean data from duplicates and noise.

        Args:
            df: Input DataFrame

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
            series: Input Series

        Returns:
            Cleaned Series
        """
        cleaned = series.astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
        cleaned = cleaned.str.replace('\xa0', ' ', regex=False)
        cleaned = cleaned.str.replace('\ufeff', '', regex=False)

        return cleaned


class CurrencyConverter(Handler):
    """Currency conversion to rubles (optional step if different currencies exist)."""

    def __init__(self, exchange_rates: Optional[Dict[str, float]] = None):
        """Initialize currency converter.

        Args:
            exchange_rates: Dictionary with exchange rates to ruble
        """
        super().__init__()
        self.exchange_rates = exchange_rates or {
            'USD': 90.0,
            'EUR': 100.0,
            'KZT': 0.2,
            'BYN': 30.0
        }

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert salary to rubles if specified in other currency.

        Args:
            df: DataFrame with data

        Returns:
            DataFrame with salary in rubles
        """
        if 'ЗП' not in df.columns:
            return self._next(df)

        logger.info("Converting currency to rubles...")

        def convert_to_rub(salary_str: str) -> float:
            """Convert salary string to rubles."""
            if pd.isna(salary_str):
                return 0.0

            salary_str = str(salary_str).upper()

            currency = None
            if 'USD' in salary_str or '$' in salary_str:
                currency = 'USD'
            elif 'EUR' in salary_str or '€' in salary_str:
                currency = 'EUR'
            elif 'KZT' in salary_str or '₸' in salary_str:
                currency = 'KZT'
            elif 'BYN' in salary_str or 'РУБ.' in salary_str:
                currency = 'BYN'
            elif 'RUB' in salary_str or 'РУБ' in salary_str or 'Р' in salary_str:
                return CurrencyConverter._extract_number(salary_str)

            amount = CurrencyConverter._extract_number(salary_str)

            if currency and currency in self.exchange_rates:
                return amount * self.exchange_rates[currency]
            elif currency:
                logger.warning(f"Unknown currency: {currency}, treating as rubles")
                return amount
            else:
                return amount

        df['ЗП_рубли'] = df['ЗП'].apply(convert_to_rub)

        if 'ЗП_рубли' in df.columns:
            valid_salaries = df['ЗП_рубли'][df['ЗП_рубли'] > 0]
            if len(valid_salaries) > 0:
                logger.info(f"Converted {len(valid_salaries)} salaries to rubles")
                logger.info(f"Range: {valid_salaries.min():.0f} - {valid_salaries.max():.0f} RUB")
                logger.info(f"Median: {valid_salaries.median():.0f} RUB")

        return self._next(df)

    @staticmethod
    def _extract_number(text: str) -> float:
        """Extract number from string."""
        cleaned = re.sub(r'[^\d.,]', '', text)
        cleaned = cleaned.replace(',', '.')

        try:
            return float(cleaned) if cleaned else 0.0
        except (ValueError, TypeError):
            return 0.0


class FeatureExtractor(Handler):
    """Extract and create features from raw data."""

    def __init__(self, salary_threshold: Optional[float] = None):
        """Initialize feature extractor.

        Args:
            salary_threshold: Threshold for salary outliers filtering (in RUB)
        """
        super().__init__()
        self.salary_threshold = salary_threshold

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and create new features from data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        logger.info("Extracting features...")

        salary_column = 'ЗП_рубли' if 'ЗП_рубли' in df.columns else 'ЗП'

        if 'Пол, возраст' in df.columns:
            df['Возраст'] = df['Пол, возраст'].apply(self._extract_age)
            df['Пол'] = df['Пол, возраст'].apply(self._extract_gender)
            df['Возрастная_группа'] = df['Возраст'].apply(self._create_age_group)

        if salary_column in df.columns:
            if salary_column == 'ЗП':
                df['ЗП_число'] = df[salary_column].apply(self._normalize_salary)
            else:
                df['ЗП_число'] = df[salary_column]

            salaries = df['ЗП_число']
            logger.info(f"Salaries before processing: min={salaries.min():.0f}, "
                        f"max={salaries.max():.0f}, median={salaries.median():.0f}")

            if self.salary_threshold:
                df = self._handle_salary_outliers(df)

        if 'Город' in df.columns:
            df['Город_очищенный'] = df['Город'].apply(self._extract_city)

        if 'Опыт работы' in df.columns:
            df['Опыт_месяцы'] = df['Опыт работы'].apply(self._extract_experience)
            df['Опыт_группа'] = df['Опыт_месяцы'].apply(self._create_experience_group)

        if 'Авто' in df.columns:
            df['Есть_авто'] = df['Авто'].apply(
                lambda x: 1 if 'Имеется' in str(x) else 0
            )
            df = df.drop(columns=['Авто'])

        if 'Образование' in df.columns:
            df['Уровень_образования'] = df['Образование'].apply(self._extract_education_level)

        if 'Командировки' in df.columns:
            df['Частота_командировок'] = df['Командировки'].apply(self._extract_business_trips)

        return self._next(df)

    def _handle_salary_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle salary outliers.

        Args:
            df: DataFrame with salaries

        Returns:
            DataFrame with processed salaries
        """
        if 'ЗП_число' not in df.columns:
            return df

        salaries = df['ЗП_число']
        logger.info(f"Salaries before processing: min={salaries.min():.0f}, "
                    f"max={salaries.max():.0f}, median={salaries.median():.0f}")

        valid_mask = salaries > 0
        valid_salaries = salaries[valid_mask]

        if len(valid_salaries) == 0:
            logger.warning("No valid salaries!")
            return df

        logger.info(f"Valid salaries (>0): {valid_mask.sum()}/{len(salaries)}")
        logger.info(f"Median valid salary: {valid_salaries.median():.0f}")

        if self.salary_threshold:
            outliers_mask = valid_salaries > self.salary_threshold
            outlier_count = outliers_mask.sum()

            if outlier_count > 0:
                logger.info(f"Found {outlier_count} salary outliers "
                            f"(>{self.salary_threshold:.0f} RUB)")

                percentile_95 = valid_salaries[~outliers_mask].quantile(0.95)
                df.loc[valid_mask & (salaries > self.salary_threshold), 'ЗП_число'] = percentile_95
                logger.info(f"Outliers replaced with 95th percentile: {percentile_95:.0f} RUB")

        df.loc[valid_mask, 'ЗП_число'] = np.log1p(df.loc[valid_mask, 'ЗП_число'])

        if (~valid_mask).any():
            median_log_salary = df.loc[valid_mask, 'ЗП_число'].median()
            df.loc[~valid_mask, 'ЗП_число'] = median_log_salary
            logger.info(f"Invalid salaries ({len(salaries) - valid_mask.sum()}) "
                        f"replaced with median log salary: {median_log_salary:.2f}")

        logger.info(f"After processing: min={df['ЗП_число'].min():.2f}, "
                    f"max={df['ЗП_число'].max():.2f}, median={df['ЗП_число'].median():.2f}")

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
    def _create_age_group(age: int) -> str:
        """Create age group."""
        if age < 0:
            return 'Не указано'
        elif age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        else:
            return '55+'

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

    @staticmethod
    def _create_experience_group(months: int) -> str:
        """Create experience group."""
        if months <= 0:
            return 'Нет опыта'
        elif months <= 12:
            return 'До 1 года'
        elif months <= 36:
            return '1-3 года'
        elif months <= 60:
            return '3-5 лет'
        elif months <= 120:
            return '5-10 лет'
        else:
            return 'Более 10 лет'

    @staticmethod
    def _extract_education_level(education: str) -> str:
        """Extract education level."""
        if not isinstance(education, str):
            return 'Не указано'

        education = education.lower()

        if 'высшее' in education or 'вуз' in education:
            return 'Высшее'
        elif 'среднее' in education or 'техникум' in education or 'колледж' in education:
            return 'Среднее специальное'
        elif 'неполное' in education:
            return 'Неполное высшее'
        elif 'среднее общее' in education or 'школа' in education:
            return 'Среднее общее'
        else:
            return 'Другое'

    @staticmethod
    def _extract_business_trips(trips: str) -> str:
        """Extract business trip frequency."""
        if not isinstance(trips, str):
            return 'Не указано'

        trips = trips.lower()

        if 'готов' in trips or 'часто' in trips or 'регулярно' in trips:
            return 'Частые'
        elif 'редко' in trips or 'иногда' in trips:
            return 'Редкие'
        elif 'не готов' in trips or 'нежелательно' in trips:
            return 'Не готов'
        elif 'не имеет' in trips or 'без командировок' in trips:
            return 'Без командировок'
        else:
            return 'По договоренности'


class DataAggregator(Handler):
    """Aggregate identical X rows, average Y."""

    def __init__(self, target_column: str = 'ЗП_число'):
        """Initialize aggregator.

        Args:
            target_column: Target variable for averaging
        """
        super().__init__()
        self.target_column = target_column

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group identical feature rows, average target variable.

        Args:
            df: Input DataFrame

        Returns:
            Aggregated DataFrame
        """
        logger.info("Aggregating data...")

        if self.target_column not in df.columns:
            logger.warning(f"Target column '{self.target_column}' not found")
            return self._next(df)

        feature_columns = [col for col in df.columns if col != self.target_column]

        initial_count = len(df)
        unique_counts = df[feature_columns].drop_duplicates().shape[0]
        logger.info(f"Unique feature combinations: {unique_counts} out of {initial_count}")

        aggregated = df.groupby(feature_columns, as_index=False)[self.target_column].mean()
        count_df = df.groupby(feature_columns, as_index=False).size()
        aggregated = aggregated.merge(count_df, on=feature_columns)
        aggregated = aggregated.rename(columns={'size': 'group_size'})

        final_count = len(aggregated)
        logger.info(f"After aggregation: {final_count} rows (reduced by {initial_count - final_count})")
        logger.info(f"Average group size: {aggregated['group_size'].mean():.1f}")

        aggregated = aggregated.drop(columns=['group_size'])

        return self._next(aggregated)


class MulticollinearityHandler(Handler):
    """Handle feature multicollinearity."""

    def __init__(self, correlation_threshold: float = 0.95, exclude_columns: Optional[List[str]] = None):
        """Initialize multicollinearity handler.

        Args:
            correlation_threshold: Correlation threshold for feature removal
            exclude_columns: Columns to exclude from analysis
        """
        super().__init__()
        self.correlation_threshold = correlation_threshold
        self.exclude_columns = exclude_columns or []

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame without multicollinear features
        """
        logger.info("Handling multicollinearity...")

        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            logger.info("Too few numeric features for analysis")
            return self._next(df)

        columns_to_analyze = [col for col in numeric_df.columns if col not in self.exclude_columns]
        if len(columns_to_analyze) < 2:
            logger.info("Insufficient columns after exclusion")
            return self._next(df)

        numeric_df = numeric_df[columns_to_analyze]
        logger.info(f"Analyzing {len(numeric_df.columns)} numeric features")

        corr_matrix = numeric_df.corr().abs()
        to_drop = self._find_correlated_features(corr_matrix)

        if to_drop:
            logger.info(f"Removed {len(to_drop)} features due to high correlation:")
            for feature in to_drop:
                logger.info(f"  - {feature}")

                if feature in numeric_df.columns:
                    correlations = corr_matrix[feature]
                    high_corr = correlations[correlations > self.correlation_threshold]
                    high_corr = high_corr.drop(feature)
                    if not high_corr.empty:
                        correlated_with = high_corr.index[0]
                        corr_value = high_corr.iloc[0]
                        logger.info(f"    (correlation {corr_value:.3f} with '{correlated_with}')")

            df = df.drop(columns=to_drop)
            logger.info(f"Remaining features: {df.shape[1]}")
        else:
            logger.info("No high correlations found")

        return self._next(df)

    def _find_correlated_features(self, corr_matrix: pd.DataFrame) -> List[str]:
        """Find highly correlated features for removal.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            List of feature names to remove
        """
        corr_matrix = corr_matrix.copy()
        to_drop = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]

                    mean_corr_i = corr_matrix[col_i].mean()
                    mean_corr_j = corr_matrix[col_j].mean()

                    if mean_corr_i > mean_corr_j:
                        to_drop.append(col_i)
                        break
                    else:
                        to_drop.append(col_j)

        to_drop = list(set(to_drop))

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
        logger.info("Handling missing values...")

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


class SmartCategoricalEncoder(Handler):
    """Smart categorical feature encoding with improved strategies."""

    def __init__(self,
                 top_cities: int = 50,
                 top_positions: int = 100,
                 max_ohe_categories: int = 20):
        """Initialize smart encoder.

        Args:
            top_cities: Number of top cities for one-hot encoding
            top_positions: Number of top positions for one-hot encoding
            max_ohe_categories: Maximum number of categories for OHE
        """
        super().__init__()
        self.top_cities = top_cities
        self.top_positions = top_positions
        self.max_ohe_categories = max_ohe_categories

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features with smart strategies.

        Args:
            df: DataFrame with categorical features

        Returns:
            DataFrame with encoded features
        """
        logger.info("Smart encoding categorical features...")

        target_column = 'ЗП_число' if 'ЗП_число' in df.columns else None
        if target_column:
            y = df[target_column]
            df_temp = df.drop(columns=[target_column])
        else:
            df_temp = df.copy()

        categorical_columns = df_temp.select_dtypes(include=['object']).columns
        logger.info(f"Found {len(categorical_columns)} categorical features")

        columns_to_drop = []
        for col in categorical_columns:
            unique_count = df_temp[col].nunique()
            if unique_count > 5000:
                columns_to_drop.append(col)
                logger.info(f"  Removing {col}: {unique_count} unique values")

        df_temp = df_temp.drop(columns=columns_to_drop)
        categorical_columns = df_temp.select_dtypes(include=['object']).columns

        df_temp = self._apply_binary_encoding(df_temp)
        df_temp = self._apply_onehot_encoding(df_temp)

        if target_column and len(categorical_columns) > 0:
            df_temp = self._apply_target_encoding(df_temp, y)

        if target_column:
            df_temp[target_column] = y

        logger.info(f"After encoding: {df_temp.shape[1]} features")

        return self._next(df_temp)

    def _apply_binary_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary encoding for columns with 2 unique values."""
        binary_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() == 2:
                binary_cols.append(col)

        if binary_cols:
            logger.info(f"Binary encoding for {len(binary_cols)} features")

        for col in binary_cols:
            unique_values = df[col].unique()
            if len(unique_values) == 2:
                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                df[col] = df[col].map(mapping).astype(np.float64)

        return df

    def _apply_onehot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-Hot Encoding for columns with limited categories."""
        result_df = df.copy()

        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()

            if 2 < unique_count <= self.max_ohe_categories:
                logger.info(f"  One-Hot for {col}: {unique_count} categories")

                dummies = pd.get_dummies(df[col], prefix=col,
                                         prefix_sep='_',
                                         dummy_na=False)

                if len(dummies.columns) > 1:
                    dummies = dummies.iloc[:, 1:]

                result_df = result_df.drop(columns=[col])
                result_df = pd.concat([result_df, dummies], axis=1)

        return result_df

    def _apply_target_encoding(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Target Encoding for categorical features."""
        result_df = df.copy()

        for col in df.select_dtypes(include=['object']).columns:
            logger.info(f"  Target Encoding for {col}")

            encoding_map = y.groupby(df[col]).mean().to_dict()
            result_df[col] = df[col].map(encoding_map).astype(np.float64)

            global_mean = y.mean()
            result_df[col] = result_df[col].fillna(global_mean)

        return result_df


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
            Passed data unchanged
        """
        if isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)
            logger.info("Saving feature metadata...")
            logger.info(f"  Found {len(self.feature_names)} features")

            metadata = {
                'feature_names': self.feature_names,
                'feature_count': len(self.feature_names),
                'data_shape': data.shape,
                'data_types': {col: str(data[col].dtype) for col in data.columns}
            }

            metadata_path = f'{self.output_dir}/feature_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"  Metadata saved to {metadata_path}")

            txt_path = f'{self.output_dir}/feature_names.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                for i, name in enumerate(self.feature_names):
                    f.write(f"{i}: {name}\n")

            logger.info(f"  Feature names saved to {txt_path}")

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
        logger.info("Handling outliers...")

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            initial_outliers = self._count_outliers(df[col], self.method, self.threshold)

            if initial_outliers > 0:
                logger.info(f"  {col}: {initial_outliers} outliers "
                            f"({initial_outliers / len(df) * 100:.1f}%)")

                if self.method == 'clip':
                    df[col] = self._clip_outliers(df[col], self.method, self.threshold)
                    logger.info(f"    Outliers clipped")
                elif self.method == 'remove':
                    pass
                elif self.method in ['iqr', 'zscore']:
                    df[col] = self._winsorize_outliers(df[col], self.method, self.threshold)
                    logger.info(f"    Outliers replaced with boundary values")

        if self.method == 'remove':
            mask = pd.Series(True, index=df.index)
            for col in numeric_columns:
                outlier_mask = self._get_outlier_mask(df[col], self.method, self.threshold)
                mask &= ~outlier_mask

            rows_removed = len(df) - mask.sum()
            if rows_removed > 0:
                df = df[mask]
                logger.info(f"  Removed {rows_removed} rows with outliers")

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
                logger.info("  Using RobustScaler (robust to outliers)")
                robust_scaler = RobustScaler()
                df[numeric_columns] = robust_scaler.fit_transform(df[numeric_columns])

            if self.use_standard_scaler:
                logger.info("  Using StandardScaler (mean=0, std=1)")
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
            df: Complete DataFrame

        Returns:
            Tuple (X, y, feature_names) where X - features, y - target variable,
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

        logger.info(f"  Found {len(self.feature_names)} features:")
        for i, name in enumerate(self.feature_names[:20]):
            logger.info(f"    [{i}] {name}")
        if len(self.feature_names) > 20:
            logger.info(f"    ... and {len(self.feature_names) - 20} more features")

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
            numpy array with target variable
        """
        logger.info("  Preparing target variable...")

        y = pd.to_numeric(target_series, errors='coerce')

        zero_mask = y <= 0
        if zero_mask.any():
            zero_count = zero_mask.sum()
            logger.warning(f"  Found {zero_count} values <= 0 in salary "
                           f"({zero_count / len(y) * 100:.1f}%)")

            min_positive = y[y > 0].min() if (y > 0).any() else 1.0
            y[zero_mask] = min_positive
            logger.info(f"  Replaced with minimum positive salary: {min_positive:.2f}")

        logger.info(f"  Before log: min={y.min():.0f}, max={y.max():.0f}, "
                    f"median={y.median():.0f}, skew={y.skew():.2f}")

        logger.info(f"  Applying logarithmic transformation (log1p)")
        y_log = np.log1p(y)

        logger.info(f"  After log: min={y_log.min():.4f}, max={y_log.max():.4f}, "
                    f"mean={y_log.mean():.4f}, skew={pd.Series(y_log).skew():.2f}")

        return y_log.values.astype(np.float64)

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
            y: Target variable vector
        """
        logger.info(f"  X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"  X dtype: {X.dtype}, y dtype: {y.dtype}")
        logger.info(f"  X min/max/mean: [{X.min():.4f}, {X.max():.4f}, {X.mean():.4f}]")
        logger.info(f"  y min/max/mean: [{y.min():.4f}, {y.max():.4f}, {y.mean():.4f}]")

        y_rub_sample = np.expm1(y[:100])
        logger.info(f"  y sample in RUB: min={y_rub_sample.min():.0f}, "
                    f"max={y_rub_sample.max():.0f}")


class AdvancedFeatureEngineer(Handler):
    """Advanced feature creation from hh.ru data."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for salary prediction.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        logger.info("Creating advanced features...")

        original_columns = list(df.columns)

        if 'Ищет работу на должность:' in df.columns:
            df['Длина_названия_должности'] = df['Ищет работу на должность:'].apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            )

            keywords = ['менеджер', 'разработчик', 'аналитик', 'директор',
                        'специалист', 'инженер', 'руководитель']

            for keyword in keywords:
                df[f'Должность_{keyword}'] = df['Ищет работу на должность:'].apply(
                    lambda x: 1 if keyword in str(x).lower() else 0
                )

        if 'Опыт_месяцы' in df.columns:
            df['Опыт_лет'] = df['Опыт_месяцы'] / 12
            df['Опыт_группа_детальная'] = pd.cut(
                df['Опыт_лет'],
                bins=[0, 1, 3, 5, 10, 20, 100],
                labels=['0-1 год', '1-3 года', '3-5 лет', '5-10 лет', '10-20 лет', '20+ лет']
            )

        if 'Образование и ВУЗ' in df.columns:
            df['Высшее_образование'] = df['Образование и ВУЗ'].apply(
                lambda x: 1 if 'высшее' in str(x).lower() else 0
            )

            tech_keywords = ['техническ', 'инженер', 'программист', 'it', 'компьютер']
            df['Техническое_образование'] = df['Образование и ВУЗ'].apply(
                lambda x: 1 if any(keyword in str(x).lower() for keyword in tech_keywords) else 0
            )

        if 'Город_очищенный' in df.columns:
            capital_cities = ['москва', 'санкт-петербург', 'минск', 'киев', 'алматы']
            df['Столица'] = df['Город_очищенный'].apply(
                lambda x: 1 if any(city in str(x).lower() for city in capital_cities) else 0
            )

        if 'Возраст' in df.columns and 'Опыт_лет' in df.columns:
            df['Начало_карьеры'] = df['Возраст'] - df['Опыт_лет']

        new_columns_count = len(df.columns) - len(original_columns)
        logger.info(f"  Created {new_columns_count} new features")

        if new_columns_count > 0:
            new_columns = [col for col in df.columns if col not in original_columns]
            logger.info(f"  New features: {new_columns[:10]}")
            if len(new_columns) > 10:
                logger.info(f"    ... and {len(new_columns) - 10} more features")

        if 'ЗП_число' in df.columns and 'Опыт_группа' in df.columns:
            df = self._calculate_salary_percentiles(df)

        return self._next(df)


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
                logger.info(f"  Truncated feature names to {len(feature_names)}")
            elif len(feature_names) < X.shape[1]:
                missing_count = X.shape[1] - len(feature_names)
                for i in range(missing_count):
                    feature_names.append(f"feature_{len(feature_names) + i}")
                logger.info(f"  Added {missing_count} generic names")

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
            Tuple (X, y) with corrected types
        """
        if X.dtype != np.float64:
            logger.info(f"  Converting X to float64 (was {X.dtype})")
            X = X.astype(np.float64)

        if y.dtype != np.float64:
            logger.info(f"  Converting y to float64 (was {y.dtype})")
            y = y.astype(np.float64)

        return X, y

    @staticmethod
    def _validate_data_quality(X: np.ndarray, y: np.ndarray) -> None:
        """Validate data quality before saving.

        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("  Validating data quality...")

        x_nan = np.isnan(X).sum()
        y_nan = np.isnan(y).sum()
        x_inf = np.isinf(X).sum()
        y_inf = np.isinf(y).sum()

        if x_nan > 0 or y_nan > 0 or x_inf > 0 or y_inf > 0:
            logger.warning(f"  Found problematic values")
            logger.info(f"     NaN: X={x_nan}, y={y_nan}")
            logger.info(f"     Inf: X={x_inf}, y={y_inf}")

        logger.info(f"  X stats: mean={X.mean():.4f}, std={X.std():.4f}")
        logger.info(f"  y stats: mean={y.mean():.4f}, std={y.std():.4f}, "
                    f"skew={stats.skew(y):.4f}")

        if abs(y.mean()) > 1.0 or y.std() > 10.0:
            logger.warning(f"  y has unusual statistics")

        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X, rowvar=False)
            high_corr_count = 0
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > 0.99:
                        high_corr_count += 1

            if high_corr_count > 0:
                logger.warning(f"  Found {high_corr_count} feature pairs with correlation > 0.99")
                logger.info(f"     Possible dummy variable trap in One-Hot Encoding")

    def _save_files(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, str]:
        """Save arrays to files.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of saved file paths
        """
        x_path = f"{self.output_dir}/x_data.npy"
        y_path = f"{self.output_dir}/y_data.npy"

        np.save(x_path, X)
        np.save(y_path, y)

        logger.info(f"  Saved: {x_path} ({X.shape}, {X.dtype})")
        logger.info(f"  Saved: {y_path} ({y.shape}, {y.dtype})")

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

        logger.info(f"  Feature names saved to {feature_names_path}")

        metadata = {
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'note': 'Saved with x_data.npy and y_data.npy'
        }

        json_path = f"{self.output_dir}/feature_names.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"  Feature metadata saved to {json_path}")

    def _verify_saved_files(self, x_path: str, y_path: str) -> None:
        """Verify correctness of saved files.

        Args:
            x_path: Path to X file
            y_path: Path to y file
        """
        logger.info("  Verifying saved files...")

        try:
            X_loaded = np.load(x_path)
            y_loaded = np.load(y_path)

            logger.info(f"  Files loaded correctly")
            logger.info(f"    X: {X_loaded.shape}, {X_loaded.dtype}")
            logger.info(f"    y: {y_loaded.shape}, {y_loaded.dtype}")

            feature_names_path = f"{self.output_dir}/feature_names.txt"
            if Path(feature_names_path).exists():
                with open(feature_names_path, 'r', encoding='utf-8') as f:
                    feature_names_loaded = [line.strip() for line in f.readlines()]

                if X_loaded.shape[1] == len(feature_names_loaded):
                    logger.info(f"  Feature names match X size")
                else:
                    logger.warning(f"  Feature name count ({len(feature_names_loaded)}) "
                                   f"doesn't match X size ({X_loaded.shape[1]})")

        except ValueError as e:
            logger.error(f"  Error loading: {e}")
            logger.error(f"  Files not in correct format. Check data processing.")
            raise
