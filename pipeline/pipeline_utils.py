"""
Utilities for creating data processing pipeline.
Improved version - optimized for linear regression with smart encoding.
"""

from .handlers import (
    DataLoader, DataCleaner, CurrencyConverter, FeatureExtractor, DataAggregator,
    MulticollinearityHandler, MissingValueHandler, SmartCategoricalEncoder,
    FeatureMetadataSaver, OutlierHandler, Normalizer, TargetSplitter, NumpySaver,
    AdvancedFeatureEngineer
)


def create_improved_pipeline(output_dir: str = '.') -> DataLoader:
    """Create chain of data processing handlers optimized for linear regression.

    Args:
        output_dir: Directory for saving results

    Returns:
        Initial handler of the pipeline
    """
    loader = DataLoader()
    cleaner = DataCleaner()
    converter = CurrencyConverter()
    extractor = FeatureExtractor(salary_threshold=1000000.0)
    feature_engineer = AdvancedFeatureEngineer()
    aggregator = DataAggregator(target_column='ЗП_число')
    missing_handler = MissingValueHandler()

    encoder = SmartCategoricalEncoder(
        top_cities=20,
        top_positions=50,
        max_ohe_categories=10
    )

    multicollinearity_handler = MulticollinearityHandler(
        correlation_threshold=0.9,
        exclude_columns=['ЗП_число']
    )

    metadata_saver = FeatureMetadataSaver(output_dir=output_dir)
    outlier_handler = OutlierHandler(method='clip', threshold=3.0)
    normalizer = Normalizer(use_robust_scaler=True, use_standard_scaler=True)
    splitter = TargetSplitter(target_column='ЗП_число')
    saver = NumpySaver(output_dir=output_dir)

    loader.set_next(cleaner) \
        .set_next(converter) \
        .set_next(extractor) \
        .set_next(feature_engineer) \
        .set_next(aggregator) \
        .set_next(missing_handler) \
        .set_next(encoder) \
        .set_next(multicollinearity_handler) \
        .set_next(metadata_saver) \
        .set_next(outlier_handler) \
        .set_next(normalizer) \
        .set_next(splitter) \
        .set_next(saver)

    return loader


def create_simple_pipeline(output_dir: str = '.') -> DataLoader:
    """Create simplified pipeline for quick testing.

    Args:
        output_dir: Directory for saving results

    Returns:
        Initial handler of the pipeline
    """
    loader = DataLoader()
    cleaner = DataCleaner()
    converter = CurrencyConverter()
    extractor = FeatureExtractor(salary_threshold=500000.0)
    aggregator = DataAggregator(target_column='ЗП_число')
    missing_handler = MissingValueHandler()
    splitter = TargetSplitter(target_column='ЗП_число')
    saver = NumpySaver(output_dir=output_dir)

    loader.set_next(cleaner) \
        .set_next(converter) \
        .set_next(extractor) \
        .set_next(aggregator) \
        .set_next(missing_handler) \
        .set_next(splitter) \
        .set_next(saver)

    return loader
