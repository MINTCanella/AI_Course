"""
Utilities for creating data processing pipeline.
"""

from .handlers import (
    DataLoader, DataCleaner, FeatureExtractor, MulticollinearityHandler,
    MissingValueHandler, CategoricalEncoder, FeatureMetadataSaver,
    OutlierHandler, Normalizer, TargetSplitter, NumpySaver
)


def create_pipeline(output_dir: str = '.') -> DataLoader:
    """
    Create chain of data processing handlers.

    Args:
        output_dir: Directory for saving results

    Returns:
        Initial handler of the pipeline
    """
    loader = DataLoader()
    cleaner = DataCleaner()

    extractor = FeatureExtractor(salary_threshold=500000.0)
    multicollinearity_handler = MulticollinearityHandler(correlation_threshold=0.98)
    missing_handler = MissingValueHandler()

    encoder = CategoricalEncoder(
        max_categories_for_ohe=10,
        special_columns={'Пол': 'label'},
        drop_first=True
    )

    metadata_saver = FeatureMetadataSaver(output_dir=output_dir)
    outlier_handler = OutlierHandler(method='iqr', threshold=1.5)
    normalizer = Normalizer(use_robust_scaler=True, use_standard_scaler=True)
    splitter = TargetSplitter(target_column='ЗП_число')
    saver = NumpySaver(output_dir=output_dir)

    loader.set_next(cleaner) \
        .set_next(extractor) \
        .set_next(multicollinearity_handler) \
        .set_next(missing_handler) \
        .set_next(encoder) \
        .set_next(metadata_saver) \
        .set_next(outlier_handler) \
        .set_next(normalizer) \
        .set_next(splitter) \
        .set_next(saver)

    return loader