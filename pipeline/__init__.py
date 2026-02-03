"""
Data processing pipeline module.
"""

from .chain import Handler
from .handlers import (
    DataLoader, DataCleaner, FeatureExtractor,
    MissingValueHandler, Normalizer, TargetSplitter, NumpySaver,
    CurrencyConverter, DataAggregator, MulticollinearityHandler,
    SmartCategoricalEncoder, FeatureMetadataSaver, OutlierHandler
)
from .pipeline_utils import create_improved_pipeline, create_simple_pipeline

__all__ = [
    'Handler',
    'DataLoader',
    'DataCleaner',
    'FeatureExtractor',
    'MissingValueHandler',
    'Normalizer',
    'TargetSplitter',
    'NumpySaver',
    'CurrencyConverter',
    'DataAggregator',
    'MulticollinearityHandler',
    'SmartCategoricalEncoder',
    'FeatureMetadataSaver',
    'OutlierHandler',
    'create_improved_pipeline',
    'create_simple_pipeline'
]
