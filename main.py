"""
Main application entry point for data processing pipeline.
Improved version with better feature engineering.
"""

import logging
import sys
import os

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('improved_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

from pipeline.pipeline_utils import create_improved_pipeline, create_simple_pipeline


def main() -> None:
    """
    Main function to run improved data processing pipeline.

    This function handles command line arguments and orchestrates
    the data processing pipeline.
    """
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        logger.error("Usage: python main.py <path_to_csv_file> [--simple]")
        logger.error("Examples:")
        logger.error("  python main.py files/hh.csv           # Improved pipeline")
        logger.error("  python main.py files/hh.csv --simple  # Simple pipeline")
        sys.exit(1)

    filepath = sys.argv[1]
    use_simple = len(sys.argv) == 3 and sys.argv[2] == '--simple'

    if not os.path.exists(filepath):
        logger.error("File not found: %s", filepath)
        sys.exit(1)

    logger.info("Starting IMPROVED data processing pipeline...")
    logger.info("=" * 60)
    logger.info("IMPROVEMENTS:")
    logger.info("1. Currency conversion to RUB")
    logger.info("2. Smart categorical encoding")
    logger.info("3. Top-100 cities, Top-500 positions for One-Hot")
    logger.info("4. Label encoding for ordinal features")
    logger.info("5. Data aggregation for duplicate feature combinations")
    logger.info("=" * 60)

    if use_simple:
        logger.info("Using SIMPLE pipeline (faster, less features)")
    else:
        logger.info("Using IMPROVED pipeline (better for linear models)")

    try:
        input_dir = os.path.dirname(filepath)
        output_dir = input_dir if input_dir else '.'

        logger.info("Saving results to directory: %s", output_dir)

        if use_simple:
            pipeline = create_simple_pipeline(output_dir=output_dir)
        else:
            pipeline = create_improved_pipeline(output_dir=output_dir)

        pipeline.handle(filepath)

        logger.info("=" * 60)
        logger.info("IMPROVED processing completed successfully!")
        logger.info("Created files in directory '%s':", output_dir)
        logger.info("  - x_data.npy (features matrix)")
        logger.info("  - y_data.npy (target vector, log1p scale)")
        logger.info("  - feature_names.txt (feature names)")
        logger.info("  - feature_names.json (feature metadata)")
        logger.info("  - feature_metadata.json (full metadata)")
        logger.info("  - improved_pipeline.log (this log file)")

        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Train model: python linear_model.py --x-data files/x_data.npy --y-data files/y_data.npy")
        logger.info("2. Use RidgeCV for automatic alpha selection")
        logger.info("3. The target is already log-transformed for better linearity")

    except Exception as e:
        logger.error("Error during processing: %s", e)
        logger.exception("Detailed error traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
