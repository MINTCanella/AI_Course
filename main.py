"""
Main application entry point for data processing pipeline.
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
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

from pipeline.pipeline_utils import create_pipeline


def main() -> None:
    """
    Main function to run data processing pipeline.

    Parses command line arguments and executes the pipeline.
    """
    if len(sys.argv) != 2:
        logger.error("Usage: python main.py <path_to_csv_file>")
        logger.error("Example: python main.py files/hh.csv")
        sys.exit(1)

    filepath = sys.argv[1]

    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        sys.exit(1)

    logger.info("Starting data processing pipeline...")
    logger.info("=" * 50)

    try:
        input_dir = os.path.dirname(filepath)
        output_dir = input_dir if input_dir else '.'

        logger.info(f"Saving results to directory: {output_dir}")

        pipeline = create_pipeline(output_dir=output_dir)
        pipeline.handle(filepath)

        logger.info("=" * 50)
        logger.info("Processing completed successfully!")
        logger.info(f"Created files in directory '{output_dir}':")
        logger.info("  - x_data.npy")
        logger.info("  - y_data.npy")
        logger.info("  - feature_names.txt")
        logger.info("  - feature_names.json")
        logger.info("  - pipeline.log (this log file)")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.exception("Detailed error traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()