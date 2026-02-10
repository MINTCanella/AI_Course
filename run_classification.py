"""
Quick runner for classification PoC.
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run classification PoC with default parameters."""
    if len(sys.argv) < 2:
        logger.error("Usage: python run_classification.py <path_to_hh_csv>")
        logger.error("Example: python run_classification.py files/hh.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Create output directory for results
    output_dir = "classification_results"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Run preprocessing pipeline if needed
    if not os.path.exists('x_data.npy'):
        logger.info("=" * 60)
        logger.info("Step 1: Running preprocessing pipeline...")
        logger.info("=" * 60)
        result = subprocess.run(['python', 'main.py', csv_path],
                                capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Preprocessing failed!")
            logger.error(result.stderr)
            sys.exit(1)

    # Step 2: Run classification PoC
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Running classification PoC...")
    logger.info("=" * 60)

    # Try different models
    models = ['logistic', 'random_forest']

    for model in models:
        logger.info(f"\n>>> Training {model.upper()} model <<<")
        result = subprocess.run(
            ['python', 'classification_pipeline.py', csv_path, '--model', model],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            logger.info(f"Successfully trained {model} model")
        else:
            logger.error(f"Error with {model}:")
            logger.error(result.stderr)

    logger.info("=" * 60)
    logger.info("PoC completed! Check the following files:")
    logger.info(f"  - {output_dir}/classification_poc.log (detailed log)")
    logger.info(f"  - {output_dir}/classification_poc_report.md (final report)")
    logger.info(f"  - {output_dir}/developer_classifier_model.pkl (trained model)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()