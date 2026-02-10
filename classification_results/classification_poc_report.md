# Proof of Concept: Developer Level Classification

## Executive Summary

This PoC demonstrates that developer levels can be automatically classified from hh.ru resume data.

## Results Summary

- **Model Type**: random_forest
- **Overall Accuracy**: 0.8727
- **Number of Samples**: 2011

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Junior | 0.7885 | 0.6543 | 0.7151 | 188 |
| Middle | 0.8941 | 0.9418 | 0.9173 | 1443 |
| Senior | 0.8149 | 0.7184 | 0.7636 | 380 |

## Key Findings

1. **Feasibility**: Approach shows promise for automatic level classification
2. **Data Quality**: Level extraction depends on position title quality
3. **IT Filter**: Filtering IT positions improves model relevance

## Limitations and Recommendations

1. **Class Imbalance**: Consider class weighting techniques
2. **Label Quality**: Manual validation of extracted levels needed
3. **Hyperparameter Tuning**: Optimize model parameters
