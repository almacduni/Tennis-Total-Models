# Tennis Set Total Games Prediction Using Probability-Based Feature Engineering

## Overview
Novel approach to predicting total games in tennis sets achieving 85.6% accuracy by transforming match statistics into probability bins and using a Mixture of Experts neural network.

## Key Innovation
Transform raw features into probability distributions across the dataset instead of using raw statistics. Each feature is converted to its historical probability before feeding to the model - this significantly outperformed traditional ML algorithms.

## Results

### Overall Performance
* Accuracy: 85.6%
* F1 Score: 0.727
* AUROC: 0.741
* ECE: 0.144

### Class Performance
* Common Class F1: 0.884 (predictions with >30% base rate)
* Minority Class F1: 0.256 (predictions with <30% base rate)

### Best Performing Thresholds
* Over 8.5 games: F1=0.940, AUC=0.902
* Under 8.5 games: F1=0.869, AUC=0.900
* 9.5 games (both): F1=0.85+, AUC=0.86

## Method:
- Parse point-by-point JSON data from tennis matches
- Engineer features from first 6 games of each set
- Calculate relative differences from league averages
- Transform features into probability bins (key step)
- Train Mixture of Experts model with custom loss function

## Repository Contents
- notebook.ipynb - Complete implementation
- sample_data.csv - Sample tennis data for testing
- This README

## Full Dataset
For complete 1GB dataset (30,000+ matches with point-by-point data), contact: almacduni@gmail.com

## Usage
python# Run notebook to:
1. Load tennis data
2. Create probability features  
3. Train MoE model
4. Get predictions with 85.6% accuracy


## License
Free for research. Commercial use requires permission.
Contact: almacduni@gmail.com
