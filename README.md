# Foundation Models for Satellite Image Intelligence

**Team Name:** Ursa Space Systems 1A  
**Company:** Ursa Space Systems

## Project Overview

This project explores the use of open-source pre-trained computer vision models for classifying vessels and icebergs in synthetic aperture radar (SAR) satellite imagery. Since labeled data is expensive and time-consuming to obtain, our goal is to evaluate the applicability of pre-trained models and their embedding spaces to this domain, potentially reducing the cost and effort required for model development in key intelligence tasks.

## Objectives

- **Establish Baseline Models:** Build and evaluate baseline classification models for the SAR dataset.
- **Embedding Analysis:** Analyze embeddings from several open-source models, including those trained on satellite imagery and those trained on other domains.
- **Transfer Learning:** Assess whether pre-trained embeddings can improve downstream classification performance.
- **Domain Adaptation:** Investigate the generalizability of foundation models to new domains with limited labeled data.

## Dataset

We use an open-source SAR imagery dataset containing labeled examples of vessels and icebergs. The dataset is preprocessed and split into training and validation sets. See [clean_training_data.py](clean_training_data.py) and [split_training_validation_dataset.py](split_training_validation_dataset.py) for data preparation steps.

## Approach

1. **Data Cleaning & Preparation:**  
   - Handle missing values and transform image bands into matrices.
   - Split the dataset into training and validation sets with class balance.

2. **Baseline Modeling:**  
   - Train standard classifiers on the raw and preprocessed data.

3. **Embedding Extraction:**  
   - Use open-source pre-trained models to extract embeddings from SAR images.
   - Compare models trained on satellite imagery vs. general vision models.

4. **Downstream Classification:**  
   - Train classifiers using extracted embeddings.
   - Evaluate performance improvements over baseline models.

5. **Analysis:**  
   - Visualize and interpret embedding spaces.
   - Assess the effectiveness of transfer learning for SAR image classification.

## Getting Started

1. Clone the repository.
2. Install dependencies (see requirements in your notebooks/scripts).
3. Prepare the dataset using the provided scripts:
   - `python clean_training_data.py`
   - `python split_training_validation_dataset.py`
4. Explore and run the analysis in [Copy_of_Loading_the_training_data_as_images.ipynb](Copy_of_Loading_the_training_data_as_images.ipynb).

## File Structure

- `clean_training_data.py` — Cleans and preprocesses the raw SAR dataset.
- `split_training_validation_dataset.py` — Splits the cleaned data into training and validation sets.
- `Copy_of_Loading_the_training_data_as_images.ipynb` — Main notebook for data exploration, visualization, and modeling.
- `Datasets/` — Contains processed CSV files for training and validation.

## Acknowledgments

This project is part of Bridge to Studio, in collaboration with Ursa Space Systems.

---

*Details may be subject to change following the project kickoff meeting.*