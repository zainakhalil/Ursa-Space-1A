# Foundation Models for Satellite Image Intelligence

**Team Name:** Ursa Space Systems 1A  
**Company:** Ursa Space Systems

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Tasnia Chowdhury   | @ChowdhuryTasnia | Understanding dataset and SAR images, Training preliminary model              |
| Dev Jyoti Ghosh Arnab   | @Dev-Arnab     | Data preprocessing, exploratory data analysis (EDA), PCA analysis  |
| Katy Nguyen     | @katymn  | Data preprocessing, embedding extraction, hyperparameter tuning, performance analysis                 |
| Caliese Beckford      | @Caliese       | Pre-trained model selection, Training GBDTS model on embeddings and optimization, performance analysis  |
| Chiamanda Ononiwu      | @Chiamanda07       | Data preprocessing, embedding extraction, Training neural network model on embeddings and optimization  |
| Zaina Khalil       | @zainakhalil    | Training base preliminray model, hyperparameter tuning, performance analysis           |

---

## 🎯 **Project Highlights**

-  Developed a machine learning pipeline using **convolutional neural networks and pre-trained vision transformers** to address **labeled data scarcity in SAR-based vessel vs. iceberg classification**.
- Achieved **83% baseline accuracy with a CNN and improved performance using transformer-based embeddings**, demonstrating **more efficient and scalable satellite image analysis**.
- Generated **high-quality feature embeddings and comparative model evaluations** to inform **model selection and preprocessing strategies** for **remote sensing analysts and AI stakeholders**.
- Implemented **advanced SAR preprocessing (Lee speckle filtering, image masking, zoom-based cropping) and PCA-based dimensionality reduction** to address **noise-heavy imagery and computational constraints common in satellite remote sensing**.
---


## Project Overview

This project explores the use of open-source pre-trained computer vision models for classifying vessels and icebergs in synthetic aperture radar (SAR) satellite imagery. Since labeled data is expensive and time-consuming to obtain, our goal is to evaluate the applicability of pre-trained models and their embedding spaces to this domain, potentially reducing the cost and effort required for model development in key intelligence tasks.

## Objectives

- **Establish Baseline Models:** Build and evaluate baseline classification models for the SAR dataset.
- **Embedding Analysis:** Analyze embeddings from several open-source models, including those trained on satellite imagery and those trained on other domains.
- **Transfer Learning:** Assess whether pre-trained embeddings can improve downstream classification performance.
- **Domain Adaptation:** Investigate the generalizability of foundation models to new domains with limited labeled data.

## Dataset

We use an open-source SAR imagery dataset containing labeled examples of vessels and icebergs. The dataset is preprocessed and split into training and validation sets. See [clean_training_data.py](clean_training_data.py), [split_training_validation_dataset.py](split_training_validation_dataset.py), and [EDA.ipynb](EDA.ipynb) for data preparation steps.

## Approach

1. **Data Cleaning & Preparation:**  
   - Handle missing values and transform image bands into matrices.
   - Split the dataset into training and validation sets with class balance.
   - Reduced SAR noise and background interference using:
     - Lee (speckle) filtering for noise suppression
     - 70% image masking to remove background clutter
     - Zoom-based cropping to minimize edge noise


2. **Baseline Modeling:**  
   - Train standard classifiers on the raw and preprocessed data.

3. **Embedding Extraction:**  
   - Use open-source pre-trained models like ViT and Swin to extract embeddings from SAR images.
   - Applied Principal Component Analysis (PCA) to compress embeddings from **151,296 → 50 features** (**99.97% dimensionality reduction**) while preserving meaningful variance for downstream models..
   - Compare models trained on satellite imagery vs. general vision models.

4. **Downstream Classification:**  
   - Train classifiers using extracted embeddings.
   - Evaluate performance improvements over baseline models.

5. **Analysis:**  
   - Visualize and interpret embedding spaces.
   - Assess the effectiveness of transfer learning for SAR image classification.
  
6. **Modeling & Evaluation**
   - Trained and evaluated multiple models on extracted embeddings, including:
     - Dense Neural Networks
     - Gradient Boosting Decision Trees (GBDT)

   - Achieved best performance with:
     - Simple Convolutional Neural Network (highest overall accuracy)
     - Dense Neural Network trained on **masked ViT embeddings** (highest performance on embeddings)

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
