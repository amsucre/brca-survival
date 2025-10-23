# Multimodal Fusion for BRCA Survival Prediction
This repository contains the complete, reproducible code and scripts associated with the manuscript, "Multimodal Fusion Strategies for Survival Prediction in Breast Cancer: A Comparative Deep Learning Study"

## Overview

This project systematically investigates the most robust and generalizable deep learning architecture for integrating highly heterogeneous multi-modal data (omics, clinical, and imaging) for breast cancer overall survival prediction using the TCGA-BRCA cohort.

Our optimized Late Fusion framework achieves superior discriminatory power and stability compared to Early Fusion and established State-of-the-Art methods.

Please cite our paper: [Multimodal Fusion Strategies for Survival Prediction in Breast Cancer: A Comparative Deep Learning Study](https://www.sciencedirect.com/science/article/pii/S2001037025004404), if you use this code in your research.

## Setup and Requirements

The study was performed using Python 3 and R, in a Linux system. Each submodule contains a requirements file needed to setup the envirnonment. 

## Analytical modules

### 1. Data processing

Tabular (omics and clinical) data was downloaded from the [Xena Browser](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)) and imaging data from the [GDC data portal](https://portal.gdc.cancer.gov/)

Downloaded tabular data (needed to train unimodal, early-fusion and benchmark models) and outcomes from all unimodal models (needed to train late-fusion models) were preprocessed using the scripts in the [**data_processing**](https://github.com/amsucre/brca-survival/tree/main/data_processing) module. 

Imaging data was processed using the scripts in the [**img_processing**](https://github.com/amsucre/brca-survival/tree/main/img_processing) module, based on [CLAM](http://clam.mahmoodlab.org).

### 2. Benchmark model training

In this study, we benchmarked our models against three well-stablished state of the art models: [PORPOISE](https://github.com/mahmoodlab/PORPOISE), [MCAT](https://github.com/mahmoodlab/MCAT) and [MGCT](https://github.com/lmxmercy/MGCT), using the same TCGA dataset as we used for our models. 

In the [**model_training_soa**](https://github.com/amsucre/brca-survival/tree/main/model_training_soa) module you can find all the scripts needed to replicate these three models within the context of our study.

### 3. Image model training

We built models using histopathology images. These were based on the AMIL approach to predict survival using WSIs. Two different encoders were tested: ResNet-50 general-purpose model, and UNI pathology foundation model. The general-purpose model yielded the best results.

In the [**model_training_img**](https://github.com/amsucre/brca-survival/tree/main/model_training_img) module you can find all the scripts needed to optimize, train and test the image-based models developed in  our study.

### 4. Tabular and multimodal model training

We built independent models using clinical variables, SNVs, RNA-seq, CNV, and miRNA data. These models were based on fully connected feedforward neural networks. Aditionally, we built early and late integration models based on this same architecture. 

For each model we first performed a hyperparameter optimization process using the Optuna framework to define the neural network structure. Then the optimal models were trained, cross-validated and tested using Pytorch.    

In the [**model_training**](https://github.com/amsucre/brca-survival/tree/main/model_training) module you can find all the scripts needed to optimize, train and test the tabular (unimodal) and multimodal models developed in  our study.

The "_sh.py" files can be used to (1) perform optimization, (2) train and eval models and (3) get predictions from trained models from SH files/terminal, providing the required inputs.  

### 5. Model explainability analysis

To further assess the results obtained in our study, a secondary analysis module was defined that included: 

* IBS and C-Index metrics calculation
* SHAP analysis
* Survival analysis
* Enrichment analysis
* Plotting: Violin and  density plots

All the necessary scripts for this analysis can be found in the [**model_explainability**](https://github.com/amsucre/brca-survival/tree/main/model_explainability) module. 
