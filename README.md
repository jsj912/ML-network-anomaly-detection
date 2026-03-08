# Malware Traffic Analysis using Machine Learning
A machine learning system for detecting malicious network traffic using anomaly detection and supervised reranking. The project builds a two-stage detection pipeline and visualizes security alerts through a SOC-style dashboard.

## Overview
This project focuses on detecting malicious network behavior from network traffic data. Instead of relying purely on signatures, the system uses machine learning to identify anomalous behavior patterns in network flows.
The pipeline combines unsupervised anomaly detection with supervised classification and provides explainable results using SHAP.

## System Architecture
Network Traffic (PCAP)  
↓  
Flow Extraction  
↓  
Feature Engineering  
↓  
Isolation Forest (Anomaly Detection)  
↓  
LightGBM Re-ranking  
↓  
SHAP Explainability  
↓  
SOC Dashboard Visualization

## Key Features
• Two-stage machine learning pipeline for anomaly detection  
• Isolation Forest for unsupervised behavioral analysis  
• LightGBM model for improved classification accuracy  
• SHAP explanations for model interpretability  
• SOC-style dashboard for monitoring anomalies  
• Visualization of feature importance and anomaly distributions  

## Technologies Used
**Programming**
- Python

**Machine Learning**
- Scikit-Learn
- LightGBM
- SHAP

**Data Processing**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Dash
  
## Dataset
Due to size and privacy considerations, raw PCAP datasets are **not included** in this repository.
The project was tested using publicly available datasets:
• CIC IDS 2017  
https://www.unb.ca/cic/datasets/ids-2017.html
• CTU-13 Botnet Dataset  
https://www.stratosphereips.org/datasets-ctu13

