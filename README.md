# Transforming Landscapes: Impact of LULC on Air Quality

## Overview

This project explores the relationship between **Land Use Land Cover (LULC)** patterns and **Air Quality** across two contrasting regions in Maharashtra — Mumbai and Satara. It uses remote sensing and machine learning to classify land use, predict future air quality trends, and assess potential health risks using explainable AI.

The project combines data from **Landsat-8/9**, **Sentinel-5P**, and government sources, integrating spatial-temporal models to support sustainable urban planning and public health insights.

---

## Objectives

- Classify LULC in Mumbai and Satara using satellite data
- Predict future LULC trends using QGIS and AI models
- Forecast AQI (Air Quality Index) levels from 2017–2035 using GRU+GCN
- Interpret the role of LULC and pollutants on AQI using SHAP and LIME
- Predict diphtheria risk using Composite Kernel Gaussian Process Regression

---

## Technologies Used

| Technology              | Purpose                                      |
| ----------------------- | -------------------------------------------- |
| **Python**              | ML modeling and data preprocessing           |
| **Google Earth Engine** | Satellite data processing and map extraction |
| **QGIS**                | Geospatial visualization and prediction      |
| **Colab**               | Cloud execution of AI models                 |
| **SHAP / LIME**         | Explainable AI frameworks                    |

---

## Models Implemented

- **LULC Classification Models**: 1D-CNN, MLP, LSTM, CNN+LSTM, CNN+MLP, MLP+LSTM

- **AQI Forecasting**: GRU + Graph Convolutional Network (GCN)

- **Explainable AI**: SHAP for LULC interpretation, LIME for AQI classification

- **Health Risk Prediction**: Composite Kernel Gaussian Process Regression (GPR)

---

## Results

- Mumbai shows a sharp rise in built-up areas and worsening AQI by 2035
- Satara maintains higher green cover with lower pollution risk
- SHAP identified built-up and vegetation loss as major drivers
- LIME showed NO₂ and PM2.5 as dominant AQI indicators
- GPR-based forecasting reveals rising diphtheria risk with pollution

---

## Key Learnings

- High correlation between LULC changes and air quality
- Deep learning models outperform traditional classifiers in LULC mapping
- Explainable AI helps in identifying actionable insights
- Remote sensing data significantly aids in environmental prediction

---

## References

- Google Earth Engine
- Maharashtra Pollution Control Board
- Landsat & Sentinel Datasets
- SHAP / LIME Documentation
- Research papers referenced in report

---

## Authors

- Kapil Sunil Bhatia
- Dakshita Sanjay Kolte
- Palak Piyush Desai

Supervised by:  
**Dr. Jyoti Wadmare**  
Dept. of Computer Engineering, KJSIT

---

## Note

This project is academic and research-oriented. Data sources were used under fair academic usage.
