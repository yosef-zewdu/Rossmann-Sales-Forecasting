# Rossmann Sales Forecasting

## Project Overview

This repository contains a machine learning project aimed at forecasting sales for Rossmann Pharmaceuticals across multiple stores and cities. The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores. The goal is to provide accurate sales predictions six weeks in advance, utilizing various factors such as promotions, competition, holidays, seasonality, and locality.

## Business Objective

The project aim to provide accurate sales predictions six weeks in advance. The goal of the project is to build and serve an end-to-end product that delivers the prediction to analysts in the finance team.

The task is divided into the following specific objectives

- Exploration of customer purchasing behavior

- Prediction of store sales

  - Machine learning approach

  - Deep Learning approach

- Serving predictions on a web interface

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling.
- **Model Development**: Implementation of machine learning models for sales prediction.
- **Model Evaluation**: Metrics to assess model performance (e.g., RMSE, MAE).
- **Deployment**: Serving the model through a RESTful API for easy access by analysts.

## Project Structure

```plaintext

Rossmann-Sales-Forecasting/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml   # GitHub Actions
├── src/
│   └── __init__.py
├── notebooks/
|   |data_processing.ipynb              # Jupyter notebook for data cleaning and processing 
│   └── README.md                       # Description of notebooks directory 
├── tests/
│   └── __init__.py
├── scripts/
|    ├── __init__.py
│    ├── data_processing.py              # Script data processing, cleaning 
│    ├── plots.py                        # Script plots
│    └── README.md                       # Description of scripts directory
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # License information
└── .gitignore              # Files and directories to ignore in Git  
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Rossmann-Sales-Forecasting.git
   cd rossmann-sales-forecasting


2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv\Scripts\activate  # On Linux, use `venv/bin/activate`
   

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

