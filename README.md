# Automated Data Drift Monitoring Tool

This project is designed to automatically detect data drift between a reference dataset and a new dataset. It leverages statistical tests such as the Kolmogorov–Smirnov (KS) test for numeric features and Jensen–Shannon (JS) divergence for both numeric and categorical features. The tool is versatile and can be deployed either as a command-line interface (CLI) application or through a Streamlit web interface.

## Overview

Data drift refers to changes in the statistical properties of data over time, which can affect the performance of machine learning models. This tool provides:
- **Automated Drift Detection:** Uses the KS test and JS divergence to identify significant differences in data distributions.
- **Versatility:** Can be run as a CLI tool for batch analysis or deployed as a Streamlit web application for interactive exploration.
- **Extensibility:** A modular design that allows for easy customization and extension.

## Features

- **Kolmogorov–Smirnov (KS) Test:** Compares the distribution of a numeric feature between the reference and new data.
- **Jensen–Shannon Divergence:** Quantifies the similarity between probability distributions for both numeric and categorical features.
- **Flexible Data Input:** Supports CSV files as input for both reference and new datasets.
- **Dual Deployment Modes:**
  - **CLI Mode:** Run the tool with command-line arguments.
  - **Streamlit Mode:** Launch an interactive web interface for real-time drift detection.
- **Clear Reporting:** Outputs statistical metrics and drift detection flags for each common column in the datasets.

## Prerequisites

- **Python 3.7 or higher**
- **Required Python Packages:**
  - `numpy`
  - `pandas`
  - `scipy`
  - `streamlit` (only if using the web interface)
  - `argparse` (typically included with Python)

You can install the required packages using pip:

```bash
pip install numpy pandas scipy streamlit
