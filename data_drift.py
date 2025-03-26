import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
import streamlit as st

def compute_ks_test(ref_series, new_series):
    """
    Compute the KS test statistic and p-value for two samples.
    """
    statistic, p_value = ks_2samp(ref_series, new_series)
    return statistic, p_value

def compute_js_divergence(ref_series, new_series, bins=50):
    """
    Compute the Jensen-Shannon divergence between two numeric distributions.
    The function bins the data into a histogram, converts counts into a probability
    distribution, and then computes the JS divergence.
    """
    ref_hist, bin_edges = np.histogram(ref_series, bins=bins, density=True)
    new_hist, _ = np.histogram(new_series, bins=bin_edges, density=True)
    
    # Add a small constant to avoid division by zero issues
    epsilon = 1e-10
    ref_hist += epsilon
    new_hist += epsilon
    
    # Normalize the histograms to form probability distributions
    ref_prob = ref_hist / np.sum(ref_hist)
    new_prob = new_hist / np.sum(new_hist)
    
    m = 0.5 * (ref_prob + new_prob)
    js_div = 0.5 * (entropy(ref_prob, m) + entropy(new_prob, m))
    return js_div

def detect_data_drift(ref_df, new_df, threshold_p=0.05, js_threshold=0.1):
    """
    For each common column between the two datasets, compute the KS test (if numeric)
    and JS divergence. The function returns a dictionary with test statistics and a 
    flag indicating whether drift was detected based on the thresholds provided.
    """
    drift_results = {}
    # Use only columns present in both datasets
    common_columns = list(set(ref_df.columns) & set(new_df.columns))
    
    for col in common_columns:
        if pd.api.types.is_numeric_dtype(ref_df[col]):
            # Drop NaNs to ensure tests run correctly
            ks_stat, ks_p = compute_ks_test(ref_df[col].dropna(), new_df[col].dropna())
            js_div = compute_js_divergence(ref_df[col].dropna(), new_df[col].dropna())
            drift_results[col] = {
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p,
                "js_divergence": js_div,
                "drift_detected_ks": ks_p < threshold_p,
                "drift_detected_js": js_div > js_threshold,
            }
        else:
            # For categorical features, compute normalized value counts
            ref_counts = ref_df[col].value_counts(normalize=True)
            new_counts = new_df[col].value_counts(normalize=True)
            common_index = ref_counts.index.intersection(new_counts.index)
            ref_probs = ref_counts[common_index].values
            new_probs = new_counts[common_index].values
            
            # Smoothing to avoid zeros
            epsilon = 1e-10
            ref_probs += epsilon
            new_probs += epsilon
            ref_probs = ref_probs / np.sum(ref_probs)
            new_probs = new_probs / np.sum(new_probs)
            
            m = 0.5 * (ref_probs + new_probs)
            js_div = 0.5 * (entropy(ref_probs, m) + entropy(new_probs, m))
            drift_results[col] = {
                "ks_statistic": None,
                "ks_p_value": None,
                "js_divergence": js_div,
                "drift_detected_ks": None,
                "drift_detected_js": js_div > js_threshold,
            }
    return drift_results

def load_csv_data(file_path):
    """
    Utility function to load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)

# Streamlit Interface
def run_streamlit_app():
    st.title("Automated Data Drift Monitoring Tool")
    st.markdown("This tool detects data drift between a reference dataset and new data using statistical tests.")

    # File upload widgets for CSV files
    ref_file = st.file_uploader("Upload Reference CSV", type=["csv"])
    new_file = st.file_uploader("Upload New Data CSV", type=["csv"])
    
    if ref_file is not None and new_file is not None:
        ref_df = pd.read_csv(ref_file)
        new_df = pd.read_csv(new_file)
        
        st.subheader("Data Preview")
        st.markdown("**Reference Data:**")
        st.dataframe(ref_df.head())
        st.markdown("**New Data:**")
        st.dataframe(new_df.head())
        
        if st.button("Detect Data Drift"):
            drift_results = detect_data_drift(ref_df, new_df)
            st.subheader("Drift Detection Results")
            for col, result in drift_results.items():
                st.markdown(f"**Column:** {col}")
                st.json(result)

if __name__ == "__main__":
    # Determine if the script should run in Streamlit mode or CLI mode.
    if "streamlit" in os.environ.get("STREAMLIT_SERVER_PORT", ""):
        run_streamlit_app()
    else:
        parser = argparse.ArgumentParser(description="Automated Data Drift Monitoring Tool")
        parser.add_argument("--ref", type=str, required=True, help="Path to the reference CSV file")
        parser.add_argument("--new", type=str, required=True, help="Path to the new data CSV file")
        parser.add_argument("--threshold_p", type=float, default=0.05, help="KS test p-value threshold")
        parser.add_argument("--js_threshold", type=float, default=0.1, help="Jensen-Shannon divergence threshold")
        args = parser.parse_args()
        
        ref_df = load_csv_data(args.ref)
        new_df = load_csv_data(args.new)
        
        results = detect_data_drift(ref_df, new_df, threshold_p=args.threshold_p, js_threshold=args.js_threshold)
        for col, res in results.items():
            print(f"Column: {col}")
            print(res)