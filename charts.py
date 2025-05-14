import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

TEMP_CHART_DIR = "temp_charts"  # Directory to store temporary chart images
os.makedirs(TEMP_CHART_DIR, exist_ok=True)  # Ensure the directory exists

def create_chart(x_test_index, data, chart_type):
    """
    Generate a chart based on the specified chart type ('prediction' or 'error').
    Ensure all time series have data for the same dates.

    Args:
        x_test_index (array-like): Index for the x-axis (e.g., dates).
        data (dict): Dictionary containing series data for models.
                     For 'prediction', it should include actual values under the key 'Actual'.
        chart_type (str): Type of chart to generate ('prediction' or 'error').

    Returns:
        str: Filepath of the saved chart image.
    """
    # Align lengths of x_test_index and data
    min_length = min(len(x_test_index), *[len(series) for series in data.values()])
    x_test_index = x_test_index[:min_length]

    # Remove duplicates from x_test_index
    unique_index = ~pd.Series(x_test_index).duplicated(keep='first')  # Mask for unique indices
    x_test_index = x_test_index[unique_index]

    # Apply the same mask to data
    data = {
        key: series[:min(len(series), len(x_test_index))].flatten()
        for key, series in data.items()
    }

    # Find common dates across all series
    common_dates = sorted(set(x_test_index))
    data = {
        key: pd.Series(
            series[:min(len(series), len(x_test_index))],
            index=x_test_index[:min(len(series), len(x_test_index))]
        ).reindex(common_dates).dropna().values  # Align data with common_dates
        for key, series in data.items()
    }

    # Plot the chart
    fig, ax = plt.subplots(figsize=(15, 8))
    if chart_type == 'prediction':
        ax.plot(common_dates, data.pop('Actual'), label="Actual", linestyle='dashed')
        for model_name, y_pred in data.items():
            ax.plot(common_dates, y_pred, label=f"Predicted ({model_name})")
        ax.set_title("Predictions vs Actual Values")
        ax.set_ylabel("Values")
    elif chart_type == 'error':
        for model_name, error in data.items():
            ax.plot(common_dates, error, label=f"Error ({model_name})")
        ax.set_title("Error Between Predictions and Actual Values")
        ax.set_ylabel("Error")
    else:
        raise ValueError("Invalid chart_type. Use 'prediction' or 'error'.")

    ax.set_xlabel("Date")
    ax.legend()
    plt.xticks(rotation=45)
    chart_filename = "prediction_chart.png" if chart_type == 'prediction' else "error_chart.png"
    chart_path = os.path.join(TEMP_CHART_DIR, chart_filename)
    plt.savefig(chart_path, format='png', bbox_inches='tight')
    plt.close(fig)
    return chart_path

