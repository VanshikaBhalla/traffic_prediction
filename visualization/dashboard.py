"""
Optional dashboard for real-time visualization.
You can extend with Streamlit, Dash, or matplotlib live plots.
"""

import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, sensor_id=0):
    plt.plot(y_true[:, sensor_id], label="True")
    plt.plot(y_pred[:, sensor_id], label="Pred")
    plt.legend()
    plt.title(f"Sensor {sensor_id} Prediction")
    plt.show()
