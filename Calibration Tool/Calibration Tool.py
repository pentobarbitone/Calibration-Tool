import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# USER CONFIGURATION
# ==========================

# Folder where input CSVs live (relative to this script)
DATA_FOLDER = "data"

# Folder where outputs will be written
OUTPUT_FOLDER = "output"

# Names of the columns in your CSV
REFERENCE_COLUMN = "reference"  # true/calibrator value
READING_COLUMN = "reading"      # device output

# Degree of polynomial for calibration (1 = linear)
POLY_DEGREE = 1


# ==========================
# HELPER FUNCTIONS
# ==========================

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a single CSV file into a pandas DataFrame and
    check that required columns exist.
    """
    df = pd.read_csv(csv_path)

    if REFERENCE_COLUMN not in df.columns or READING_COLUMN not in df.columns:
        raise ValueError(
            f"File {csv_path} must contain columns "
            f"'{REFERENCE_COLUMN}' and '{READING_COLUMN}'. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with missing values in those columns
    df = df[[REFERENCE_COLUMN, READING_COLUMN]].dropna()

    return df


def perform_regression(x: np.ndarray, y: np.ndarray, degree: int = 1):
    """
    Fit a polynomial of given degree using least squares.

    Returns:
        coeffs: array of polynomial coefficients (highest power first)
        coeffs_std: array of standard deviations of coefficients
        r_squared: coefficient of determination
        residual_std: standard deviation of residuals (sigma)
        y_fit: fitted y values
    """
    # Fit polynomial and get covariance matrix for coefficients
    coeffs, cov = np.polyfit(x, y, deg=degree, cov=True)

    # Standard deviation of coefficients = sqrt(diagonal of covariance)
    coeffs_std = np.sqrt(np.diag(cov))

    # Evaluate fitted y values
    y_fit = np.polyval(coeffs, x)

    # Residuals and residual statistics
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    # R^2
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Residual standard deviation (sigma)
    dof = max(len(x) - (degree + 1), 1)  # degrees of freedom
    residual_std = np.sqrt(ss_res / dof)

    return coeffs, coeffs_std, r_squared, residual_std, y_fit


def make_plot(x: np.ndarray,
              y: np.ndarray,
              x_fit: np.ndarray,
              y_fit: np.ndarray,
              residual_std: float,
              title: str,
              save_path: str):
    """
    Make a scatter plot of data + fitted curve + +/- 2*residual_std band.
    """
    plt.figure()
    # Scatter of data
    plt.scatter(x, y, label="Data points")

    # Fitted curve
    plt.plot(x_fit, y_fit, label="Fitted curve")

    # Simple ±2σ band as an approximate uncertainty band
    upper = y_fit + 2 * residual_std
    lower = y_fit - 2 * residual_std
    plt.fill_between(x_fit, lower, upper, alpha=0.2, label="±2σ band")

    plt.xlabel(f"{REFERENCE_COLUMN} (reference)")
    plt.ylabel(f"{READING_COLUMN} (device reading)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(device_name: str,
                 coeffs,
                 coeffs_std,
                 r_squared: float,
                 residual_std: float,
                 summary_path: str,
                 per_device_path: str):
    """
    Save regression results:

    - Per-device CSV with all coefficients and stats.
    - Summary row appended to summary CSV.
    """
    degree = len(coeffs) - 1

    # Build per-device table
    rows = []
    for i, (c, s) in enumerate(zip(coeffs, coeffs_std)):
        power = degree - i
        rows.append({
            "parameter": f"a_{power}",  # e.g., a_1 (slope), a_0 (intercept) for degree=1
            "value": c,
            "std_dev": s
        })

    rows.append({"parameter": "r_squared", "value": r_squared, "std_dev": np.nan})
    rows.append({"parameter": "residual_std", "value": residual_std, "std_dev": np.nan})

    per_device_df = pd.DataFrame(rows)
    per_device_df.to_csv(per_device_path, index=False)

    # Append or create summary CSV
    summary_row = {
        "device": device_name,
        "degree": degree,
        "r_squared": r_squared,
        "residual_std": residual_std
    }

    # Add slope and intercept if they exist
    if degree >= 1:
        # For degree=1, coeffs = [slope, intercept]
        summary_row["slope"] = coeffs[-2] if degree >= 1 else np.nan
        summary_row["intercept"] = coeffs[-1]

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(summary_path, index=False)


# ==========================
# MAIN BATCH PIPELINE
# ==========================

def process_all_devices():
    """
    Find all CSV files in DATA_FOLDER, run regression, save plots and results.
    """
    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Path for overall summary CSV
    summary_path = os.path.join(OUTPUT_FOLDER, "calibration_summary.csv")

    # Find all CSV files in data folder
    pattern = os.path.join(DATA_FOLDER, "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found in '{DATA_FOLDER}' folder. "
              f"Please add some CSVs and try again.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in '{DATA_FOLDER}'.")
    print("Processing...")

    for csv_path in csv_files:
        device_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n=== Device: {device_name} ===")
        print(f"Reading data from {csv_path} ...")

        try:
            df = load_dataset(csv_path)
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue

        x = df[REFERENCE_COLUMN].values
        y = df[READING_COLUMN].values

        if len(x) < POLY_DEGREE + 2:
            print(f"  Not enough data points for degree {POLY_DEGREE} fit. Skipping.")
            continue

        # Perform regression
        coeffs, coeffs_std, r_squared, residual_std, y_fit = perform_regression(
            x, y, degree=POLY_DEGREE
        )

        # Make a smooth x grid for plotting fitted curve
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit_plot = np.polyval(coeffs, x_fit)

        # Save plot
        plot_path = os.path.join(OUTPUT_FOLDER, f"{device_name}_fit.png")
        plot_title = f"Calibration Curve - {device_name}"
        make_plot(x, y, x_fit, y_fit_plot, residual_std, plot_title, plot_path)
        print(f"  Saved plot to {plot_path}")

        # Save per-device results
        per_device_path = os.path.join(OUTPUT_FOLDER, f"{device_name}_results.csv")
        save_results(device_name, coeffs, coeffs_std, r_squared, residual_std,
                     summary_path, per_device_path)
        print(f"  Saved detailed results to {per_device_path}")
        print(f"  R^2 = {r_squared:.4f}, residual σ ≈ {residual_std:.4g}")

    print("\nDone. Summary file:")
    print(f"  {summary_path}")


if __name__ == "__main__":
    process_all_devices()
