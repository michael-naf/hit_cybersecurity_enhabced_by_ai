# EDA.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Config
# -----------------------------
DATASET_LETTERS = ["A", "B", "C"]
BASE_DATA_DIR = "../data"
BASE_OUTPUT_DIR = "../Results/EDA"

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# -----------------------------
# Load Data
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)

    # ניקוי whitespace בעמודת timestamp
    df["timestamp"] = df["timestamp"].astype(str).str.strip()

    # המרת timestamp – מאפשר dayfirst=True כדי לזהות dd/mm/yyyy
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        dayfirst=True,   # חשוב: dd/mm/yyyy
        errors="coerce"  # ערכים לא תקינים יהפכו ל-NaT
    )

    # בדיקה: אם יש ערכי NaT, כדאי לדווח
    if df["timestamp"].isna().any():
        print(f"Warning: {df['timestamp'].isna().sum()} invalid timestamps found in {path}")

    # הוספת עמודת hour
    df["hour"] = df["timestamp"].dt.hour

    # במקרה שיש NaN ב-hour (מ timestamp לא חוקי), אפשר למלא ב- -1 או להשמיט
    df = df.dropna(subset=["hour"])

    # הפיכת hour ל-categorical מסודר מ-0 עד 23 (לגרפים)
    df["hour"] = pd.Categorical(df["hour"], categories=range(24), ordered=True)

    return df


# -----------------------------
# Basic Info
# -----------------------------
def basic_info(df):
    print("\n===== CLASS DISTRIBUTION =====")
    print(df["is_anomaly"].value_counts(normalize=True) * 100)


# -----------------------------
# Plots
# -----------------------------
def plot_anomaly_distribution(df, output_dir):
    ax = sns.countplot(data=df, x="is_anomaly")

    total = len(df)

    for p in ax.patches:
        height = p.get_height()
        percentage = 100 * height / total
        ax.annotate(f"{percentage:.2f}%",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold')

    plt.title("Anomaly Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.savefig(f"{output_dir}/anomaly_distribution.png")
    plt.close()


def plot_command_length(df, output_dir):
    sns.histplot(data=df, x="command_length", hue="is_anomaly",
                 bins=50, kde=True)

    plt.title("Command Length Distribution")
    plt.savefig(f"{output_dir}/command_length_distribution.png")
    plt.close()


def plot_num_arguments(df, output_dir):
    max_val = df["num_arguments"].max()

    ax = sns.histplot(
        data=df,
        x="num_arguments",
        hue="is_anomaly",
        bins=range(0, max_val + 2),
        discrete=True
    )

    ax.set_xticks(range(0, max_val + 1, 1))

    plt.title("Number of Arguments Distribution")
    plt.xlabel("Number of Arguments")
    plt.ylabel("Count")

    plt.savefig(f"{output_dir}/num_arguments_distribution.png")
    plt.close()


def plot_time_distribution(df, output_dir):
    # ודא סדר שעות מ-0 עד 23
    df['hour'] = pd.Categorical(df['hour'], categories=range(24), ordered=True)

    sns.countplot(data=df, x="hour", hue="is_anomaly")

    plt.title("Activity Distribution by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")

    plt.savefig(f"{output_dir}/time_distribution_by_hour.png")
    plt.close()


# -----------------------------
# Run EDA for one dataset
# -----------------------------
def run_eda(dataset_letter):
    dataset_path = f"{BASE_DATA_DIR}/Enterprise_{dataset_letter}.csv"
    output_dir = f"{BASE_OUTPUT_DIR}/Enterprise_{dataset_letter}_EDA"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(dataset_path)

    print(f"\n===== Running EDA for Enterprise_{dataset_letter} =====")
    basic_info(df)

    plot_anomaly_distribution(df, output_dir)
    plot_command_length(df, output_dir)
    plot_num_arguments(df, output_dir)
    plot_time_distribution(df, output_dir)

    print(f"EDA completed. Results saved in '{output_dir}' folder.")


# -----------------------------
# Entry Point: Run all datasets
# -----------------------------
if __name__ == "__main__":
    for letter in DATASET_LETTERS:
        run_eda(letter)