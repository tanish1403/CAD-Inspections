"""Small helpers for Colab / Kaggle notebooks."""
import csv
import glob
import os


def result_dirs(output_dir):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(output_dir, "**", "report.json"), recursive=True))


def show_results(output_dir, width=18, which="result.png", limit=None):
    """Display the report card of every photo (original | result | CAD + status summary)."""
    import cv2
    import matplotlib.pyplot as plt
    dirs = result_dirs(output_dir)[:limit]
    if not dirs:
        print(f"no results found in {output_dir}")
    for d in dirs:
        img = cv2.cvtColor(cv2.imread(os.path.join(d, which)), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(width, width * img.shape[0] / img.shape[1]))
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.relpath(d, output_dir), fontsize=13)
        plt.show()


def summary_table(output_dir, hide_unverifiable=True):
    """pandas DataFrame: one row per (photo, part) with its status and score."""
    import pandas as pd
    p = os.path.join(output_dir, "summary.csv")
    df = pd.read_csv(p)
    return df[df.status != "UNVERIFIABLE"] if hide_unverifiable else df


def problems(output_dir):
    """Only the interesting rows: ABSENT / UNCERTAIN / altered parts."""
    df = summary_table(output_dir)
    return df[(df.status != "PRESENT") | (df.altered.astype(str).str.lower() == "true")].reset_index(drop=True)
