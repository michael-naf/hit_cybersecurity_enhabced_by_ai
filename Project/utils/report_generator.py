# report_generator.py
import os
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from sklearn.metrics import confusion_matrix
import pandas as pd


def export_metrics_to_csv(filename, agents_info, ensemble_info=None):
    rows = []

    for info in agents_info:
        metrics = info.get("metrics", {})

        rows.append({
            "Agent name": info.get("name"),
            "Recall": metrics.get("Recall"),
            "Precision": metrics.get("Precision"),
            "F1 Score": metrics.get("F1_score")
        })

    if ensemble_info is not None:
        metrics = ensemble_info.get("metrics", {})

        rows.append({
            "Agent name": "Ensemble",
            "Recall": metrics.get("Recall"),
            "Precision": metrics.get("Precision"),
            "F1 Score": metrics.get("F1_score")
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

    print(f"Metrics CSV saved to: {filename}")

def generate_pdf_report(filename, agents_info, ensemble_info, weights=None):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50

    def write_section(title, metrics, graph_path, show_weights=False):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"----- {title} -----")
        y -= 25

        c.setFont("Helvetica", 12)

        # Write metrics
        for key, value in metrics.items():
            c.drawString(70, y, f"{key}: {value:.4f}")
            y -= 20

        # Write weights with agent names
        if show_weights and weights is not None:
            y -= 10
            c.setFont("Helvetica-Bold", 12)
            c.drawString(70, y, "Weights:")
            y -= 20
            c.setFont("Helvetica", 12)

            for agent_info, weight in zip(agents_info, weights):
                agent_name = agent_info["name"]
                c.drawString(90, y, f"{agent_name}: {weight:.2f}")
                y -= 20

        # Draw graph
        if os.path.exists(graph_path):
            img = ImageReader(graph_path)
            img_width, img_height = img.getSize()
            max_width = 400
            max_height = 200
            aspect = img_height / img_width

            if max_height / max_width < aspect:
                draw_height = max_height
                draw_width = draw_height / aspect
            else:
                draw_width = max_width
                draw_height = draw_width * aspect

            if y - draw_height < 50:
                c.showPage()
                y = A4[1] - 50

            c.drawImage(img, 70, y - draw_height, width=draw_width, height=draw_height)
            y -= draw_height + 10
        else:
            y -= 20

        y -= 10

    # Write individual agents
    for info in agents_info:
        write_section(info['name'], info['metrics'], info['graph'])

    # Write ensemble (with named weights)
    write_section("Ensemble", ensemble_info['metrics'], ensemble_info['graph'], show_weights=True)

    c.save()


def plot_confusion_matrix_graph(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap='Blues')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compare_enterprises(results_folder="Results"):

    files = {
        "Enterprise_A": f"{results_folder}/Enterprise_A/Enterprise_A_metrics_summary.csv",
        "Enterprise_B": f"{results_folder}/Enterprise_B/Enterprise_B_metrics_summary.csv",
        "Enterprise_C": f"{results_folder}/Enterprise_C/Enterprise_C_metrics_summary.csv",
    }

    for f in files.values():
        if not os.path.exists(f):
            print("❌ Not all enterprise CSV files exist. Skipping comparison.")
            return

    print("✅ All enterprise CSV files found. Generating comparison graphs...")

    # יצירת תיקיית output
    output_folder = os.path.join(results_folder, "Enterprise_Comparison")
    os.makedirs(output_folder, exist_ok=True)

    # טעינת הטבלאות
    dfs = {name: pd.read_csv(path) for name, path in files.items()}

    agents = dfs["Enterprise_A"]["Agent name"]

    metrics = ["Recall", "Precision", "F1 Score"]

    for metric in metrics:

        fig, ax = plt.subplots(figsize=(8, 5))

        width = 0.25
        x = range(len(agents))

        for i, (enterprise, df) in enumerate(dfs.items()):
            values = df[metric]
            positions = [p + width * i for p in x]

            ax.bar(positions, values, width=width, label=enterprise)

        ax.set_title(f"{metric} Comparison Across Enterprises")
        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels(agents, rotation=30)
        ax.set_ylabel(metric)
        ax.legend()

        plt.tight_layout()

        save_path = os.path.join(output_folder, f"{metric}_comparison.png")
        plt.savefig(save_path)
        plt.close()

        print(f"📊 Saved graph: {save_path}")

    print("\n✅ Enterprise comparison graphs created.")