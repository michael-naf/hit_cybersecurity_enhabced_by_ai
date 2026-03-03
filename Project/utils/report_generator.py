# report_generator.py
import os
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from sklearn.metrics import confusion_matrix

def generate_pdf_report(filename, agents_info, ensemble_info):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50

    def write_section(title, metrics, graph_path):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"----- {title} -----")
        y -= 25

        c.setFont("Helvetica", 12)
        for key, value in metrics.items():
            c.drawString(70, y, f"{key}: {value:.4f}")
            y -= 20

        # Draw graph with correct aspect ratio
        if os.path.exists(graph_path):
            img = ImageReader(graph_path)
            img_width, img_height = img.getSize()
            max_width = 400
            max_height = 200
            aspect = img_height / img_width

            # Compute new dimensions keeping aspect ratio
            if max_height / max_width < aspect:
                draw_height = max_height
                draw_width = draw_height / aspect
            else:
                draw_width = max_width
                draw_height = draw_width * aspect

            # --- Check if there's enough space, else new page ---
            if y - draw_height < 50:  # 50 px margin bottom
                c.showPage()
                y = A4[1] - 50

            c.drawImage(img, 70, y - draw_height, width=draw_width, height=draw_height)
            y -= draw_height + 10
        else:
            y -= 20

        y -= 10
    # Write each agent
    for info in agents_info:
        write_section(info['name'], info['metrics'], info['graph'])

    # Write ensemble
    write_section("Ensemble", ensemble_info['metrics'], ensemble_info['graph'])

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
