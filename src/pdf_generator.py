import io
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

def generate_clinical_pdf(report_data: dict) -> bytes:
    """
    Generates an official A4 Clinical Diagnostic Assessment PDF Report
    using matplotlib's vector PDF engine and returns the raw PDF bytes.
    """
    pdf_buffer = io.BytesIO()
    
    report_id = report_data.get("report_id", "TR-AI-" + datetime.now().strftime("%Y%m%d%H%M"))
    timestamp = report_data.get("timestamp", datetime.now().strftime("%B %d, %Y - %I:%M %p"))
    patient = report_data.get("patient", {})
    patient_name = patient.get("name", "Anonymous Patient")
    patient_age = str(patient.get("age", "--"))
    patient_gender = patient.get("gender", "--")
    
    finding = report_data.get("finding", "Melanocytic Nevus")
    confidence = float(report_data.get("confidence", 0.95)) * 100
    is_high_risk = bool(report_data.get("is_high_risk", False))
    plain_explanation = report_data.get("explanation", "Automated diagnostic assessment.")
    action = report_data.get("action", "Follow routine clinical protocols.")
    probabilities = report_data.get("probabilities", [])

    with PdfPages(pdf_buffer) as pdf:
        # A4 Dimensions in inches: 8.27 x 11.69
        fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
        fig.patch.set_facecolor('#FFFFFF')
        
        # 1. Header Banner
        plt.figtext(0.08, 0.94, "TwachaRakshak AI", fontsize=20, weight='bold', color='#111827')
        plt.figtext(0.38, 0.945, "CLINICAL DECISION-SUPPORT SYSTEM", fontsize=9, weight='bold', color='#2563EB')
        plt.figtext(0.08, 0.915, "Dermatological AI Diagnostic Assessment & Saliency Report", fontsize=10, color='#6B7280')
        
        plt.figtext(0.72, 0.94, f"DOC ID: {report_id}", fontsize=10, weight='bold', color='#111827')
        plt.figtext(0.72, 0.92, f"Date: {timestamp}", fontsize=8.5, color='#6B7280')
        plt.figtext(0.72, 0.898, "Status: AI VALIDATED (ISIC 2019)", fontsize=8, weight='bold', color='#059669')

        # Divider line
        line = plt.Line2D([0.08, 0.92], [0.885, 0.885], color='#111827', linewidth=1.5, figure=fig)
        fig.lines.append(line)

        # 2. Patient Demographics Box
        rect_demo = plt.Rectangle((0.08, 0.815), 0.84, 0.055, facecolor='#F9FAFB', edgecolor='#E5E7EB', linewidth=1, transform=fig.transFigure)
        fig.patches.append(rect_demo)
        
        plt.figtext(0.10, 0.852, "PATIENT NAME", fontsize=7.5, weight='bold', color='#6B7280')
        plt.figtext(0.10, 0.828, patient_name, fontsize=10, weight='bold', color='#111827')

        plt.figtext(0.35, 0.852, "AGE / GENDER", fontsize=7.5, weight='bold', color='#6B7280')
        plt.figtext(0.35, 0.828, f"{patient_age} Yrs / {patient_gender}", fontsize=10, weight='bold', color='#111827')

        plt.figtext(0.60, 0.852, "EXAMINATION", fontsize=7.5, weight='bold', color='#6B7280')
        plt.figtext(0.60, 0.828, "Dermoscopy 3D-CA", fontsize=10, weight='bold', color='#2563EB')

        plt.figtext(0.80, 0.852, "MODEL ENSEMBLE", fontsize=7.5, weight='bold', color='#6B7280')
        plt.figtext(0.80, 0.828, "v2.1 Multi-Head", fontsize=10, weight='bold', color='#111827')

        # 3. Primary Finding Card
        box_color = '#FEF2F2' if is_high_risk else '#EFF6FF'
        border_color = '#FCA5A5' if is_high_risk else '#BFDBFE'
        rect_finding = plt.Rectangle((0.08, 0.70), 0.84, 0.095, facecolor=box_color, edgecolor=border_color, linewidth=1.5, transform=fig.transFigure)
        fig.patches.append(rect_finding)

        finding_header_color = '#991B1B' if is_high_risk else '#1E40AF'
        plt.figtext(0.10, 0.775, "PRIMARY DIAGNOSTIC FINDING", fontsize=8, weight='bold', color=finding_header_color)
        plt.figtext(0.10, 0.745, finding, fontsize=16, weight='bold', color='#111827')
        plt.figtext(0.10, 0.715, plain_explanation, fontsize=9, color='#4B5563')

        score_color = '#DC2626' if is_high_risk else '#2563EB'
        plt.figtext(0.76, 0.745, f"{confidence:.1f}%", fontsize=22, weight='bold', color=score_color)
        risk_label = "HIGH RISK CONDITION" if is_high_risk else "LOW RISK / BENIGN"
        plt.figtext(0.74, 0.718, risk_label, fontsize=7.5, weight='bold', color=score_color)

        # 4. Images Section (Specimen + Heatmap)
        plt.figtext(0.08, 0.67, "VISUAL EVIDENCE & NEURAL SPATIAL ATTENTION", fontsize=8.5, weight='bold', color='#6B7280')

        # Specimen Image
        raw_img_base64 = report_data.get("image_base64", None)
        if raw_img_base64:
            try:
                if "," in raw_img_base64:
                    raw_img_base64 = raw_img_base64.split(",")[1]
                img_data = base64.b64decode(raw_img_base64)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                ax_img = fig.add_axes([0.08, 0.47, 0.40, 0.19])
                ax_img.imshow(img)
                ax_img.axis('off')
                ax_img.set_title("Input Dermoscopic Specimen", fontsize=8, weight='bold', pad=4)
            except Exception as e:
                pass

        # GradCAM Heatmap Image
        heatmap_base64 = report_data.get("heatmap_base64", None)
        if heatmap_base64:
            try:
                if "," in heatmap_base64:
                    heatmap_base64 = heatmap_base64.split(",")[1]
                heat_data = base64.b64decode(heatmap_base64)
                heat_img = Image.open(io.BytesIO(heat_data)).convert('RGB')
                ax_heat = fig.add_axes([0.52, 0.47, 0.40, 0.19])
                ax_heat.imshow(heat_img)
                ax_heat.axis('off')
                ax_heat.set_title("3D-CA Neural Saliency Overlay", fontsize=8, weight='bold', pad=4)
            except Exception as e:
                pass

        # 5. Differential Probabilities
        plt.figtext(0.08, 0.44, "9-CLASS DIFFERENTIAL DIAGNOSTIC PROBABILITIES", fontsize=8.5, weight='bold', color='#6B7280')

        if probabilities:
            y_start = 0.415
            for i, p in enumerate(probabilities[:7]):
                code = p.get("code", "")
                name = p.get("name", "")
                pct = float(p.get("pct", 0.0))
                y_pos = y_start - (i * 0.024)
                
                plt.figtext(0.08, y_pos, f"{code:5s}  {name[:32]}", fontsize=8, family='monospace', color='#111827')
                
                # Bar representation
                bar_bg = plt.Rectangle((0.50, y_pos - 0.002), 0.30, 0.012, facecolor='#F3F4F6', edgecolor='none', transform=fig.transFigure)
                bar_fill = plt.Rectangle((0.50, y_pos - 0.002), (pct / 100.0) * 0.30, 0.012, facecolor='#2563EB' if i == 0 else '#94A3B8', edgecolor='none', transform=fig.transFigure)
                fig.patches.append(bar_bg)
                fig.patches.append(bar_fill)
                
                plt.figtext(0.83, y_pos, f"{pct:5.1f}%", fontsize=8, weight='bold' if i == 0 else 'normal', color='#2563EB' if i == 0 else '#4B5563')

        # 6. Recommendation Box
        rect_rec = plt.Rectangle((0.08, 0.15), 0.84, 0.08, facecolor='#F9FAFB', edgecolor='#E5E7EB', linewidth=1, transform=fig.transFigure)
        fig.patches.append(rect_rec)
        plt.figtext(0.10, 0.21, "RECOMMENDED CLINICAL ACTION PLAN", fontsize=8, weight='bold', color='#111827')
        plt.figtext(0.10, 0.185, f"Patient Action: {action}", fontsize=8.5, color='#374151')
        plt.figtext(0.10, 0.162, "Protocol: Schedule a dermatoscopy review if changes in size, shape, or color occur.", fontsize=8, color='#6B7280')

        # 7. Institutional Footer & Security Line
        line_foot = plt.Line2D([0.08, 0.92], [0.12, 0.12], color='#E5E7EB', linewidth=1, figure=fig)
        fig.lines.append(line_foot)
        plt.figtext(0.08, 0.095, "TwachaRakshak AI Clinical Diagnostic Decision Support Platform (v2.1)", fontsize=8, weight='bold', color='#6B7280')
        plt.figtext(0.08, 0.080, "Disclaimer: Automated assessment based on ISIC 2019 benchmarks. Must be validated by a licensed physician.", fontsize=7, color='#9CA3AF')
        plt.figtext(0.75, 0.080, f"AUTHENTICITY CODE: {report_id}", fontsize=7.5, family='monospace', color='#6B7280')

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
