import os
import io
import base64
import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
from datetime import datetime
import tempfile
import time
from scoring_system import CustomScoringSystem, DiagnosisScore


LOCAL_MODEL_PATH = r"c:\NEWPROJECT\medgemma-4b-it"

app = FastAPI(title="Medical Image Analysis System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def load_model():
    global model
    try:
        print("Loading model from:", LOCAL_MODEL_PATH)
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        model = pipeline(
            "image-text-to-text",
            model=LOCAL_MODEL_PATH,
            model_kwargs=model_kwargs,
        )
        model.model.generation_config.do_sample = False
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        print(traceback.format_exc())


def parse_model_json(text: str):
    """
    Extract JSON object from model output.
    Returns Python dict on success, otherwise None.
    """
    if not isinstance(text, str):
        return None
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        raw = text[start:end]
        raw = raw.strip('` \n\r\t')
        return json.loads(raw)
    except Exception:
        try:
            import re
            matches = re.findall(r'\{(?:[^{}]|(?R))*\}', text, flags=re.DOTALL)
            for m in matches:
                try:
                    return json.loads(m)
                except Exception:
                    continue
        except Exception:
            pass
    return None


def analyze_pathology(image):
    try:
        prompt = """
You are a board-certified pathologist. Examine the histopathology image and return EXACT JSON:

{
 "diagnosis": "<benign | malignant | suspicious>",
 "malignancy_features": ["mitotic_activity", "necrosis", "atypical_nuclei", "invasion"],
 "benign_features": ["benign", "normal", "cyst", "hyperplasia"],
 "overall_assessment": "<short summary>",
 "malignancy_probability": <0.00-1.00>,
 "recommendation": "<biopsy | follow-up | further_staining>"
}

Use EXACT keyword tokens: mitotic_activity, necrosis, atypical_nuclei, invasion, benign.
Do not add extra text outside JSON.
"""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a pathology diagnosis assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image", "image": image}]}
        ]

        start_time = time.time()
        output = model(text=messages, max_new_tokens=300)
        response = output[0]["generated_text"][-1]["content"]

        elapsed_time = time.time() - start_time
        scorer = CustomScoringSystem("pathology")
        diagnosis_score = scorer.score_analysis(response)

        return response, "Pathology Cancer Assessment", elapsed_time, diagnosis_score

    except Exception as e:
        return f"Error during analysis: {str(e)}", "Pathology Cancer Assessment", 0, None


def analyze_mammography(image):
    try:
        prompt = """
You are a breast imaging radiologist. Analyze the mammography image and return EXACT JSON:

{
 "breast_density": "A | B | C | D",
 "suspicious_findings": ["spiculated", "microcalcifications", "architectural_distortion"],
 "birads": "<0-6>",
 "likelihood_malignancy": <0.00-1.00>,
 "overall_impression": "<benign | probably_benign | suspicious | highly_suspicious>",
 "recommendation": "<follow-up | biopsy | routine>"
}

Use EXACT tokens: birads_5, birads_4, microcalcifications, spiculated.
No text outside JSON.
"""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a mammography cancer detection assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image", "image": image}]}
        ]

        start_time = time.time()
        output = model(text=messages, max_new_tokens=300)
        response = output[0]["generated_text"][-1]["content"]

        elapsed_time = time.time() - start_time
        scorer = CustomScoringSystem("mammography")
        diagnosis_score = scorer.score_analysis(response)

        return response, "Mammography Breast Cancer Analysis", elapsed_time, diagnosis_score

    except Exception as e:
        return f"Error during analysis: {str(e)}", "Mammography Breast Cancer Analysis", 0, None


def analyze_brain_mri(image):
    try:
        prompt = """
You are a neuroradiologist. Analyze the brain MRI and return EXACT JSON:

{
 "tumor_presence": "yes | no",
 "tumor_type": "glioma | meningioma | pituitary | no_tumor",
 "malignant_features": ["necrosis", "edema", "enhancement", "mass_effect"],
 "benign_features": ["low_grade", "meningioma", "stable"],
 "tumor_location": "<brain region>",
 "summary": "<1-2 sentence clinical note>",
 "risk_score": <0.00-1.00>,
 "recommendation": "<urgent | follow-up | no_action>"
}

Use EXACT tokens: necrosis, edema, enhancement, high_grade, low_grade.
Return only JSON.
"""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a brain tumor MRI analysis assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image", "image": image}]}
        ]

        start_time = time.time()
        output = model(text=messages, max_new_tokens=350)
        response = output[0]["generated_text"][-1]["content"]

        elapsed_time = time.time() - start_time
        scorer = CustomScoringSystem("brain_mri")
        diagnosis_score = scorer.score_analysis(response)

        return response, "Brain Tumor MRI Analysis", elapsed_time, diagnosis_score

    except Exception as e:
        return f"Error during analysis: {str(e)}", "Brain Tumor MRI Analysis", 0, None


def analyze_skin_cancer(image):
    try:
        prompt = """
You are a dermatologist. Analyze the dermatoscopic image and return EXACT JSON:

{
 "classification": "melanoma | nevus | basal_cell | benign_keratosis | actinic_keratosis",
 "abcd": {
    "asymmetry": true/false,
    "border_irregularity": true/false,
    "color_variation": true/false,
    "diameter_mm": <number>
 },
 "malignant_keywords": ["asymmetric", "irregular_border", "multiple_colors"],
 "benign_keywords": ["symmetric", "regular_border", "uniform_color"],
 "malignancy_probability": <0.00-1.00>,
 "recommendation": "<biopsy | follow-up | reassure>"
}

Use EXACT keyword tokens: melanoma, irregular_border, multiple_colors, benign_nevus.
Only output JSON.
"""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a skin cancer analysis assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image", "image": image}]}
        ]

        start_time = time.time()
        output = model(text=messages, max_new_tokens=350)
        response = output[0]["generated_text"][-1]["content"]

        elapsed_time = time.time() - start_time
        scorer = CustomScoringSystem("skin_cancer")
        diagnosis_score = scorer.score_analysis(response)

        return response, "Skin Cancer Dermatoscopic Analysis", elapsed_time, diagnosis_score

    except Exception as e:
        return f"Error during analysis: {str(e)}", "Skin Cancer Dermatoscopic Analysis", 0, None


def analyze_lung_cancer(image):
    try:
        prompt = """
You are a thoracic radiologist. Analyze the chest CT/X-ray and return EXACT JSON:

{
 "findings": [
   {
     "id": "F1",
     "location": "RUL | RML | RLL | LUL | LLL",
     "size_mm": <number>,
     "margins": "spiculated | smooth | cavitary",
     "density": "solid | part-solid | ground-glass",
     "suspicious_features": ["spiculated", "pleural_involvement", "lymphadenopathy"]
   }
 ],
 "overall_assessment": "benign | low_risk | intermediate_risk | high_risk",
 "suspected_cancer_type": "adenocarcinoma | squamous_cell | small_cell | none",
 "malignancy_probability": <0.00-1.00>,
 "recommendation": "<PET-CT | biopsy | follow-up>"
}

Use EXACT keywords: spiculated, cavitary, pleural_involvement, lymphadenopathy.
Return ONLY JSON.
"""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are a lung cancer CT detection assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image", "image": image}]}
        ]

        start_time = time.time()
        output = model(text=messages, max_new_tokens=350)
        response = output[0]["generated_text"][-1]["content"]

        elapsed_time = time.time() - start_time
        scorer = CustomScoringSystem("lung_cancer")
        diagnosis_score = scorer.score_analysis(response)

        return response, "Lung Cancer Screening Analysis", elapsed_time, diagnosis_score

    except Exception as e:
        return f"Error during analysis: {str(e)}", "Lung Cancer Screening Analysis", 0, None


def generate_pdf_report(analysis_type, response, image_bytes, diagnosis_score: DiagnosisScore = None, parsed_json: dict = None):
    """
    Build human-friendly PDF with structured content rendering.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=8
    )
    body_style = styles['BodyText']

    temp_pdf = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", mode='w+b')
    pdf_path = temp_pdf.name
    temp_pdf.close()

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    story.append(Paragraph("Medical Image Analysis Report", title_style))
    story.append(Spacer(1, 0.15*inch))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Report Date:</b> {timestamp}", body_style))
    story.append(
        Paragraph(f"<b>Analysis Type:</b> {analysis_type}", body_style))
    story.append(Spacer(1, 0.25*inch))

    story.append(Paragraph("Analyzed Image", heading_style))
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_width = 4*inch
        aspect_ratio = img.height / img.width if img.width else 1.0
        img_height = img_width * aspect_ratio

        temp_img = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", mode='w+b')
        temp_img_path = temp_img.name
        temp_img.close()
        img.save(temp_img_path)

        rl_image = RLImage(temp_img_path, width=img_width, height=img_height)
        story.append(rl_image)
    except Exception as img_error:
        print(f"Image processing error: {img_error}")
        story.append(Paragraph("Image could not be displayed", body_style))

    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Analysis Results", heading_style))

    if parsed_json:
        if 'summary' in parsed_json:
            story.append(Paragraph("<b>Summary</b>", body_style))
            story.append(Paragraph(parsed_json.get('summary', ''), body_style))
            story.append(Spacer(1, 0.1*inch))

        canonical_pairs = [
            ('diagnosis', 'Diagnosis'),
            ('classification', 'Classification'),
            ('tumor_presence', 'Tumor Presence'),
            ('tumor_type', 'Tumor Type'),
            ('suspected_cancer_type', 'Suspected Cancer Type'),
            ('overall_assessment', 'Overall Assessment'),
            ('breast_density', 'Breast Density'),
            ('birads', 'BI-RADS'),
            ('risk_score', 'Risk Score'),
            ('malignancy_probability', 'Malignancy Probability'),
            ('likelihood_malignancy', 'Likelihood (malignancy)'),
        ]
        for key, label in canonical_pairs:
            if key in parsed_json:
                val = parsed_json[key]
                if isinstance(val, float):
                    display_val = f"{val*100:.1f}%"
                else:
                    display_val = str(val)
                story.append(
                    Paragraph(f"<b>{label}:</b> {display_val}", body_style))
        story.append(Spacer(1, 0.15*inch))

        if 'findings' in parsed_json and isinstance(parsed_json['findings'], list):
            story.append(Paragraph("<b>Findings:</b>", body_style))
            for f in parsed_json['findings']:
                if isinstance(f, dict):
                    parts = []
                    if 'location' in f:
                        parts.append(f"Location: {f['location']}")
                    if 'size_mm' in f:
                        parts.append(f"Size: {f['size_mm']} mm")
                    if 'margins' in f:
                        parts.append(f"Margins: {f['margins']}")
                    if 'density' in f:
                        parts.append(f"Density: {f['density']}")
                    if 'suspicious_features' in f:
                        parts.append(
                            "Suspicious: " + ", ".join(f.get('suspicious_features', [])))
                    story.append(
                        Paragraph(" • " + " | ".join(parts), body_style))
                else:
                    story.append(Paragraph(f" • {str(f)}", body_style))
            story.append(Spacer(1, 0.15*inch))

        for list_key, title in [('malignancy_features', 'Malignant Features'),
                                ('malignant_features', 'Malignant Features'),
                                ('benign_features', 'Benign Features'),
                                ('malignant_keywords', 'Malignant Keywords'),
                                ('benign_keywords', 'Benign Keywords')]:
            if list_key in parsed_json and isinstance(parsed_json[list_key], (list, tuple)):
                vals = parsed_json[list_key]
                story.append(
                    Paragraph(f"<b>{title}:</b> " + ", ".join(map(str, vals)), body_style))
                story.append(Spacer(1, 0.08*inch))

        if 'recommendation' in parsed_json:
            story.append(Paragraph("<b>Recommendation:</b>", body_style))
            story.append(
                Paragraph(str(parsed_json['recommendation']), body_style))
            story.append(Spacer(1, 0.15*inch))
    else:
        response_text = response.strip()
        if len(response_text) > 400:
            story.append(Paragraph(response_text[:800] + "...", body_style))
        else:
            story.append(Paragraph(response_text, body_style))
        story.append(Spacer(1, 0.15*inch))

    if diagnosis_score:
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("AI Assessment Metrics", heading_style))
        scoring_data = [
            ["Metric", "Value"],
            ["Malignancy Risk", f"{diagnosis_score.malignancy_risk*100:.1f}%"],
            ["Confidence Level", diagnosis_score.confidence_level],
            ["Overall Confidence", f"{diagnosis_score.confidence*100:.1f}%"],
            ["Weighted Score", f"{diagnosis_score.weighted_score:.3f}"],
        ]
        scoring_table = Table(scoring_data, colWidths=[3*inch, 2*inch])
        scoring_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(scoring_table)
        story.append(Spacer(1, 0.15*inch))

        story.append(Paragraph("Clinical Indicators", ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=11,
            textColor=colors.HexColor('#2e5c8a'),
            spaceAfter=8
        )))
        indicators_text = f"""
        <b>Malignant Keywords:</b> {'Yes' if diagnosis_score.indicators.get('has_malignant_keywords') else 'No'}<br/>
        <b>Benign Keywords:</b> {'Yes' if diagnosis_score.indicators.get('has_benign_keywords') else 'No'}<br/>
        <b>Biopsy Recommended:</b> {'Yes' if diagnosis_score.indicators.get('mentions_biopsy') else 'No'}<br/>
        <b>Follow-up Needed:</b> {'Yes' if diagnosis_score.indicators.get('mentions_follow_up') else 'No'}<br/>
        <b>Urgent Attention:</b> {'Yes' if diagnosis_score.indicators.get('mentions_urgent') else 'No'}<br/>
        """
        story.append(Paragraph(indicators_text, body_style))
        story.append(Spacer(1, 0.15*inch))

        story.append(Paragraph("AI Reasoning", heading_style))
        story.append(Paragraph(diagnosis_score.reasoning, body_style))

    try:
        doc.build(story)
        print(f"PDF built successfully at: {pdf_path}")
    except Exception as build_error:
        print(f"PDF build error: {build_error}")
        raise

    return pdf_path


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), agent_type: str = Form(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if agent_type == "pathology":
            response, analysis_type, elapsed_time, diagnosis_score = analyze_pathology(
                image)
        elif agent_type == "mammography":
            response, analysis_type, elapsed_time, diagnosis_score = analyze_mammography(
                image)
        elif agent_type == "brain_mri":
            response, analysis_type, elapsed_time, diagnosis_score = analyze_brain_mri(
                image)
        elif agent_type == "skin_cancer":
            response, analysis_type, elapsed_time, diagnosis_score = analyze_skin_cancer(
                image)
        elif agent_type == "lung_cancer":
            response, analysis_type, elapsed_time, diagnosis_score = analyze_lung_cancer(
                image)
        else:
            return {"error": "Invalid agent type"}

        parsed = parse_model_json(response)
        pdf_path = generate_pdf_report(
            analysis_type, response, image_bytes, diagnosis_score, parsed)

        image_data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        scoring_info = {}
        if diagnosis_score:
            scoring_info = {
                "malignancy_risk": f"{diagnosis_score.malignancy_risk*100:.1f}%",
                "confidence_level": diagnosis_score.confidence_level,
                "overall_confidence": f"{diagnosis_score.confidence*100:.1f}%",
                "weighted_score": f"{diagnosis_score.weighted_score:.3f}",
                "reasoning": diagnosis_score.reasoning,
            }
            if parsed:
                scoring_info['parsed'] = parsed

        return {
            "report_date": report_date,
            "analysis_type": analysis_type,
            "response": response,
            "image_data": image_data_url,
            "pdf_path": pdf_path,
            "analysis_time": f"{elapsed_time:.2f}s",
            "scoring": scoring_info
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error: {error_trace}")
        return {"error": str(e), "trace": error_trace}


@app.get("/download-pdf")
async def download_pdf(pdf_path: str):
    try:
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
