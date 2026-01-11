# Medical Image Analysis System @

A web-based multi-agent medical image analysis platform using FastAPI and the MedGemma-4B-IT model. The system provides AI-powered diagnostic analysis for pathology, mammography, brain MRI, and skin cancer detection.

## Features

- **4 Specialized Analysis Agents:**
  - 🔬 Pathology: Cell and tissue abnormality detection
  - 🫀 Mammography: Breast cancer detection and BI-RADS classification
  - 🧠 Brain MRI: Brain tumor analysis and classification
  - 🩹 Skin Cancer: Dermatoscopic lesion analysis

- **PDF Report Generation:** Automated reports with analysis results and images
- **Web Interface:** Clean, intuitive UI for image upload and agent selection
- **Local Model:** Uses local MedGemma-4B-IT model for privacy and speed

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the local model is available at:
```
c:\NEWPROJECT\medgemma-4b-it
```

## Running the Application

1. Start the FastAPI server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Select an analysis type
4. Upload a medical image
5. Click "Generate PDF Report"
6. The PDF report will be automatically downloaded

## Application Structure

```
c:\NEWPROJECT\
├── app.py                 # FastAPI backend application
├── copy_of_medgemma_4b_it.py  # Original analysis reference
├── requirements.txt       # Python dependencies
├── medgemma-4b-it/       # Local model directory
└── static/
    └── index.html        # Web interface
```

## API Endpoints

### GET `/`
Returns the web interface (HTML)

### POST `/analyze`
Analyzes an uploaded medical image

**Parameters:**
- `file` (UploadFile): Medical image file
- `agent_type` (str): Type of analysis - `pathology`, `mammography`, `brain_mri`, or `skin_cancer`

**Returns:**
- PDF report with analysis results

## Model Configuration

The application uses the following model settings:
- **Model:** MedGemma-4B-IT (local)
- **Data Type:** bfloat16 (optimized for memory)
- **Device Mapping:** Automatic (CPU/GPU)
- **Sampling:** Disabled (deterministic output)
- **Max Tokens:** 300 per analysis

## Performance Tips

- First model load may take 2-3 minutes
- Subsequent analyses are faster (typically 30-60 seconds)
- For GPU usage, ensure CUDA is properly installed
- For CPU usage, analysis will be slower but still functional

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)

## Notes

- All analyses are performed locally on your system
- PDF reports include the original image and AI analysis
- The model provides clinical insights and should be reviewed by medical professionals
- Maximum recommended image size: 10MB

## Troubleshooting

**Model fails to load:**
- Verify the model path is correct
- Ensure sufficient disk space for model files
- Check CUDA/PyTorch installation for GPU usage

**Image upload fails:**
- Check file format is supported
- Verify file size is under 10MB
- Ensure correct file permissions

**Slow analysis:**
- First run loads the model (takes longer)
- CPU processing is slower than GPU
- Larger images take longer to process

