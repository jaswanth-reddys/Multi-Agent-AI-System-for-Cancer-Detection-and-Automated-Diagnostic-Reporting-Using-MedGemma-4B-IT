
import os
import torch
from PIL import Image
from transformers import pipeline

try:
    from IPython.display import display, Markdown, Image as IPImage
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

LOCAL_MODEL_PATH = r"c:\NEWPROJECT\medgemma-4b-it"


"""## Agent 1 Cell and Tissue Abnormality Detection"""

# Define the path to the image file

image_path = "photos/WhatsApp Image 2025-11-04 at 11.14.54_617dde88.jpg"
image = Image.open(image_path)

# Display the loaded image
image


def load_model():
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    pipe = pipeline(
        "image-text-to-text",
        model=LOCAL_MODEL_PATH,
        model_kwargs=model_kwargs,
    )
    return pipe


pipe = load_model()

pipe.model.generation_config.do_sample = False  # For deterministic output

# Define the prompt with instructions for the AI pathologist
prompt = """
You are a highly specialized AI pathologist. Analyze the histopathology image and classify the type of cells or tissue abnormality present.
If possible, describe if there are signs of cancer or other cellular anomalies.
Respond in a concise medical style suitable for pathology reports.
"""

# Display the defined prompt (optional)

# Create the message structure for the MedGemma chat format (system + user roles)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a medical image classification assistant specialized in pathology."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]
    }
]

# Run the pipeline to generate a response based on the input messages
output = pipe(text=messages, max_new_tokens=300)
response = output[0]["generated_text"][-1]["content"]

# Display the original user prompt
display(Markdown(f"---\n\n**[ User Prompt ]**\n\n{prompt}\n\n---"))

# Display the input image
display(IPImage(filename=image_path, height=300))

# Display the AI-generated response from MedGemma
display(Markdown(f"---\n\n**[ MedGemma Response ]**\n\n{response}\n\n---"))

"""## Agent 2 Breast Cancer"""

# Image path
image_path = "photos/WhatsApp Image 2025-11-04 at 13.25.05_29619055.jpg"
image = Image.open(image_path)

# Display the loaded image
image

# Mammography analysis prompt
prompt = """
You are an expert breast radiologist. Analyze this mammography image (right breast, craniocaudal view - R CC).

Your tasks:

1. Describe the breast density according to BI-RADS classification (A, B, C, D).
2. Identify and describe any suspicious findings (masses, calcifications, distortions).
3. Provide a BI-RADS category (0 to 6) with brief justification.
"""

# Message formatted for MedGemma (system + user roles)
messages = [

    {"role": "system",
     "content": [{"type": "text", "text": "You are a breast cancer detection assistant specialized in mammography interpretation."}]},

    {"role": "user",
     "content": [{"type": "text", "text": prompt},
                 {"type": "image", "image": image}]
     }
]

# Define model parameters: data type and device mapping (automatic CPU/GPU allocation)
model_kwargs = {"torch_dtype": torch.bfloat16,  # Use bfloat16 for optimized memory usage
                # Automatically select available device (CPU or GPU)
                "device_map": "auto",
                }

# Create the Hugging Face pipeline for image-to-text generation using MedGemma
pipe = pipeline("image-text-to-text",  # Task type
                model=LOCAL_MODEL_PATH,  # Model identifier
                model_kwargs=model_kwargs,  # Model configuration parameters
                )

# Disable sampling to ensure deterministic output
pipe.model.generation_config.do_sample = False

# Run the pipeline to generate a response based on the input messages
output = pipe(text=messages, max_new_tokens=300)
response = output[0]["generated_text"][-1]["content"]


pipe = pipeline(
    "image-text-to-text",
    model=LOCAL_MODEL_PATH,
    model_kwargs={
        "torch_dtype": "auto",
        "device_map": "auto",
        "load_in_8bit": True,   # or "load_in_4bit": True
    },
)

# Display the original user prompt
display(Markdown(f"---\n\n**[ User Prompt ]**\n\n{prompt}\n\n---"))

# Display the input image
display(IPImage(filename=image_path, height=500))

# Display the AI-generated response from MedGemma
display(Markdown(f"---\n\n**[ MedGemma Response ]**\n\n{response}\n\n---"))

"""##  - Agent 3 Brain Tumor by Magnetic Resonance Imaging (MRI)"""

# Image path

image_paths = [
    "photos/br1WhatsApp Image 2025-11-08 at 00.29.03_bb7cd591.jpg",
    "photos/br2WhatsApp Image 2025-11-08 at 00.29.22_687ab2a2.jpg",
    "photos/br3WhatsApp Image 2025-11-08 at 00.29.45_f5ca4fa1.jpg",
    "photos/brWhatsApp Image 2025-11-08 at 00.28.32_4becd411.jpg"
]

# Display the loaded images
for path in image_paths:
    image = Image.open(path)
    display(image)

prompt = """
You are an expert neuroradiologist AI assistant.

Given a brain MRI image, perform the following tasks:

1. Detect if there is a brain tumor present.

2. If present, classify the tumor as one of the following:
- Meningioma
- Glioma
- Pituitary Tumor

If no tumor is detected, classify as: No Tumor.

3. Describe the tumor location, size (if visible), mass effect, presence of edema, and enhancement pattern.

4. Provide a structured summary in this format:

Tumor Presence: [Yes/No]
Tumor Type: [Meningioma / Glioma / Pituitary Tumor / No Tumor]
Location: [Text]
Radiological Findings: [Text]
"""

# Model configuration parameters: data type and automatic device mapping (CPU/GPU)
model_kwargs = {
    "torch_dtype": torch.bfloat16,  # Use bfloat16 for better memory efficiency
    "device_map": "auto",  # Automatically select available device (CPU or GPU)
}

# Create the Hugging Face pipeline for image-to-text generation using MedGemma
pipe = pipeline(
    "image-text-to-text",  # Task type
    model=LOCAL_MODEL_PATH,  # Model identifier
    model_kwargs=model_kwargs,  # Model configuration parameters
)

# Disable sampling to ensure deterministic output
pipe.model.generation_config.do_sample = False

# Loop through each image
for path in image_paths:

    # Open the image
    image = Image.open(path)

    # Prepare the message structure for MedGemma (system + user roles)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an AI assistant for brain tumor MRI classification."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    # Run the pipeline and generate a response
    output = pipe(text=messages, max_new_tokens=300)
    response = output[0]["generated_text"][-1]["content"]

    # Display the image name and the AI-generated response
    display(
        Markdown(f"---\n\n**[ Image Analysis: {path.split('/')[-1]} ]**\n\n"))
    display(IPImage(filename=path, height=300))
    display(Markdown(f"**[ MedGemma Response ]**\n\n{response}\n\n---"))

"""##Agent 4 Cancer skin

```
# This is formatted as code
```


"""

# Image path
image_path = "photos/skin_results___29_0.png"
image = Image.open(image_path)

# Display the loaded image
image

# Specialized prompt for skin cancer classification
prompt = """
You are an expert dermatologist AI specialized in dermatoscopic skin lesion analysis.

Tasks for this image:

1. Analyze the dermatoscopic features (asymmetry, border irregularity, color variation, diameter, and dermoscopic structures).

2. Classify the lesion as one of the following:
- Melanoma
- Melanocytic Nevus
- Basal Cell Carcinoma
- Actinic Keratosis / Bowen's Disease
- Benign Keratosis
- Dermatofibroma
- Vascular Lesion

3. Give a brief explanation for your classification based on the image appearance.

Respond as a dermatology report for clinicians.
"""

# Model configuration parameters: data type and automatic device mapping (CPU/GPU)
model_kwargs = {
    "torch_dtype": torch.bfloat16,  # Use bfloat16 for optimized memory usage
    # Automatically select the available device (CPU or GPU)
    "device_map": "auto",
}

# Create the Hugging Face pipeline for image-to-text generation using MedGemma
pipe = pipeline(
    "image-text-to-text",  # Task type
    model=LOCAL_MODEL_PATH,  # Model identifier
    model_kwargs=model_kwargs,  # Model configuration parameters
)

# Disable sampling to ensure deterministic output
pipe.model.generation_config.do_sample = False

# Message structure for MedGemma (system + user roles)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a dermatology diagnostic assistant for skin cancer detection."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}
        ]
    }
]

# Run the pipeline to generate a response based on the input messages
output = pipe(text=messages, max_new_tokens=300)
response = output[0]["generated_text"][-1]["content"]

# Display the original user prompt
display(Markdown(f"---\n\n**[ User Prompt ]**\n\n{prompt}\n\n---"))

# Display the input image
display(IPImage(filename=image_path, height=300))

# Display the AI-generated response from MedGemma
display(Markdown(f"---\n\n**[ MedGemma Response ]**\n\n{response}\n\n---"))
