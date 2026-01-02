from fastapi import FastAPI, Request
from pydantic import BaseModel
import spacy
from utility import PIIUtility
from fastapi.testclient import TestClient
import asyncio

# Load spaCy model once at startup
MODEL_PATH = "./en_core_web_lg-3.8.0"

# Initialize the PIIUtility
handler = PIIUtility(model_path=MODEL_PATH, persist=False)

# FastAPI app
app = FastAPI(title="PII Masking Service", version="1.0")
client = TestClient(app)

# Request body model
class MaskRequest(BaseModel):
    text: str

@app.post("/mask")
async def mask_text(request: MaskRequest):
    """
    Anonymize sensitive PII in the input text.
    Example JSON for request body:
    {
        "text": "Patient John Doe with SSN 123-45-6789 visited on January 15th."
    }
    """
    text = request.text
    masked_text, mapping_id = await handler.mask(text, store_mapping=True)
    return {"masked_text": masked_text, "mapping_id": mapping_id}

@app.post("/unmask")
async def unmask_text(request: MaskRequest):
    """
    De-anonymize text using the mapping ID.
    Example JSON for request body:
    {
        "text": "Anonymized text with placeholders."
    }
    """
    text = request.text
    mapping_id = request.headers.get("Mapping-ID")
    if not mapping_id:
        return {"error": "Mapping-ID header is required."}

    unmasked_text = await handler.unmask(text, mapping_id=mapping_id)
    return {"unmasked_text": unmasked_text}

# health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "PII Masking Service is running"}

# Sample records for testing
sample_records = [
    {"text": "Patient John Doe with SSN 123-45-6789 visited on January 15th."},
    {"text": "Subscriber ID: 9877656. Contact: john.maria@gmail.com, phone +1 (555) 123-4567."},
    {"text": "Member ID: 90867. Pharmacy claim number: 123456789."}
]

# Test the /mask endpoint
for record in sample_records:
    response = client.post("/mask", json=record)
    print("Input:", record["text"])
    print("Response:", response.json())
    print("---")
