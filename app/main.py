from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModel, pipeline
import datetime
import os

app = FastAPI(title="MedGenie API")

# Get the absolute path to the credentials file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED_PATH = os.path.join(BASE_DIR, 'firebase_credentials.json')

# Initialize Firebase
cred = credentials.Certificate(CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize ClinicalBERT model
print("Loading ClinicalBERT model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Model loaded successfully!")

class Summary(BaseModel):
    text: str
    patient_id: str
    visit_date: Optional[str] = None

class DiagnosisUpdate(BaseModel):
    document_id: str
    diagnosis: str
    patient_id: str

def clean_text(text: str) -> str:
    """Clean special tokens and unnecessary whitespace"""
    return ' '.join(text.replace('[CLS]', '')
                   .replace('[SEP]', '')
                   .replace('<s>', '')
                   .replace('</s>', '')
                   .split())

def generate_summary(text: str) -> str:
    """Generate summary using BART model"""
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        return clean_text(summary)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return text[:200] + "..."  # Fallback to simple truncation

@app.get("/")
async def read_root():
    """Health check endpoint"""
    try:
        # Test database connection
        db.collection('summaries').limit(1).get()
        return {
            "status": "healthy",
            "message": "MedGenie API is running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database connection error")

@app.post("/summarize")
async def create_summary(summary_request: Summary):
    """Create a new summary for a patient visit"""
    try:
        summary = generate_summary(summary_request.text)
        
        doc_ref = db.collection('summaries').document()
        doc_data = {
            'original_text': summary_request.text,
            'summary': summary,
            'patient_id': summary_request.patient_id,
            'visit_date': summary_request.visit_date or datetime.datetime.now().isoformat(),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'diagnosis': '',
            'status': 'pending_diagnosis'
        }
        doc_ref.set(doc_data)
        
        return {
            "document_id": doc_ref.id,
            "summary": summary,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in create_summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating summary")

@app.post("/update_diagnosis")
async def update_diagnosis(diagnosis_update: DiagnosisUpdate):
    """Update diagnosis for a visit summary"""
    try:
        # First check if document exists
        doc_ref = db.collection('summaries').document(diagnosis_update.document_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found")
            
        doc_ref.update({
            'diagnosis': diagnosis_update.diagnosis,
            'status': 'completed',
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        
        return {
            "status": "success",
            "document_id": diagnosis_update.document_id,
            "message": "Diagnosis updated successfully"
        }
    except Exception as e:
        print(f"Error in update_diagnosis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patient/{patient_id}/history")
async def get_patient_history(patient_id: str):
    """Get complete medical history for a specific patient"""
    try:
        summaries_ref = db.collection('summaries')\
            .where('patient_id', '==', patient_id)\
            .order_by('visit_date', direction=firestore.Query.DESCENDING)
        
        docs = summaries_ref.stream()
        summaries = []
        
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            summaries.append(data)
            
        if not summaries:
            return {
                "status": "success",
                "message": "No records found for this patient",
                "data": []
            }
            
        return {
            "status": "success",
            "data": summaries
        }
    except Exception as e:
        print(f"Error in get_patient_history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)