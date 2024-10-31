from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the summarization model once to reuse it across requests
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize text function
def summarize_text(text, max_length=10000, min_length=30):
    # Generate a summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the /summarize endpoint
@app.post("/summarize")
async def summarize(input_text):
    try:
        summary = summarize_text(input_text.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
