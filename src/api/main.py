from fastapi import FastAPI, HTTPException
from ..models.schemas import AnalysisRequest, TaskResponse
from datetime import datetime

app = FastAPI(
    title="Product Feedback AI",
    description="Multi-agent system for analyzing product feedback",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Product Feedback AI is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze", response_model=TaskResponse)
async def analyze_product(request: AnalysisRequest):
    """Start product analysis"""
    # TODO: Implement analysis workflow
    return TaskResponse(
        task_id="placeholder",
        status="pending", 
        message=f"Analysis setup for {request.product_name}",
        created_at=datetime.now()
    )

# TODO: Add more endpoints for status, results, etc.
