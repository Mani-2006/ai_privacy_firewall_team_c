"""
Team C API Service - Step 5 Completion
Formal API contracts for Teams A & B integration
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
import asyncio
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.enhanced_graphiti_privacy_bridge import EnhancedGraphitiPrivacyBridge

# Global bridge variable
bridge = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global bridge
    print("🔧 Initializing Privacy Bridge...")
    bridge = EnhancedGraphitiPrivacyBridge()
    yield
    # Shutdown
    if bridge:
        print("🔚 Closing Privacy Bridge...")
        await bridge.close()

app = FastAPI(
    title="AI Privacy Firewall - Team C API",
    version="1.0.0",
    description="Ontology Integration & Semantic Classification Service",
    lifespan=lifespan
)

# Request/Response Models
class ClassificationRequest(BaseModel):
    data_field: str
    context: Optional[str] = None

class PrivacyDecisionRequest(BaseModel):
    requester: str
    data_field: str
    purpose: str
    context: Optional[str] = None
    emergency: bool = False
    # Integration fields for Teams A & B
    temporal_context: Optional[Dict] = None
    org_context: Optional[Dict] = None

class ClassificationResponse(BaseModel):
    field: str
    data_type: str
    sensitivity: str
    context_dependent: bool
    reasoning: List[str]
    confidence: float

class PrivacyDecisionResponse(BaseModel):
    allowed: bool
    reason: str
    confidence: float
    data_classification: Dict
    emergency_used: bool
    integration_ready: bool = True

@app.post("/api/v1/classify", response_model=ClassificationResponse)
async def classify_data_field(request: ClassificationRequest):
    """
    Classify data field using Team C ontology
    Used by Teams A & B for data understanding
    """
    try:
        classification = await bridge.classify_data_field(
            request.data_field, 
            request.context
        )
        
        return ClassificationResponse(
            field=classification["field"] if "field" in classification else request.data_field,
            data_type=classification["data_type"],
            sensitivity=classification["sensitivity"],
            context_dependent=classification.get("context_dependent", False),
            reasoning=classification.get("reasoning", []),
            confidence=0.90  # Default confidence for classification
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/api/v1/privacy-decision", response_model=PrivacyDecisionResponse)
async def make_privacy_decision(request: PrivacyDecisionRequest):
    """
    Make privacy decision with Team A & B integration
    Core endpoint for collaborative decision making
    """
    try:
        # Convert to bridge format
        bridge_request = {
            "requester": request.requester,
            "data_field": request.data_field,
            "purpose": request.purpose,
            "context": request.context,
            "emergency": request.emergency
        }
        
        # Add integration context if provided
        if request.temporal_context:
            bridge_request["temporal_context"] = request.temporal_context
        if request.org_context:
            bridge_request["org_context"] = request.org_context
            
        # Use enhanced Groq-powered privacy decision
        decision = await bridge.make_enhanced_privacy_decision(bridge_request)
        
        return PrivacyDecisionResponse(
            allowed=decision["allowed"],
            reason=decision["reason"],
            confidence=decision["confidence"],
            data_classification=decision["data_classification"],
            emergency_used=decision["emergency_used"],
            integration_ready=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision error: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Team C Privacy Ontology API",
        "version": "1.0.0",
        "integration_ready": True
    }

@app.get("/api/v1/contracts")
async def get_api_contracts():
    """
    Return API contracts for Teams A & B integration
    Documents how other teams should call our service
    """
    return {
        "team_a_integration": {
            "endpoint": "/api/v1/privacy-decision",
            "method": "POST",
            "temporal_context_fields": {
                "current_time": "ISO datetime string",
                "business_hours": "boolean",
                "emergency_time": "boolean",
                "shift_type": "string"
            },
            "example": {
                "requester": "dr_smith",
                "data_field": "patient_record",
                "purpose": "treatment",
                "context": "medical",
                "emergency": True,
                "temporal_context": {
                    "current_time": "2024-01-15T02:30:00Z",
                    "business_hours": False,
                    "emergency_time": True,
                    "shift_type": "night_shift"
                }
            }
        },
        "team_b_integration": {
            "endpoint": "/api/v1/privacy-decision", 
            "method": "POST",
            "org_context_fields": {
                "role": "string",
                "department": "string",
                "clearance_level": "string",
                "employment_type": "string"
            },
            "example": {
                "requester": "manager_sarah",
                "data_field": "team_data",
                "purpose": "review",
                "context": "hr",
                "emergency": False,
                "org_context": {
                    "role": "engineering_manager",
                    "department": "engineering",
                    "clearance_level": "manager",
                    "employment_type": "full_time"
                }
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Team C Privacy API Service...")
    print("📋 API Documentation: http://localhost:8002/docs")
    print("🔧 Fixed deprecation warnings and port conflict")
    uvicorn.run(app, host="0.0.0.0", port=8002)
