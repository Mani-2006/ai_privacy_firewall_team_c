#!/usr/bin/env python3
"""
Enhanced Privacy API Service with Team A Temporal Integration
============================================================

This enhanced API service integrates Team A's temporal framework with Team C's
privacy firewall, providing comprehensive privacy decisions with temporal context.

New Features in this integration:
- Enhanced endpoints with temporal context support
- Emergency override capabilities from Team A
- 6-tuple contextual integrity framework
- Time-aware access control decisions
- Combined audit logging and Neo4j storage

API Endpoints:
- GET /health - Health check
- GET /contracts - API contracts for Teams A & B
- POST /classify - Data classification with temporal context
- POST /privacy-decision - Enhanced privacy decision with temporal analysis
- POST /temporal-privacy-decision - New integrated endpoint
- POST /emergency-override - Emergency access evaluation

Author: Team C + Team A Integration
Date: 2024-12-30
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager

# FastAPI and Pydantic imports
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Team C privacy components
from ontology.privacy_ontology import AIPrivacyOntology
from integration.enhanced_graphiti_privacy_bridge import EnhancedGraphitiPrivacyBridge
from docs.system_architecture import TeamCArchitecture

# Add Team A's temporal framework to path
TEAM_A_PATH = Path(__file__).parent.parent / "ai_temporal_framework"
if TEAM_A_PATH.exists():
    sys.path.insert(0, str(TEAM_A_PATH))
    TEAM_A_AVAILABLE = True
else:
    TEAM_A_AVAILABLE = False
    print("⚠️  Team A temporal framework not found - running in privacy-only mode")

# Team A temporal imports (conditional)
if TEAM_A_AVAILABLE:
    try:
        from core.tuples import EnhancedContextualIntegrityTuple, TemporalContext, TimeWindow
        from core.policy_engine import TemporalPolicyEngine
        from core.graphiti_manager import TemporalGraphitiManager, GraphitiConfig
        print("✅ Team A temporal framework loaded successfully")
    except ImportError as e:
        print(f"⚠️  Team A components not fully available: {e}")
        TEAM_A_AVAILABLE = False

# Global components
privacy_ontology = None
privacy_bridge = None
temporal_manager = None
temporal_policy_engine = None
architecture = TeamCArchitecture()

# Enhanced Pydantic Models for Team A + C Integration

class TemporalContextModel(BaseModel):
    """Temporal context for Team A integration."""
    situation: str = Field(..., description="Current situational context")
    urgency_level: str = Field(default="normal", description="Urgency level: low, normal, high, critical")
    data_type: Optional[str] = Field(default=None, description="Type of data being accessed")
    transmission_principle: Optional[str] = Field(default="secure", description="How data should be transmitted")
    time_window_start: Optional[datetime] = Field(default=None, description="Start of access time window")
    time_window_end: Optional[datetime] = Field(default=None, description="End of access time window")
    emergency_override_requested: bool = Field(default=False, description="Whether emergency override is requested")

class EnhancedClassificationRequest(BaseModel):
    """Enhanced classification request with temporal context."""
    data_field: str = Field(..., description="Data field to classify")
    context: str = Field(..., description="Context in which data is being accessed")
    temporal_context: Optional[TemporalContextModel] = Field(default=None, description="Temporal context from Team A")

class EnhancedPrivacyDecisionRequest(BaseModel):
    """Enhanced privacy decision request with temporal integration."""
    data_field: str = Field(..., description="Data field requiring access decision")
    requester_role: str = Field(..., description="Role of the requester")
    context: str = Field(..., description="Current context")
    organizational_context: Optional[str] = Field(default=None, description="Additional organizational context")
    temporal_context: Optional[TemporalContextModel] = Field(default=None, description="Temporal context from Team A")

class TemporalPrivacyDecisionRequest(BaseModel):
    """Dedicated request model for integrated temporal + privacy decisions."""
    data_field: str = Field(..., description="Data field requiring access decision")
    requester_role: str = Field(..., description="Role of the requester")
    context: str = Field(..., description="Current context")
    temporal_context: TemporalContextModel = Field(..., description="Required temporal context")
    organizational_context: Optional[str] = Field(default=None, description="Additional organizational context")
    force_temporal_evaluation: bool = Field(default=True, description="Force temporal framework evaluation")

class EmergencyOverrideRequest(BaseModel):
    """Emergency override request using Team A's capabilities."""
    data_field: str = Field(..., description="Critical data field requiring emergency access")
    requester_role: str = Field(..., description="Role requesting emergency access")
    emergency_situation: str = Field(..., description="Description of emergency situation")
    justification: str = Field(..., description="Justification for emergency override")
    expected_duration_minutes: int = Field(default=60, description="Expected duration of emergency access")

class IntegratedDecisionResponse(BaseModel):
    """Response model for integrated Team A + C decisions."""
    decision: str = Field(..., description="Final access decision: ALLOW or DENY")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    privacy_component: Dict[str, Any] = Field(..., description="Team C privacy analysis")
    temporal_component: Optional[Dict[str, Any]] = Field(default=None, description="Team A temporal analysis")
    integration_method: str = Field(..., description="How the decision was integrated")
    emergency_override_used: bool = Field(default=False, description="Whether emergency override was applied")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audit_trail: List[str] = Field(default_factory=list, description="Audit trail of decision process")

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application components."""
    global privacy_ontology, privacy_bridge, temporal_manager, temporal_policy_engine
    
    print("🚀 Starting Enhanced Privacy API Service (Team A + C Integration)")
    
    # Initialize Team C privacy components
    try:
        privacy_ontology = AIPrivacyOntology()
        privacy_bridge = EnhancedGraphitiPrivacyBridge()
        print("✅ Team C privacy components initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Team C components: {e}")
        raise
    
    # Initialize Team A temporal components (if available)
    if TEAM_A_AVAILABLE:
        try:
            config = GraphitiConfig(
                neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.getenv("NEO4J_USER", "llm_security"),
                neo4j_password=os.getenv("NEO4J_PASSWORD"),
                team_namespace="llm_security"
            )
            
            # Set the correct path to Team A's rules file
            team_a_rules_path = "../ai_temporal_framework/mocks/rules.yaml"
            
            temporal_manager = TemporalGraphitiManager(config)
            temporal_policy_engine = TemporalPolicyEngine(
                config_file=team_a_rules_path, 
                graphiti_manager=temporal_manager
            )
            print("✅ Team A temporal components initialized")
        except Exception as e:
            print(f"⚠️  Team A temporal components failed to initialize: {e}")
            print("   Continuing with Team C privacy-only mode...")
    
    print("🌟 Enhanced API service ready!")
    
    yield
    
    print("🔄 Shutting down Enhanced Privacy API Service")

# Create FastAPI app with enhanced integration
app = FastAPI(
    title="Enhanced Privacy API Service (Team A + C Integration)",
    description="Integrated privacy firewall with temporal context awareness",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced API Endpoints

@app.get("/health")
async def health_check():
    """Enhanced health check including Team A temporal framework status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc),
        "service": "Enhanced Privacy API (Team A + C)",
        "version": "2.0.0",
        "components": {
            "team_c_privacy": privacy_ontology is not None,
            "team_c_bridge": privacy_bridge is not None,
            "team_a_temporal": TEAM_A_AVAILABLE and temporal_manager is not None,
            "team_a_policy_engine": temporal_policy_engine is not None,
            "integration_mode": "full" if TEAM_A_AVAILABLE else "privacy_only"
        }
    }

@app.get("/contracts")
async def get_api_contracts():
    """Get enhanced API contracts including temporal integration."""
    base_contracts = architecture.INTEGRATION_CONTRACTS
    
    # Add Team A temporal integration contracts
    enhanced_contracts = {
        **base_contracts,
        "team_a_temporal_integration": {
            "description": "Team A temporal framework integration endpoints",
            "endpoints": {
                "temporal_privacy_decision": {
                    "method": "POST",
                    "path": "/temporal-privacy-decision",
                    "description": "Integrated temporal + privacy decision",
                    "requires": ["temporal_context"]
                },
                "emergency_override": {
                    "method": "POST", 
                    "path": "/emergency-override",
                    "description": "Emergency access evaluation with temporal override",
                    "requires": ["emergency_situation", "justification"]
                }
            },
            "temporal_context_fields": {
                "situation": "Current situational context",
                "urgency_level": "low|normal|high|critical",
                "emergency_override_requested": "boolean",
                "time_window_start": "ISO datetime",
                "time_window_end": "ISO datetime"
            }
        }
    }
    
    return enhanced_contracts

@app.post("/classify", response_model=Dict[str, Any])
async def classify_data_field(request: EnhancedClassificationRequest):
    """Enhanced data classification with optional temporal context."""
    try:
        # Base Team C classification
        classification = privacy_ontology.classify_data_field(
            field_name=request.data_field,
            context=request.context
        )
        
        result = {
            "data_field": request.data_field,
            "classification": classification,
            "context": request.context,
            "timestamp": datetime.now(timezone.utc),
            "team_c_analysis": True
        }
        
        # Add temporal classification if available
        if request.temporal_context and TEAM_A_AVAILABLE and temporal_policy_engine:
            try:
                temporal_classification = _get_temporal_classification(
                    request.data_field,
                    request.context,
                    request.temporal_context
                )
                result["temporal_analysis"] = temporal_classification
                result["integration_mode"] = "full"
            except Exception as e:
                result["temporal_error"] = str(e)
                result["integration_mode"] = "privacy_only"
        else:
            result["integration_mode"] = "privacy_only"
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )

@app.post("/privacy-decision", response_model=IntegratedDecisionResponse)
async def enhanced_privacy_decision(request: EnhancedPrivacyDecisionRequest):
    """Enhanced privacy decision with optional temporal integration."""
    try:
        # Team C privacy decision
        privacy_decision_raw = privacy_ontology.make_privacy_decision(
            requester=request.requester_role,
            data_field=request.data_field,
            purpose=request.context,
            context=request.organizational_context,
            emergency=False
        )
        
        # Convert Team C format to standard format
        privacy_decision = {
            "decision": "ALLOW" if privacy_decision_raw.get("allowed", False) else "DENY",
            "confidence": privacy_decision_raw.get("confidence", 0.5),
            "reasoning": privacy_decision_raw.get("reason", "No reasoning provided"),
            "raw_response": privacy_decision_raw
        }
        
        # Create audit trail
        audit_trail = [
            f"Team C privacy decision: {privacy_decision['decision']}",
            f"Privacy confidence: {privacy_decision['confidence']:.2f}"
        ]
        
        # Team A temporal analysis (if available and requested)
        temporal_analysis = None
        integration_method = "privacy_only"
        
        if request.temporal_context and TEAM_A_AVAILABLE and temporal_policy_engine:
            try:
                temporal_analysis = _analyze_temporal_context(
                    request.data_field,
                    request.requester_role,
                    request.context,
                    request.temporal_context
                )
                audit_trail.append(f"Team A temporal analysis: {temporal_analysis['decision']}")
                integration_method = "integrated"
            except Exception as e:
                audit_trail.append(f"Temporal analysis failed: {e}")
        
        # Combine decisions
        final_decision = _combine_privacy_temporal_decisions(
            privacy_decision, temporal_analysis
        )
        audit_trail.append(f"Final integrated decision: {final_decision['decision']}")
        
        # Store decision
        privacy_bridge.create_privacy_decision_episode(
            requester_role=request.requester_role,
            data_entity=request.data_field,
            context=request.context,
            decision=final_decision["decision"],
            reasoning=final_decision["reasoning"],
            confidence=final_decision["confidence"]
        )
        
        return IntegratedDecisionResponse(
            decision=final_decision["decision"],
            confidence=final_decision["confidence"],
            reasoning=final_decision["reasoning"],
            privacy_component=privacy_decision,
            temporal_component=temporal_analysis,
            integration_method=integration_method,
            emergency_override_used=final_decision.get("emergency_override", False),
            audit_trail=audit_trail
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Privacy decision failed: {str(e)}"
        )

@app.post("/temporal-privacy-decision", response_model=IntegratedDecisionResponse)
async def temporal_privacy_decision(request: TemporalPrivacyDecisionRequest):
    """Dedicated endpoint for integrated temporal + privacy decisions."""
    if not TEAM_A_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Team A temporal framework not available"
        )
    
    try:
        # Enhanced privacy decision request
        enhanced_request = EnhancedPrivacyDecisionRequest(
            data_field=request.data_field,
            requester_role=request.requester_role,
            context=request.context,
            organizational_context=request.organizational_context,
            temporal_context=request.temporal_context
        )
        
        # Use enhanced privacy decision with forced temporal evaluation
        return await enhanced_privacy_decision(enhanced_request)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal privacy decision failed: {str(e)}"
        )

@app.post("/emergency-override", response_model=IntegratedDecisionResponse)
async def emergency_override_evaluation(request: EmergencyOverrideRequest):
    """Emergency override evaluation using Team A's temporal capabilities."""
    if not TEAM_A_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Team A temporal framework required for emergency override"
        )
    
    try:
        # Create emergency temporal context
        emergency_temporal_context = TemporalContextModel(
            situation=request.emergency_situation,
            urgency_level="critical",
            data_type="emergency_access",
            transmission_principle="emergency_override",
            emergency_override_requested=True,
            time_window_start=datetime.now(timezone.utc),
            time_window_end=datetime.now(timezone.utc) + timedelta(minutes=request.expected_duration_minutes)
        )
        
        # Create emergency decision request
        emergency_request = TemporalPrivacyDecisionRequest(
            data_field=request.data_field,
            requester_role=request.requester_role,
            context=request.emergency_situation,
            temporal_context=emergency_temporal_context,
            organizational_context=f"Emergency: {request.justification}",
            force_temporal_evaluation=True
        )
        
        # Process as temporal privacy decision
        result = await temporal_privacy_decision(emergency_request)
        
        # Add emergency-specific audit information
        result.audit_trail.extend([
            f"Emergency override requested: {request.justification}",
            f"Expected duration: {request.expected_duration_minutes} minutes"
        ])
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency override evaluation failed: {str(e)}"
        )

# Helper functions for Team A integration

def _get_temporal_classification(data_field: str, context: str, temporal_context: TemporalContextModel) -> Dict[str, Any]:
    """Get temporal classification from Team A framework."""
    try:
        # Create enhanced tuple for classification
        enhanced_tuple = EnhancedContextualIntegrityTuple(
            data_subject=data_field,
            data_sender="system",
            data_recipient="classifier",
            data_type=temporal_context.data_type or "unknown",
            transmission_principle=temporal_context.transmission_principle or "secure",
            temporal_context=TemporalContext(
                situation=temporal_context.situation,
                urgency_level=temporal_context.urgency_level
            )
        )
        
        # Use temporal policy engine for classification
        classification_result = temporal_policy_engine.classify_temporal_context(enhanced_tuple)
        
        return {
            "temporal_classification": classification_result.get("classification", "unknown"),
            "urgency_assessment": temporal_context.urgency_level,
            "emergency_eligible": temporal_context.emergency_override_requested,
            "confidence": classification_result.get("confidence", 0.5)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "temporal_classification": "error",
            "confidence": 0.0
        }

def _analyze_temporal_context(
    data_field: str,
    requester_role: str,
    context: str,
    temporal_context: TemporalContextModel
) -> Dict[str, Any]:
    """Analyze temporal context using Team A framework."""
    try:
        # Create time window
        time_window = TimeWindow(
            start=temporal_context.time_window_start or datetime.now(timezone.utc),
            end=temporal_context.time_window_end or datetime.now(timezone.utc) + timedelta(hours=1),
            window_type="emergency" if temporal_context.urgency_level == "critical" else "access_window"
        )
        
        # Create enhanced tuple
        enhanced_tuple = EnhancedContextualIntegrityTuple(
            data_subject=data_field,
            data_sender="system",
            data_recipient=requester_role,
            data_type=temporal_context.data_type or "sensitive",
            transmission_principle=temporal_context.transmission_principle or "secure",
            temporal_context=TemporalContext(
                situation=temporal_context.situation,
                urgency_level=temporal_context.urgency_level,
                time_window=time_window
            )
        )
        
        # Evaluate using temporal policy engine
        temporal_result = temporal_policy_engine.evaluate_temporal_access(enhanced_tuple)
        
        return {
            "decision": temporal_result.get("decision", "DENY"),
            "confidence": temporal_result.get("confidence", 0.5),
            "reasoning": temporal_result.get("reasoning", "Temporal evaluation"),
            "emergency_override": temporal_context.emergency_override_requested,
            "urgency_level": temporal_context.urgency_level,
            "time_window_valid": temporal_result.get("time_window_valid", True)
        }
        
    except Exception as e:
        return {
            "decision": "DENY",
            "confidence": 0.0,
            "reasoning": f"Temporal analysis error: {e}",
            "emergency_override": False,
            "urgency_level": "unknown",
            "time_window_valid": False
        }

def _combine_privacy_temporal_decisions(
    privacy_decision: Dict[str, Any],
    temporal_analysis: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Combine Team A temporal + Team C privacy decisions."""
    if not temporal_analysis:
        return {
            "decision": privacy_decision["decision"],
            "confidence": privacy_decision["confidence"],
            "reasoning": f"Privacy only: {privacy_decision['reasoning']}",
            "emergency_override": False
        }
    
    privacy_allow = privacy_decision["decision"] == "ALLOW"
    temporal_allow = temporal_analysis["decision"] == "ALLOW"
    emergency_override = temporal_analysis.get("emergency_override", False)
    
    if emergency_override and not privacy_allow:
        # Emergency can override privacy denial
        return {
            "decision": "ALLOW",
            "confidence": max(0.7, temporal_analysis["confidence"]),
            "reasoning": f"Emergency override: {temporal_analysis['reasoning']}",
            "emergency_override": True
        }
    elif privacy_allow and temporal_allow:
        # Both agree to allow
        return {
            "decision": "ALLOW",
            "confidence": min(1.0, (privacy_decision["confidence"] + temporal_analysis["confidence"]) / 2 + 0.1),
            "reasoning": f"Consensus allow: {privacy_decision['reasoning']}",
            "emergency_override": False
        }
    else:
        # Default to deny for security
        return {
            "decision": "DENY",
            "confidence": max(privacy_decision["confidence"], temporal_analysis.get("confidence", 0.5)),
            "reasoning": f"Security priority: {privacy_decision['reasoning']} | {temporal_analysis['reasoning']}",
            "emergency_override": False
        }

if __name__ == "__main__":
    print("🚀 Starting Enhanced Privacy API Service (Team A + C Integration)")
    print(f"Team A Temporal Framework: {'✅ Available' if TEAM_A_AVAILABLE else '❌ Not Available'}")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")