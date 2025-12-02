#!/usr/bin/env python3
"""
Team A Temporal Integration Models
=================================

Data models that match Team A's temporal framework requirements for proper integration.
Includes EnhancedContextualIntegrityTuple and TemporalContext with all required fields.

Author: Team C Privacy Firewall - Integration Update
Date: 2025-12-02
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import uuid


@dataclass
class TemporalContext:
    """
    Enhanced TemporalContext matching Team A's temporal framework requirements.
    
    Includes all new fields required by Team A's updated policy engine.
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timezone: str = "UTC"
    business_hours: bool = True
    emergency_override: bool = False
    service_id: Optional[str] = None
    temporal_role: Optional[str] = None
    situation: str = "normal"
    data_freshness_seconds: Optional[int] = None
    
    # NEW FIELDS REQUIRED BY TEAM A
    urgency_level: str = "normal"  # Team A requires: critical, high, medium, normal, low
    emergency_authorization_id: Optional[str] = None
    base_role: Optional[str] = None
    inherited_permissions: List[str] = field(default_factory=list)
    temporal_role_valid_until: Optional[datetime] = None


@dataclass
class EnhancedContextualIntegrityTuple:
    """
    Enhanced 6-tuple for contextual integrity matching Team A's requirements.
    
    Includes all new audit and tracking fields expected by Team A's policy engine.
    """
    data_type: str
    data_sender: str
    data_recipient: str
    transmission_principle: str
    temporal_context: TemporalContext
    data_subject: Optional[str] = None
    
    # NEW FIELDS REQUIRED BY TEAM A
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    audit_required: bool = False


class TeamAIntegrationClient:
    """
    Client for making requests to Team A's temporal policy engine.
    
    Formats requests according to Team A's expected format and processes responses.
    """
    
    def __init__(self, team_a_endpoint: str = "http://localhost:8000"):
        self.endpoint = team_a_endpoint
        self.session_id = str(uuid.uuid4())
    
    def create_temporal_tuple(
        self,
        privacy_request: Dict[str, Any],
        emergency: bool = False
    ) -> EnhancedContextualIntegrityTuple:
        """
        Create a properly formatted EnhancedContextualIntegrityTuple for Team A.
        
        Args:
            privacy_request: Team C privacy request data
            emergency: Whether this is an emergency request
            
        Returns:
            EnhancedContextualIntegrityTuple formatted for Team A
        """
        # Create temporal context with Team A's required fields
        temporal_context = TemporalContext(
            timestamp=datetime.now(timezone.utc),
            timezone=privacy_request.get('timezone', 'UTC'),
            business_hours=self._is_business_hours(),
            emergency_override=emergency,
            service_id=privacy_request.get('service_id', 'team_c_privacy_firewall'),
            temporal_role=privacy_request.get('requester_role'),
            situation="EMERGENCY" if emergency else "NORMAL",  # Team A expects uppercase
            data_freshness_seconds=None,
            # Team A REQUIRED: urgency_level field
            urgency_level="critical" if emergency else "normal",
            # New Team A fields
            emergency_authorization_id=privacy_request.get('emergency_auth_id') if emergency else None,
            base_role=privacy_request.get('base_role', privacy_request.get('requester')),
            inherited_permissions=privacy_request.get('inherited_permissions', []),
            temporal_role_valid_until=None
        )
        
        # Create enhanced tuple with Team A's required fields
        return EnhancedContextualIntegrityTuple(
            data_type=self._map_data_type(privacy_request['data_field']),
            data_sender="team_c_privacy_system",
            data_recipient=privacy_request['requester'],
            transmission_principle="access_control",
            temporal_context=temporal_context,
            data_subject=privacy_request.get('data_subject'),
            # New Team A fields
            request_id=str(uuid.uuid4()),
            correlation_id=self.session_id,
            audit_required=self._requires_audit(privacy_request)
        )
    
    def format_request_for_team_a(
        self, 
        tuple_data: EnhancedContextualIntegrityTuple
    ) -> Dict[str, Any]:
        """
        Format the tuple as a JSON request matching Team A's expected format.
        
        Args:
            tuple_data: The EnhancedContextualIntegrityTuple to format
            
        Returns:
            Dict formatted according to Team A's request examples
        """
        return {
            # REQUIRED by Team A
            "request_id": tuple_data.request_id,
            "emergency_authorization_id": tuple_data.temporal_context.emergency_authorization_id,
            
            # Core tuple data
            "data_type": tuple_data.data_type,
            "data_sender": tuple_data.data_sender,
            "data_recipient": tuple_data.data_recipient,
            "transmission_principle": tuple_data.transmission_principle,
            "data_subject": tuple_data.data_subject,
            
            # Temporal context
            "temporal_context": {
                "timestamp": tuple_data.temporal_context.timestamp.isoformat(),
                "timezone": tuple_data.temporal_context.timezone,
                "business_hours": tuple_data.temporal_context.business_hours,
                "emergency_override": tuple_data.temporal_context.emergency_override,
                "service_id": tuple_data.temporal_context.service_id,
                "temporal_role": tuple_data.temporal_context.temporal_role,
                "situation": tuple_data.temporal_context.situation,
                # Team A REQUIRED: urgency_level
                "urgency_level": tuple_data.temporal_context.urgency_level,
                # Team A REQUIRED: access_window structure
                "access_window": {
                    "start": tuple_data.temporal_context.timestamp.isoformat(),
                    "end": (tuple_data.temporal_context.timestamp + timedelta(hours=1)).isoformat()
                },
                "data_freshness_seconds": tuple_data.temporal_context.data_freshness_seconds,
                "emergency_authorization_id": tuple_data.temporal_context.emergency_authorization_id,
                "base_role": tuple_data.temporal_context.base_role,
                "inherited_permissions": tuple_data.temporal_context.inherited_permissions,
                "temporal_role_valid_until": tuple_data.temporal_context.temporal_role_valid_until.isoformat() if tuple_data.temporal_context.temporal_role_valid_until else None
            },
            
            # Audit fields
            "correlation_id": tuple_data.correlation_id,
            "audit_required": tuple_data.audit_required
        }
    
    def parse_team_a_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Team A's response format into Team C's expected format.
        
        Args:
            response: Raw response from Team A's policy engine
            
        Returns:
            Dict in Team C's expected format matching Team A's specification
        """
        # Team A returns these fields per their specification:
        # - decision, decision_id, evaluation_timestamp, confidence
        # - reasoning, policy_rule_matched, emergency_override
        # - urgency_level, time_window_valid, audit_required, cache_ttl_seconds
        
        return {
            # Team C format (updated to match Team A spec)
            "allowed": response.get("decision") == "ALLOW",
            "reason": response.get("reasoning", ""),  # Use "reasoning" not "reasons"
            "confidence": response.get("confidence", 0.0),  # Use "confidence" not "confidence_score"
            
            # Team A metadata we should preserve (per specification)
            "request_id": response.get("request_id"),  # Team A echoes back original request_id
            "decision_id": response.get("decision_id"),
            "evaluation_timestamp": response.get("evaluation_timestamp"),
            "policy_matched": response.get("policy_rule_matched"),
            "emergency_override": response.get("emergency_override", False),
            "urgency_level": response.get("urgency_level", "normal"),
            "time_window_valid": response.get("time_window_valid", True),
            "audit_required": response.get("audit_required", False),
            
            # Integration metadata
            "team_a_integration": True,
            "team_a_compliant": True,
            "original_response": response
        }
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        now = datetime.now(timezone.utc)
        # Simple business hours check (9 AM - 5 PM UTC weekdays)
        return (
            now.weekday() < 5 and  # Monday-Friday
            9 <= now.hour < 17     # 9 AM - 5 PM
        )
    
    def _map_data_type(self, data_field: str) -> str:
        """Map Team C data field to Team A's data type categories."""
        field_lower = data_field.lower()
        
        if any(keyword in field_lower for keyword in ["patient", "medical", "health", "diagnosis"]):
            return "medical_data"
        elif any(keyword in field_lower for keyword in ["salary", "financial", "payment", "billing"]):
            return "financial_data"
        elif any(keyword in field_lower for keyword in ["ssn", "social", "id", "identifier"]):
            return "identification_data"
        else:
            return "personal_data"
    
    def _requires_audit(self, privacy_request: Dict[str, Any]) -> bool:
        """Determine if this request requires audit logging."""
        # Require audit for sensitive data or emergency requests
        return (
            privacy_request.get('emergency', False) or
            'medical' in privacy_request.get('data_field', '').lower() or
            'financial' in privacy_request.get('data_field', '').lower() or
            'ssn' in privacy_request.get('data_field', '').lower()
        )