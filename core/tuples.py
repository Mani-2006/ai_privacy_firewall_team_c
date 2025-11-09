# core/tuples.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import uuid
import logging

# Get loggers
from .logging_config import loggers
logger = loggers['main']
audit_logger = loggers['audit']


class TimeWindow(BaseModel):
    """Time window for access control with Pydantic validation"""
    
    # Graph-specific fields
    node_id: str = Field(default_factory=lambda: f"tw_{uuid.uuid4().hex[:8]}")
    node_type: str = "TimeWindow"
    
    # Time data
    start: Optional[datetime] = None  # ISO datetime (can be timezone-aware)
    end: Optional[datetime] = None
    
    # Graph metadata
    window_type: str = "access_window"  # "business_hours", "emergency", "access_window"
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        # Pydantic V2 handles datetime serialization automatically
        # No need for deprecated json_encoders
    )
        
    @field_validator('window_type')
    @classmethod
    def validate_window_type(cls, v):
        valid_types = ["business_hours", "emergency", "access_window", "maintenance", "holiday"]
        if v not in valid_types:
            raise ValueError(f'window_type must be one of {valid_types}')
        return v
    
    @model_validator(mode='after')
    def validate_end_after_start(self):
        if self.end is not None and self.start is not None:
            if self.end <= self.start:
                raise ValueError('end time must be after start time')
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO formatted datetimes"""
        logger.debug(f"Converting TimeWindow {self.node_id} to dict")
        return self.model_dump()
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create TimeWindow from dictionary"""
        logger.debug(f"Creating TimeWindow from dict: {d.get('node_id', 'unknown')}")
        return cls(**d)


class TemporalContext(BaseModel):
    """Temporal context for 6-tuple with comprehensive validation"""
    
    # Graph-specific fields
    node_id: str = Field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:8]}")
    node_type: str = "TemporalContext"
    
    # Relationship IDs (references to other graph nodes)
    incident_id: Optional[str] = None
    service_id: Optional[str] = None  
    user_id: Optional[str] = None
    access_window_id: Optional[str] = None  # Reference to TimeWindow node
    
    # Temporal data
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timezone: str = "UTC"
    business_hours: bool = False
    emergency_override: bool = False
    data_freshness_seconds: Optional[int] = Field(None, ge=0)  # Must be >= 0
    situation: Optional[str] = "NORMAL"
    temporal_role: Optional[str] = None
    event_correlation: Optional[str] = None
    
    # For backward compatibility - will be deprecated
    access_window: Optional[TimeWindow] = None
    
    # Graph metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        # Pydantic V2 handles datetime serialization automatically
        # No need for deprecated json_encoders
    )
        
    @field_validator('situation')
    @classmethod
    def validate_situation(cls, v):
        if v is not None:
            valid_situations = ["NORMAL", "EMERGENCY", "MAINTENANCE", "INCIDENT", "AUDIT"]
            if v not in valid_situations:
                raise ValueError(f'situation must be one of {valid_situations}')
        return v
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v):
        # Basic timezone validation - could be enhanced with pytz
        if not v:
            raise ValueError('timezone cannot be empty')
        return v
    
    @field_validator('temporal_role')
    @classmethod
    def validate_temporal_role(cls, v):
        if v is not None:
            valid_roles = ["user", "admin", "system", "emergency_responder", "auditor", 
                          "oncall_low", "oncall_medium", "oncall_high", "oncall_critical"]
            if v not in valid_roles:
                raise ValueError(f'temporal_role must be one of {valid_roles}')
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enhanced logging"""
        logger.debug(f"Converting TemporalContext {self.node_id} to dict")
        audit_logger.info(f"TemporalContext serialized: {self.node_id}, situation={self.situation}, emergency={self.emergency_override}")
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create TemporalContext from dictionary with validation"""
        logger.debug(f"Creating TemporalContext from dict: {d.get('node_id', 'unknown')}")
        
        # Handle access_window nested object
        if d.get("access_window") and isinstance(d["access_window"], dict):
            d["access_window"] = TimeWindow.from_dict(d["access_window"])
        
        try:
            instance = cls(**d)
            audit_logger.info(f"TemporalContext created: {instance.node_id}, situation={instance.situation}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create TemporalContext from dict: {e}")
            raise ValueError(f"Invalid TemporalContext data: {e}")

    def get_graph_properties(self) -> Dict[str, Any]:
        """Get properties suitable for Neo4j node creation"""
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "timezone": self.timezone,
            "business_hours": self.business_hours,
            "emergency_override": self.emergency_override,
            "data_freshness_seconds": self.data_freshness_seconds,
            "situation": self.situation,
            "temporal_role": self.temporal_role,
            "event_correlation": self.event_correlation,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def get_relationships(self) -> Dict[str, str]:
        """Get relationship mappings for graph storage"""
        relationships = {}
        if self.incident_id:
            relationships["RELATES_TO_INCIDENT"] = self.incident_id
        if self.service_id:
            relationships["APPLIES_TO_SERVICE"] = self.service_id
        if self.user_id:
            relationships["GOVERNS_USER"] = self.user_id
        if self.access_window_id:
            relationships["HAS_ACCESS_WINDOW"] = self.access_window_id
        return relationships
    
    def save_to_neo4j(self, neo4j_manager) -> str:
        """
        Save this TemporalContext to Neo4j database
        
        Args:
            neo4j_manager: TemporalNeo4jManager instance
            
        Returns:
            str: Node ID of saved context
        """
        return neo4j_manager.create_temporal_context(self)
    
    def save_to_graphiti(self, graphiti_manager) -> str:
        """
        Save this TemporalContext to Graphiti knowledge graph
        
        Args:
            graphiti_manager: TemporalGraphitiManager instance
            
        Returns:
            str: Entity ID of saved context
        """
        return graphiti_manager.create_temporal_context(self)
    
    @classmethod
    def find_by_service_neo4j(cls, neo4j_manager, service_id: str, limit: int = 10):
        """
        Find temporal contexts for a service from Neo4j
        
        Args:
            neo4j_manager: TemporalNeo4jManager instance
            service_id: Service identifier
            limit: Maximum results to return
            
        Returns:
            List of TemporalContext instances
        """
        results = neo4j_manager.find_temporal_contexts_by_service(service_id, limit)
        contexts = []
        
        for result in results:
            tc_data = result["temporal_context"]
            # Convert Neo4j datetime strings back to datetime objects
            if "timestamp" in tc_data:
                tc_data["timestamp"] = datetime.fromisoformat(tc_data["timestamp"].replace("Z", "+00:00"))
            if "created_at" in tc_data:
                tc_data["created_at"] = datetime.fromisoformat(tc_data["created_at"].replace("Z", "+00:00"))
            if "updated_at" in tc_data:
                tc_data["updated_at"] = datetime.fromisoformat(tc_data["updated_at"].replace("Z", "+00:00"))
            
            contexts.append(cls.from_dict(tc_data))
        
        return contexts
    
    @classmethod
    def find_by_service_graphiti(cls, graphiti_manager, service_id: str, limit: int = 10):
        """
        Find temporal contexts for a service from Graphiti knowledge graph
        
        Args:
            graphiti_manager: TemporalGraphitiManager instance
            service_id: Service identifier
            limit: Maximum results to return
            
        Returns:
            List of TemporalContext instances
        """
        results = graphiti_manager.find_temporal_contexts_by_service(service_id, limit)
        contexts = []
        
        for result in results:
            tc_data = result["temporal_context"]
            # Convert Graphiti data back to TemporalContext
            if "timestamp" in tc_data:
                try:
                    tc_data["timestamp"] = datetime.fromisoformat(tc_data["timestamp"])
                except (ValueError, TypeError):
                    tc_data["timestamp"] = datetime.now(timezone.utc)
            
            contexts.append(cls.from_dict(tc_data))
        
        return contexts


class EnhancedContextualIntegrityTuple(BaseModel):
    """Enhanced 6-tuple with comprehensive validation and audit logging"""
    
    # Core 6-tuple
    data_type: str = Field(..., min_length=1, description="Type of data being accessed")
    data_subject: str = Field(..., min_length=1, description="Subject of the data")
    data_sender: str = Field(..., min_length=1, description="Sender requesting access")
    data_recipient: str = Field(..., min_length=1, description="Recipient of the data")
    transmission_principle: str = Field(..., min_length=1, description="Principle governing transmission")
    temporal_context: TemporalContext = Field(..., description="Temporal context for the request")
    
    # Enhanced attributes for Week 2 assignment
    node_id: str = Field(default_factory=lambda: f"eci_{uuid.uuid4().hex[:8]}")
    node_type: str = "EnhancedContextualIntegrityTuple"
    
    # Data freshness and session tracking
    data_freshness_timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    
    # Audit and compliance flags
    audit_required: bool = False
    compliance_tags: List[str] = Field(default_factory=list)
    risk_level: str = "MEDIUM"
    
    # Processing metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    decision_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Enhanced context attributes
    data_classification: Optional[str] = None
    purpose_limitation: Optional[str] = None
    retention_period: Optional[timedelta] = None
    
    # Cross-system correlation
    correlation_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    related_incident_ids: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        # Pydantic V2 handles datetime and timedelta serialization automatically
        # No need for deprecated json_encoders
    )
        
    @field_validator('risk_level')
    @classmethod
    def validate_risk_level(cls, v):
        valid_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f'risk_level must be one of {valid_levels}')
        return v
    
    @field_validator('data_classification')
    @classmethod
    def validate_data_classification(cls, v):
        if v is not None:
            valid_classifications = ["public", "internal", "confidential", "restricted"]
            if v not in valid_classifications:
                raise ValueError(f'data_classification must be one of {valid_classifications}')
        return v
    
    @field_validator('compliance_tags')
    @classmethod
    def validate_compliance_tags(cls, v):
        if v is not None:
            valid_tags = ["HIPAA", "GDPR", "PCI_DSS", "SOX", "FERPA", "CCPA"]
            for tag in v:
                if tag not in valid_tags:
                    raise ValueError(f'compliance_tag {tag} not valid. Must be one of {valid_tags}')
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with audit logging"""
        logger.debug(f"Converting EnhancedContextualIntegrityTuple {self.node_id} to dict")
        audit_logger.info(f"6-tuple serialized: {self.node_id}, data_type={self.data_type}, risk={self.risk_level}")
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create tuple from dictionary with validation"""
        logger.debug(f"Creating EnhancedContextualIntegrityTuple from dict")
        
        # Handle temporal_context nested object
        if d.get("temporal_context") and isinstance(d["temporal_context"], dict):
            d["temporal_context"] = TemporalContext.from_dict(d["temporal_context"])
        
        try:
            instance = cls(**d)
            audit_logger.info(f"6-tuple created: {instance.node_id}, data_type={instance.data_type}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create EnhancedContextualIntegrityTuple: {e}")
            raise ValueError(f"Invalid tuple data: {e}")
    
    def validate_tuple(self) -> List[str]:
        """Enhanced validation with comprehensive error checking"""
        errors = []
        logger.debug(f"Validating tuple {self.node_id}")
        
        # Data freshness validation
        if self.data_freshness_timestamp:
            age = datetime.now(timezone.utc) - self.data_freshness_timestamp
            if age > timedelta(hours=24):
                errors.append("data_freshness_timestamp is older than 24 hours")
                logger.warning(f"Stale data detected in tuple {self.node_id}: {age}")
        
        # Risk assessment validation
        if self.risk_level == "CRITICAL" and not self.audit_required:
            errors.append("CRITICAL risk level requires audit_required=True")
            
        return errors
    
    def is_valid_tuple(self) -> bool:
        """Check if tuple passes all validation"""
        try:
            self.model_validate(self.model_dump())  # Pydantic validation
            custom_errors = self.validate_tuple()  # Custom validation
            return len(custom_errors) == 0
        except Exception as e:
            logger.error(f"Tuple validation failed: {e}")
            return False
    
    def calculate_risk_score(self) -> float:
        """Calculate comprehensive risk score"""
        base_scores = {
            "LOW": 0.25,
            "MEDIUM": 0.5, 
            "HIGH": 0.75,
            "CRITICAL": 1.0
        }
        
        score = base_scores.get(self.risk_level, 0.5)
        
        # Adjust based on data classification
        if self.data_classification == "restricted":
            score += 0.2
        elif self.data_classification == "confidential":
            score += 0.1
        elif self.data_classification == "public":
            score -= 0.1
        
        # Adjust based on emergency context
        if self.temporal_context.emergency_override:
            score += 0.15
            logger.info(f"Risk score increased due to emergency override: {self.node_id}")
        
        # Adjust based on business hours
        if not self.temporal_context.business_hours:
            score += 0.1
        
        final_score = min(1.0, max(0.0, score))
        logger.debug(f"Risk score calculated for {self.node_id}: {final_score}")
        return final_score
    
    def mark_processed(self, confidence: float = None):
        """Mark tuple as processed with logging"""
        self.processed_at = datetime.now(timezone.utc)
        if confidence is not None:
            self.decision_confidence = confidence
        
        audit_logger.info(f"Tuple processed: {self.node_id}, confidence={confidence}")
        logger.debug(f"Tuple {self.node_id} marked as processed at {self.processed_at}")
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get comprehensive audit summary"""
        summary = {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "data_type": self.data_type,
            "risk_level": self.risk_level,
            "risk_score": self.calculate_risk_score(),
            "compliance_tags": self.compliance_tags,
            "audit_required": self.audit_required,
            "data_classification": self.data_classification,
            "emergency_context": self.temporal_context.emergency_override,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }
        
        audit_logger.info(f"Audit summary generated for {self.node_id}: {summary}")
        return summary
