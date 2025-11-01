#!/usr/bin/env python3
"""
Timezone Utilities for Privacy Firewall
======================================

Handles timezone-aware timestamps for Graphiti integration, ensuring proper
timestamp formatting for LLM conversion to Neo4j Cypher statements.

Author: Team C Privacy Firewall
Date: 2024-12-30
"""

from datetime import datetime, timezone
import pytz
from typing import Optional

class TimezoneHandler:
    """
    Handles timezone-aware operations for privacy decisions.
    
    Critical for policy enforcement that considers office hours across regions.
    """
    
    # Office timezone mappings for policy enforcement
    OFFICE_TIMEZONES = {
        'california': pytz.timezone('America/Los_Angeles'),
        'india': pytz.timezone('Asia/Kolkata'),
        'utc': pytz.UTC,
        'eastern': pytz.timezone('America/New_York'),
        'london': pytz.timezone('Europe/London')
    }
    
    @classmethod
    def get_current_utc(cls) -> datetime:
        """Get current time in UTC with timezone info."""
        return datetime.now(timezone.utc)
    
    @classmethod
    def get_current_in_timezone(cls, office_location: str) -> datetime:
        """Get current time in specified office timezone."""
        if office_location.lower() not in cls.OFFICE_TIMEZONES:
            raise ValueError(f"Unknown office location: {office_location}")
        
        tz = cls.OFFICE_TIMEZONES[office_location.lower()]
        return datetime.now(tz)
    
    @classmethod
    def format_for_graphiti(cls, dt: datetime, include_requester_location: Optional[str] = None) -> str:
        """
        Format datetime for Graphiti LLM processing.
        
        Follows the pattern: "Person (2024-07-30T00:01:00Z): message"
        Ensures timezone info is preserved for Cypher translation.
        """
        # Ensure datetime is timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to UTC for consistent storage
        utc_time = dt.astimezone(timezone.utc)
        iso_timestamp = utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        if include_requester_location:
            local_time = dt.astimezone(cls.OFFICE_TIMEZONES.get(
                include_requester_location.lower(), timezone.utc
            ))
            local_timestamp = local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            return f"{iso_timestamp} (Local: {local_timestamp})"
        
        return iso_timestamp
    
    @classmethod
    def create_privacy_episode_content(cls, 
                                     requester: str,
                                     data_field: str,
                                     decision: dict,
                                     timestamp: Optional[datetime] = None,
                                     requester_location: Optional[str] = None) -> str:
        """
        Create Graphiti episode content with proper timestamp formatting.
        
        This follows the conversation pattern that Graphiti expects for LLM processing.
        """
        if timestamp is None:
            timestamp = cls.get_current_utc()
        
        formatted_time = cls.format_for_graphiti(timestamp, requester_location)
        
        # Create content in conversational format for Graphiti LLM
        episode_content = f"""Privacy Decision Episode - {formatted_time}

Requester: {requester}
Data Field: {data_field}
Decision: {'ALLOWED' if decision.get('allowed', False) else 'DENIED'}
Reason: {decision.get('reason', 'No reason provided')}
Confidence: {decision.get('confidence', 0.0)}
Context: {decision.get('context', 'General request')}
Emergency: {'Yes' if decision.get('emergency', False) else 'No'}

Policy Enforcement Notes:
- Timestamp preserved for temporal policy evaluation
- Office hours consideration: {requester_location if requester_location else 'UTC standard'}
- Decision made at: {formatted_time}
"""
        return episode_content
    
    @classmethod
    def create_data_classification_content(cls,
                                         data_field: str,
                                         classification: dict,
                                         timestamp: Optional[datetime] = None) -> str:
        """
        Create Graphiti entity content for data classification.
        """
        if timestamp is None:
            timestamp = cls.get_current_utc()
            
        formatted_time = cls.format_for_graphiti(timestamp)
        
        content = f"""Data Classification - {formatted_time}

Data Field: {data_field}
Data Type: {classification.get('data_type', 'Unknown')}
Sensitivity Level: {classification.get('sensitivity_level', 'Unknown')}
Classification Confidence: {classification.get('confidence', 0.0)}
PII Status: {'Contains PII' if classification.get('is_pii', False) else 'No PII detected'}

Classification Details:
- Classified at: {formatted_time}
- Reasoning: {classification.get('reasoning', 'Automated classification')}
"""
        return content
    
    @classmethod
    def is_office_hours(cls, location: str, check_time: Optional[datetime] = None) -> bool:
        """
        Check if current time is within office hours for given location.
        
        Critical for privacy policies that consider business hours.
        """
        if check_time is None:
            check_time = cls.get_current_utc()
        
        if location.lower() not in cls.OFFICE_TIMEZONES:
            return True  # Default to allowing if timezone unknown
        
        tz = cls.OFFICE_TIMEZONES[location.lower()]
        local_time = check_time.astimezone(tz)
        
        # Define office hours (9 AM to 6 PM local time)
        hour = local_time.hour
        weekday = local_time.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend check
        if weekday >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Business hours check
        return 9 <= hour < 18
    
    @classmethod
    def get_business_context(cls, location: str, check_time: Optional[datetime] = None) -> str:
        """
        Get business context for privacy decisions.
        """
        if check_time is None:
            check_time = cls.get_current_utc()
        
        is_business_hours = cls.is_office_hours(location, check_time)
        local_time = check_time.astimezone(
            cls.OFFICE_TIMEZONES.get(location.lower(), timezone.utc)
        )
        
        return f"Business Hours: {'Yes' if is_business_hours else 'No'} " \
               f"(Local time: {local_time.strftime('%H:%M %Z')})"