#!/usr/bin/env python3
"""
Mission Phase Policy Interface with Timezone-Aware User Tracking

This interface combines mission phase policy with timezone-aware user tracking:
- User identification with email and name
- Location-based timezone detection
- Mission phase-specific access control
- Role-based permissions
- Connection tracking and logging
- Emergency override capabilities

The interface tracks who tries to connect with complete timezone awareness.
"""

import os
import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import pytz

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.nodes import EpisodeType


def load_interface_config(config_file: str = "mission_phase_interface_config.json") -> Dict[str, Any]:
    """Load interface configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default configuration")
        return {
            "users": {
                "user123": {
                    "name": "John Smith",
                    "email": "john.smith@company.com",
                    "location": "New York, NY, USA",
                    "timezone": "America/New_York",
                    "role": "analyst",
                    "clearance_level": "confidential"
                }
            },
            "mission_phases": {
                "pre_deployment": {
                    "description": "Pre-deployment phase with strict controls",
                    "timezone_aware": True,
                    "working_hours_only": True
                }
            },
            "connection_tracking": {
                "enabled": True,
                "log_all_attempts": True,
                "track_timezone": True
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_file}: {e}")
        return {}


def setup_logging() -> logging.Logger:
    """Setup logging configuration with timestamps and unique log file per run"""
    logger = logging.getLogger('mission_phase_interface')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'mission_phase_interface_{now}.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class ConnectionTracker:
    """Tracks connection attempts with timezone awareness"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config.get("connection_tracking", {})
        self.logger = logger or logging.getLogger('connection_tracker')
        self.connection_log = []
        self.suspicious_attempts = []
        
    def log_connection_attempt(self, user_info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log a connection attempt with full details"""
        try:
            if not self.config.get("enabled", True):
                return
            
            connection_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_info.get("id"),
                "user_name": user_info.get("name"),
                "user_email": user_info.get("email"),
                "location": user_info.get("location"),
                "timezone": user_info.get("timezone"),
                "local_time": result.get("local_time"),
                "time_period": result.get("time_period"),
                "mission_phase": result.get("mission_phase"),
                "data_classification": result.get("data_classification"),
                "access_decision": "allowed" if result.get("allowed") else "blocked",
                "reason": result.get("reason"),
                "query": result.get("query"),
                "user_role": user_info.get("role"),
                "user_clearance": user_info.get("clearance_level")
            }
            
            self.connection_log.append(connection_record)
            
            # Check for suspicious patterns
            if self.config.get("alert_on_suspicious", True):
                self._check_suspicious_patterns(connection_record)
            
            self.logger.info(f"Connection logged: {user_info.get('name')} ({user_info.get('email')}) - {connection_record['access_decision']}")
            
        except Exception as e:
            self.logger.error(f"Error logging connection attempt: {str(e)}")
    
    def _check_suspicious_patterns(self, connection_record: Dict[str, Any]) -> None:
        """Check for suspicious access patterns"""
        try:
            suspicious_patterns = self.config.get("suspicious_patterns", [])
            
            # Check for off-hours access attempts
            if "off_hours_access_attempts" in suspicious_patterns:
                if (connection_record.get("time_period") in ["off_hours", "weekend"] and 
                    connection_record.get("access_decision") == "blocked"):
                    self._flag_suspicious(connection_record, "Off-hours access attempt")
            
            # Check for weekend access attempts
            if "weekend_access_attempts" in suspicious_patterns:
                if (connection_record.get("time_period") == "weekend" and 
                    connection_record.get("access_decision") == "blocked"):
                    self._flag_suspicious(connection_record, "Weekend access attempt")
            
            # Check for unauthorized role attempts
            if "unauthorized_role_attempts" in suspicious_patterns:
                if connection_record.get("access_decision") == "blocked" and "not authorized" in connection_record.get("reason", "").lower():
                    self._flag_suspicious(connection_record, "Unauthorized role attempt")
            
        except Exception as e:
            self.logger.error(f"Error checking suspicious patterns: {str(e)}")
    
    def _flag_suspicious(self, connection_record: Dict[str, Any], reason: str) -> None:
        """Flag a connection as suspicious"""
        suspicious_record = {
            **connection_record,
            "suspicious_reason": reason,
            "flagged_at": datetime.now(timezone.utc).isoformat()
        }
        self.suspicious_attempts.append(suspicious_record)
        self.logger.warning(f"SUSPICIOUS ACCESS: {connection_record.get('user_name')} ({connection_record.get('user_email')}) - {reason}")
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get a summary of connection attempts"""
        try:
            total_attempts = len(self.connection_log)
            allowed_attempts = len([c for c in self.connection_log if c.get("access_decision") == "allowed"])
            blocked_attempts = len([c for c in self.connection_log if c.get("access_decision") == "blocked"])
            suspicious_count = len(self.suspicious_attempts)
            
            # Group by user
            user_attempts = {}
            for connection in self.connection_log:
                user_id = connection.get("user_id")
                if user_id not in user_attempts:
                    user_attempts[user_id] = {
                        "name": connection.get("user_name"),
                        "email": connection.get("user_email"),
                        "total_attempts": 0,
                        "allowed_attempts": 0,
                        "blocked_attempts": 0
                    }
                user_attempts[user_id]["total_attempts"] += 1
                if connection.get("access_decision") == "allowed":
                    user_attempts[user_id]["allowed_attempts"] += 1
                else:
                    user_attempts[user_id]["blocked_attempts"] += 1
            
            return {
                "total_attempts": total_attempts,
                "allowed_attempts": allowed_attempts,
                "blocked_attempts": blocked_attempts,
                "suspicious_attempts": suspicious_count,
                "user_breakdown": user_attempts,
                "suspicious_details": self.suspicious_attempts
            }
            
        except Exception as e:
            self.logger.error(f"Error generating connection summary: {str(e)}")
            return {}


class MissionPhaseInterface:
    """Combined mission phase policy with timezone-aware user tracking"""
    
    def __init__(self, graphiti: Graphiti, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.graphiti = graphiti
        self.config = config
        self.policies_added = False
        self.logger = logger or logging.getLogger('mission_phase_interface')
        
        # Initialize connection tracker
        self.connection_tracker = ConnectionTracker(config, logger)
        
        # Build lookup dictionaries for fast access
        self.users = config.get("users", {})
        self.location_timezone_mapping = config.get("location_timezone_mapping", {})
        self.mission_phases = config.get("mission_phases", {})
        self.data_classifications = config.get("data_classifications", {})
        self.timezone_policies = config.get("timezone_policies", {})
        
        # Build data classification lookup by examples
        self.classification_lookup = {}
        for class_name, class_config in self.data_classifications.items():
            for example in class_config.get("examples", []):
                self.classification_lookup[example.lower()] = class_name
    
    async def add_policies(self):
        """Add mission phase interface policies to the knowledge graph"""
        try:
            if self.policies_added:
                self.logger.info("Mission phase interface policies already added to graph - skipping")
                return
            
            self.logger.info("Starting to add mission phase interface policies to knowledge graph...")
            
            # Load policies from JSON configuration and filter out problematic ones
            all_policies = self.config.get("policy_rules", [
                "Include user email and name in all policy decisions",
                "Track all connection attempts with timezone information",
                "Apply mission phase-specific restrictions"
            ])
            
            # Filter out policies that might cause Neo4j property issues
            policies = []
            for policy in all_policies:
                # Skip policies that contain complex structures or might be parsed as objects
                if any(keyword in policy.lower() for keyword in ['timestamp', 'timezone', 'status', 'pattern', 'preservation', 'restriction']):
                    self.logger.warning(f"Skipping potentially problematic policy: {policy[:50]}...")
                    continue
                policies.append(policy)
            
            self.logger.debug(f"Preparing to add {len(policies)} text policies to graph")
            
            # Add text-based policies with minimal entity extraction
            for i, policy in enumerate(policies):
                try:
                    self.logger.debug(f"Adding text policy {i+1}/{len(policies)}")
                    
                    # Check for non-ASCII characters
                    non_ascii_chars = [c for c in policy if ord(c) > 127]
                    if non_ascii_chars:
                        self.logger.warning(f"Non-ASCII characters found in text policy {i+1}: {[repr(c) for c in non_ascii_chars]}")
                    
                    # Add episode with empty entity_types to prevent automatic entity extraction
                    await self.graphiti.add_episode(
                        name=f"Mission Phase Interface Policy - Text {i+1}",
                        episode_body=policy,
                        source=EpisodeType.text,
                        source_description="Mission phase interface policy rule",
                        reference_time=datetime.now(timezone.utc),
                        entity_types={}  # Empty dict prevents automatic entity extraction
                    )
                    self.logger.debug(f"Successfully added text policy {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to add text policy {i+1}: {str(e)}")
                    # Continue with other policies instead of failing completely
                    continue
            
            self.policies_added = True
            self.logger.info("Successfully added mission phase interface policies to knowledge graph")
            
        except Exception as e:
            self.logger.error(f"Error in add_policies: {str(e)}")
            # Mark as added to prevent retry attempts
            self.policies_added = True
            self.logger.info("Policy addition completed with some errors")
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information including name, email, location, and timezone"""
        try:
            user_info = self.users.get(user_id)
            if not user_info:
                self.logger.warning(f"User {user_id} not found in configuration")
                return None
            
            # Ensure all required fields are present
            user_info = {
                "id": user_id,
                "name": user_info.get("name", "Unknown"),
                "email": user_info.get("email", "unknown@company.com"),
                "location": user_info.get("location", "Unknown"),
                "timezone": user_info.get("timezone"),
                "role": user_info.get("role", "user"),
                "clearance_level": user_info.get("clearance_level", "internal"),
                "active_clearances": user_info.get("active_clearances", ["internal"]),
                "emergency_access": user_info.get("emergency_access", False),
                "override_permissions": user_info.get("override_permissions", False),
                "working_hours": user_info.get("working_hours", {
                    "start": "09:00",
                    "end": "17:00",
                    "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
                })
            }
            
            self.logger.debug(f"Retrieved user info for {user_id}: {user_info['name']} ({user_info['email']})")
            return user_info
            
        except Exception as e:
            self.logger.error(f"Error getting user info for {user_id}: {str(e)}")
            return None
    
    def detect_timezone_from_location(self, location: str) -> Optional[str]:
        """Detect timezone from location using mapping"""
        try:
            if not location:
                return None
            
            # Try exact match first
            for city, tz in self.location_timezone_mapping.items():
                if city.lower() in location.lower():
                    self.logger.debug(f"Detected timezone {tz} from location {location}")
                    return tz
            
            # Try partial matches
            location_parts = location.lower().split(',')
            for part in location_parts:
                part = part.strip()
                for city, tz in self.location_timezone_mapping.items():
                    if city.lower() in part:
                        self.logger.debug(f"Detected timezone {tz} from location part {part}")
                        return tz
            
            self.logger.warning(f"Could not detect timezone from location: {location}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting timezone from location {location}: {str(e)}")
            return None
    
    def get_user_timezone(self, user_info: Dict[str, Any]) -> Optional[str]:
        """Get user's timezone, with fallback to location-based detection"""
        try:
            # First try to get explicit timezone
            if user_info.get("timezone"):
                return user_info["timezone"]
            
            # Fallback to location-based detection
            location = user_info.get("location")
            if location:
                detected_tz = self.detect_timezone_from_location(location)
                if detected_tz:
                    self.logger.info(f"Using location-based timezone detection: {detected_tz}")
                    return detected_tz
            
            self.logger.warning(f"No timezone found for user {user_info.get('id', 'unknown')}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting user timezone: {str(e)}")
            return None
    
    def get_local_time(self, user_timezone: str, current_time: Optional[str] = None) -> datetime:
        """Get current time in user's timezone"""
        try:
            if current_time:
                # Parse provided time
                dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            else:
                # Use current UTC time
                dt = datetime.now(timezone.utc)
            
            # Convert to user's timezone
            tz = pytz.timezone(user_timezone)
            local_time = dt.astimezone(tz)
            
            self.logger.debug(f"Local time in {user_timezone}: {local_time}")
            return local_time
            
        except Exception as e:
            self.logger.error(f"Error getting local time for {user_timezone}: {str(e)}")
            # Fallback to UTC
            return datetime.now(timezone.utc)
    
    def determine_time_period(self, local_time: datetime, working_hours: Dict[str, Any]) -> str:
        """Determine if current time is working hours, off hours, or weekend"""
        try:
            # Check if it's weekend
            day_name = local_time.strftime('%A').lower()
            working_days = [day.lower() for day in working_hours.get("days", [])]
            
            if day_name not in working_days:
                return "weekend"
            
            # Check if it's working hours
            current_time = local_time.strftime('%H:%M')
            start_time = working_hours.get("start", "09:00")
            end_time = working_hours.get("end", "17:00")
            
            if start_time <= current_time <= end_time:
                return "working_hours"
            else:
                return "off_hours"
                
        except Exception as e:
            self.logger.error(f"Error determining time period: {str(e)}")
            return "off_hours"  # Default to restrictive
    
    def detect_data_classification(self, query: str) -> Optional[str]:
        """Detect the data classification based on query content"""
        try:
            query_lower = query.lower()
            
            for example, class_name in self.classification_lookup.items():
                if example.lower() in query_lower:
                    self.logger.debug(f"Detected classification {class_name} from query")
                    return class_name
            
            # Default to internal if no specific classification detected
            self.logger.debug("No specific classification detected, defaulting to internal")
            return "internal"
            
        except Exception as e:
            self.logger.error(f"Error detecting data classification: {str(e)}")
            return "internal"
    
    def check_mission_phase_restriction(self, user_info: Dict[str, Any], classification: str, mission_phase: str) -> Tuple[bool, str]:
        """Check mission phase restrictions"""
        try:
            phase_config = self.mission_phases.get(mission_phase.lower())
            if not phase_config:
                return False, f"Unknown mission phase: {mission_phase}"
            
            # Check role-based restrictions
            user_role = user_info.get("role")
            allowed_roles = phase_config.get("allowed_roles", [])
            
            if user_role not in allowed_roles:
                return False, f"User role '{user_role}' not authorized for {mission_phase} phase"
            
            # Check classification restrictions
            restrictions = phase_config.get("restrictions", {})
            restriction = restrictions.get(classification, "block")
            
            if restriction == "allow":
                return True, f"Access allowed for {classification} data in {mission_phase} phase"
            else:
                return False, f"Access denied: {classification} data restricted in {mission_phase} phase"
                
        except Exception as e:
            self.logger.error(f"Error checking mission phase restriction: {str(e)}")
            return False, f"Error checking mission phase restriction: {str(e)}"
    
    def check_timezone_restriction(self, classification: str, time_period: str, mission_phase: str) -> Tuple[bool, str]:
        """Check timezone-based restrictions"""
        try:
            # Emergency phase overrides all restrictions
            if mission_phase.lower() == "emergency":
                return True, "Emergency phase overrides all timezone restrictions"
            
            # Get classification restrictions
            class_config = self.data_classifications.get(classification, {})
            restrictions = class_config.get("timezone_restrictions", {})
            
            # Check if time period is allowed
            restriction = restrictions.get(time_period, "block")
            
            if restriction == "allow":
                return True, f"Access allowed during {time_period}"
            else:
                return False, f"Access denied: {time_period} restriction for {classification} data"
                
        except Exception as e:
            self.logger.error(f"Error checking timezone restriction: {str(e)}")
            return False, f"Error checking timezone restriction: {str(e)}"
    
    def check_override_permissions(self, user_info: Dict[str, Any], mission_phase: str) -> Tuple[bool, str]:
        """Check if user has override permissions"""
        try:
            # Check if user has override permissions
            if not user_info.get("override_permissions", False):
                return False, "User does not have override permissions"
            
            # Check if override is allowed in this mission phase
            phase_config = self.mission_phases.get(mission_phase.lower())
            if not phase_config or not phase_config.get("override_allowed", False):
                return False, f"Override not allowed in {mission_phase} phase"
            
            return True, "Override permissions granted"
            
        except Exception as e:
            self.logger.error(f"Error checking override permissions: {str(e)}")
            return False, f"Error checking override permissions: {str(e)}"
    
    async def check_policy(self, 
                          user_id: str, 
                          query: str, 
                          mission_phase: str,
                          current_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Check mission phase policy with timezone-aware user tracking
        
        Args:
            user_id: User identifier
            query: Query to check
            mission_phase: Current mission phase
            current_time: Current time (optional, defaults to now)
        
        Returns:
            Dict with policy decision and tracking details
        """
        try:
            self.logger.info(f"Checking mission phase policy for user {user_id}")
            
            # Validate inputs
            if not user_id or not query or not mission_phase:
                self.logger.error("Missing required parameters")
                return {
                    "success": False,
                    "error": "Missing required parameters",
                    "user_id": user_id,
                    "query": query,
                    "mission_phase": mission_phase,
                    "allowed": False
                }
            
            # Get user information
            user_info = self.get_user_info(user_id)
            if not user_info:
                return {
                    "success": False,
                    "error": f"Unknown user: {user_id}",
                    "user_id": user_id,
                    "query": query,
                    "allowed": False
                }
            
            # Get user's timezone
            user_timezone = self.get_user_timezone(user_info)
            if not user_timezone:
                return {
                    "success": False,
                    "error": f"Could not determine timezone for user {user_id}",
                    "user_id": user_id,
                    "user_name": user_info["name"],
                    "user_email": user_info["email"],
                    "query": query,
                    "allowed": False
                }
            
            # Get local time in user's timezone
            local_time = self.get_local_time(user_timezone, current_time)
            
            # Determine time period (working hours, off hours, weekend)
            time_period = self.determine_time_period(local_time, user_info["working_hours"])
            
            # Detect data classification
            classification = self.detect_data_classification(query)
            
            # Query the knowledge graph for relevant policies (optimized)
            search_query = f"mission phase interface policy {classification} {mission_phase}"
            self.logger.debug(f"Searching graph with optimized query: {search_query}")
            
            try:
                results = await self.graphiti.search(search_query)
                self.logger.debug(f"Graph search returned {len(results) if results else 0} results")
            except Exception as e:
                self.logger.error(f"Error during graph search: {str(e)}")
                # Continue without graph results - policy logic will still work
                results = []
                self.logger.info("Continuing with policy logic despite graph search error")
            
            # Check mission phase restrictions
            phase_allowed, phase_reason = self.check_mission_phase_restriction(user_info, classification, mission_phase)
            
            # Check timezone restrictions
            timezone_allowed, timezone_reason = self.check_timezone_restriction(classification, time_period, mission_phase)
            
            # Check override permissions
            override_allowed, override_reason = self.check_override_permissions(user_info, mission_phase)
            
            # Determine final decision
            allowed = False
            reason = ""
            
            if mission_phase.lower() == "emergency":
                allowed = True
                reason = "Emergency phase overrides all restrictions"
            elif override_allowed and not phase_allowed:
                allowed = True
                reason = f"Override granted: {override_reason}"
            elif phase_allowed and timezone_allowed:
                allowed = True
                reason = f"Access granted: {phase_reason} and {timezone_reason}"
            else:
                allowed = False
                if not phase_allowed:
                    reason = phase_reason
                elif not timezone_allowed:
                    reason = timezone_reason
                else:
                    reason = "Access denied: policy restrictions"
            
            # Build result
            result = {
                "success": True,
                "user_id": user_id,
                "user_name": user_info["name"],
                "user_email": user_info["email"],
                "user_location": user_info["location"],
                "user_timezone": user_timezone,
                "user_role": user_info["role"],
                "user_clearance": user_info["clearance_level"],
                "query": query,
                "data_classification": classification,
                "mission_phase": mission_phase,
                "current_time_utc": datetime.now(timezone.utc).isoformat(),
                "local_time": local_time.isoformat(),
                "time_period": time_period,
                "allowed": allowed,
                "reason": reason,
                "policy_applied": "mission_phase_interface"
            }
            
            # Log connection attempt
            self.connection_tracker.log_connection_attempt(user_info, result)
            
            self.logger.info(f"Policy decision: {'ALLOWED' if allowed else 'BLOCKED'} - {reason}")
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in check_policy: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "user_id": user_id,
                "query": query,
                "mission_phase": mission_phase,
                "allowed": False
            }
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get a summary of all connection attempts"""
        return self.connection_tracker.get_connection_summary()


def print_graphiti_representation(user_id: str, query: str, result: dict, description: str = None):
    """Print a dynamic Graphiti Representation for the mission phase interface situation."""
    try:
        if description:
            print(f"\nGraphiti Representation for: {description}")
        else:
            print("\nGraphiti Representation:")
        
        # Extract details
        user_name = result.get("user_name", "Unknown")
        user_email = result.get("user_email", "unknown@company.com")
        user_location = result.get("user_location", "Unknown")
        user_timezone = result.get("user_timezone", "Unknown")
        user_role = result.get("user_role", "unknown")
        classification = result.get("data_classification", "unknown")
        mission_phase = result.get("mission_phase", "unknown")
        time_period = result.get("time_period", "unknown")
        local_time = result.get("local_time", "unknown")
        allowed = result.get("allowed", False)
        reason = result.get("reason", "Unknown")
        
        # Show user identification
        print(f"{user_name} —[IDENTIFIED_AS]→ User")
        print(f"{user_name} —[HAS_EMAIL]→ {user_email}")
        print(f"{user_name} —[LOCATED_IN]→ {user_location}")
        print(f"{user_location} —[HAS_TIMEZONE]→ {user_timezone}")
        print(f"{user_name} —[HAS_ROLE]→ {user_role}")
        
        # Show time relationships
        print(f"CurrentTime = {local_time}")
        print(f"{user_timezone} —[CURRENT_PERIOD]→ {time_period}")
        
        # Show query relationships
        print(f"Query —[REFERENCES]→ {classification}")
        print(f"{user_name} —[HAS_CLEARANCE]→ {classification}")
        
        # Show mission phase relationship
        print(f"{classification} —[RESTRICTED_IN]→ {mission_phase}")
        print(f"{user_role} —[AUTHORIZED_FOR]→ {mission_phase}")
        
        # Show timezone restriction
        print(f"{classification} —[BLOCKED_IN]→ {time_period}")
        
        # Show resolution logic
        if allowed:
            print(f"Resolution Logic:")
            print(f"User {user_name} ({user_email}) in {user_timezone}")
            print(f"Role: {user_role}, Current time: {local_time} ({time_period})")
            print(f"Query involves {classification} data in {mission_phase} phase")
            print(f"Firewall allows access: \"{reason}\"")
        else:
            print(f"Resolution Logic:")
            print(f"User {user_name} ({user_email}) in {user_timezone}")
            print(f"Role: {user_role}, Current time: {local_time} ({time_period})")
            print(f"Query involves {classification} data in {mission_phase} phase")
            print(f"Firewall blocks access: \"{reason}\"")
        
    except Exception as e:
        print(f"Error printing Graphiti Representation: {str(e)}")


async def main():
    """Main function to demonstrate the mission phase interface"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Mission Phase Interface demonstration")
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable is required")
        print("❌ ERROR: GROQ_API_KEY environment variable is required")
        return
    
    # Initialize Graphiti
    logger.info("Initializing Graphiti...")
    
    # Get Neo4j connection details from environment or use defaults
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    logger.info(f"Connecting to Neo4j at {neo4j_uri} with user {neo4j_user}")
    
    try:
        # Initialize Groq LLM client
        logger.debug("Initializing Groq LLM client")
        llm_config = LLMConfig(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
        llm_client = GroqClient(llm_config)
        logger.debug("Groq LLM client initialized successfully")
        
        # Initialize Graphiti with the LLM client
        logger.debug("Initializing Graphiti instance")
        graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password, llm_client)
        logger.info("Graphiti instance created successfully")
        
        # Build indices and constraints
        logger.info("Building indices and constraints...")
        await graphiti.build_indices_and_constraints()
        logger.info("Indices and constraints built successfully")
        
        # Load interface configuration from JSON
        logger.info("Loading interface configuration...")
        config = load_interface_config()
        logger.info("Interface configuration loaded successfully")
        
        # Initialize the interface with configuration
        interface = MissionPhaseInterface(graphiti, config, logger)
        logger.info("MissionPhaseInterface instance created successfully")
        
        # Add policies to the knowledge graph
        logger.info("Adding policies to knowledge graph...")
        await interface.add_policies()
        
        # Test the interface with different scenarios
        logger.info("Starting interface tests...")
        print("\n" + "="*80)
        print("🎯 Testing Mission Phase Interface with Timezone-Aware Tracking")
        print("="*80)
        
        # Load test scenarios from configuration
        test_scenarios = config.get("test_scenarios", [])
        
        for scenario in test_scenarios:
            scenario_name = scenario["scenario"]
            user_id = scenario["user_id"]
            query = scenario["query"]
            mission_phase = scenario["mission_phase"]
            current_time = scenario["current_time"]
            expected_result = scenario["expected_result"]
            expected_reason = scenario["expected_reason"]
            description = scenario.get("description", scenario_name)
            
            logger.info(f"Running test: {description}")
            print(f"\n📋 Test: {description}")
            
            try:
                result = await interface.check_policy(user_id, query, mission_phase, current_time)
                
                # Print the dynamic Graphiti Representation
                print_graphiti_representation(user_id, query, result, description)
                
                if result["success"]:
                    print(f"   User: {result['user_name']} ({result['user_email']})")
                    print(f"   Role: {result['user_role']}")
                    print(f"   Location: {result['user_location']}")
                    print(f"   Timezone: {result['user_timezone']}")
                    print(f"   Local Time: {result['local_time']}")
                    print(f"   Time Period: {result['time_period']}")
                    print(f"   Query: {result['query']}")
                    print(f"   Data Classification: {result['data_classification']}")
                    print(f"   Mission Phase: {result['mission_phase']}")
                    
                    if result.get("allowed"):
                        print(f"   ✅ ALLOWED")
                        print(f"   Reason: {result['reason']}")
                        
                        # Check if result matches expected
                        if expected_result == "allowed":
                            print(f"   ✅ EXPECTED RESULT MATCH")
                        else:
                            print(f"   ⚠️  UNEXPECTED RESULT")
                    else:
                        print(f"   ❌ BLOCKED")
                        print(f"   Reason: {result['reason']}")
                        
                        # Check if result matches expected
                        if expected_result == "blocked":
                            print(f"   ✅ EXPECTED RESULT MATCH")
                        else:
                            print(f"   ⚠️  UNEXPECTED RESULT")
                    
                    logger.info(f"Test result: {'Allowed' if result.get('allowed') else 'Blocked'} - {result['reason']}")
                else:
                    print(f"   ❌ ERROR: {result.get('error', 'Unknown error')}")
                    logger.error(f"Test failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Error during test '{description}': {str(e)}"
                logger.error(error_msg)
                print(f"   ERROR: {error_msg}")
        
        # Print connection summary
        print("\n" + "="*80)
        print("📊 Connection Tracking Summary")
        print("="*80)
        
        summary = interface.get_connection_summary()
        print(f"Total Connection Attempts: {summary.get('total_attempts', 0)}")
        print(f"Allowed Attempts: {summary.get('allowed_attempts', 0)}")
        print(f"Blocked Attempts: {summary.get('blocked_attempts', 0)}")
        print(f"Suspicious Attempts: {summary.get('suspicious_attempts', 0)}")
        
        print("\nUser Breakdown:")
        for user_id, user_stats in summary.get("user_breakdown", {}).items():
            print(f"  {user_stats['name']} ({user_stats['email']}):")
            print(f"    Total: {user_stats['total_attempts']}")
            print(f"    Allowed: {user_stats['allowed_attempts']}")
            print(f"    Blocked: {user_stats['blocked_attempts']}")
        
        if summary.get("suspicious_details"):
            print("\nSuspicious Attempts:")
            for suspicious in summary["suspicious_details"]:
                print(f"  {suspicious['user_name']} ({suspicious['user_email']}) - {suspicious['suspicious_reason']}")
        
        print("\n" + "="*80)
        print("🎉 Mission Phase Interface demonstration completed!")
        print("="*80)
        logger.info("Mission Phase Interface demonstration completed successfully")
        
    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}"
        logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        print("\nMake sure:")
        print("1. Neo4j is running and accessible")
        print("2. GROQ_API_KEY is set in your environment")
        print("3. Neo4j credentials are correct")
        print("4. pytz package is installed: pip install pytz")
        print("5. Check the log file 'mission_phase_interface.log' for detailed error information")
        
    finally:
        # Close the connection
        if 'graphiti' in locals():
            try:
                logger.info("Closing Graphiti connection...")
                await graphiti.close()
                logger.info("Graphiti connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Graphiti connection: {str(e)}")
        
        logger.info("Mission Phase Interface demonstration finished")


if __name__ == "__main__":
    asyncio.run(main()) 