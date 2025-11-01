#!/usr/bin/env python3
"""
Timezone-Aware Policy Example using Graphiti

This example demonstrates how to implement timezone-aware access control policies:
- User identification with email and name
- Location-based timezone detection
- Dynamic policy application based on user's local time
- Graphiti knowledge graph integration for policy storage

Policy Rules:
- Include user email and name in policy decisions
- Look up timezone based on user location if not specified
- Apply timezone-aware restrictions based on user's local time
- Emergency phase overrides all timezone restrictions
- Working hours are defined per user's timezone
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


def load_policy_config(config_file: str = "timezone_aware_config.json") -> Dict[str, Any]:
    """Load policy configuration from JSON file"""
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
            "location_timezone_mapping": {
                "New York": "America/New_York",
                "London": "Europe/London",
                "Tokyo": "Asia/Tokyo"
            },
            "test_scenarios": [
                {
                    "scenario": "new_york_working_hours",
                    "user_id": "user123",
                    "query": "What are the coordinates of Project Zeus?",
                    "mission_phase": "pre_deployment",
                    "current_time": "2024-01-15T14:30:00",
                    "expected_result": "allowed"
                }
            ]
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_file}: {e}")
        return {}


def setup_logging() -> logging.Logger:
    """Setup logging configuration with timestamps and unique log file per run"""
    logger = logging.getLogger('timezone_aware_policy')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'timezone_aware_policy_{now}.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class TimezoneAwarePolicy:
    """Implements timezone-aware access control policies using Graphiti"""
    
    def __init__(self, graphiti: Graphiti, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.graphiti = graphiti
        self.config = config
        self.policies_added = False
        self.logger = logger or logging.getLogger('timezone_aware_policy')
        
        # Build lookup dictionaries for fast access
        self.users = config.get("users", {})
        self.location_timezone_mapping = config.get("location_timezone_mapping", {})
        self.timezone_policies = config.get("timezone_policies", {})
        self.mission_phases = config.get("mission_phases", {})
        self.data_classifications = config.get("data_classifications", {})
        
        # Build data classification lookup by examples
        self.classification_lookup = {}
        for class_name, class_config in self.data_classifications.items():
            for example in class_config.get("examples", []):
                self.classification_lookup[example.lower()] = class_name
    
    async def add_policies(self):
        """Add timezone-aware policies to the knowledge graph"""
        try:
            if self.policies_added:
                self.logger.info("Timezone-aware policies already added to graph - skipping")
                return
            
            self.logger.info("Starting to add timezone-aware policies to knowledge graph...")
            
            # Load policies from JSON configuration and filter out problematic ones
            all_policies = self.config.get("timezone_detection_rules", [
                "Include user email and name in policy decisions",
                "Look up timezone based on user location if not specified",
                "Apply timezone-aware restrictions based on user's local time"
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
                        name=f"Timezone-Aware Policy - Text {i+1}",
                        episode_body=policy,
                        source=EpisodeType.text,
                        source_description="Timezone-aware policy rule",
                        reference_time=datetime.now(timezone.utc),
                        entity_types={}  # Empty dict prevents automatic entity extraction
                    )
                    self.logger.debug(f"Successfully added text policy {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to add text policy {i+1}: {str(e)}")
                    # Continue with other policies instead of failing completely
                    continue
            
            self.policies_added = True
            self.logger.info("Successfully added timezone-aware policies to knowledge graph")
            
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
    
    def check_timezone_restriction(self, classification: str, time_period: str, mission_phase: str) -> Tuple[bool, str]:
        """Check if access is allowed based on timezone-aware restrictions"""
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
    
    async def check_policy(self, 
                          user_id: str, 
                          query: str, 
                          mission_phase: str,
                          current_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Check timezone-aware policy
        
        Args:
            user_id: User identifier
            query: Query to check
            mission_phase: Current mission phase
            current_time: Current time (optional, defaults to now)
        
        Returns:
            Dict with policy decision and timezone details
        """
        try:
            self.logger.info(f"Checking timezone-aware policy for user {user_id}")
            
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
            search_query = f"timezone aware policy {classification} {mission_phase}"
            self.logger.debug(f"Searching graph with optimized query: {search_query}")
            
            try:
                results = await self.graphiti.search(search_query)
                self.logger.debug(f"Graph search returned {len(results) if results else 0} results")
            except Exception as e:
                self.logger.error(f"Error during graph search: {str(e)}")
                # Continue without graph results - policy logic will still work
                results = []
                self.logger.info("Continuing with policy logic despite graph search error")
            
            # Check timezone-based restrictions
            allowed, reason = self.check_timezone_restriction(classification, time_period, mission_phase)
            
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
                "policy_applied": "timezone_aware_access_control"
            }
            
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


def print_graphiti_representation(user_id: str, query: str, result: dict, description: str = None):
    """Print a dynamic Graphiti Representation for the timezone-aware situation."""
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
        
        # Show time relationships
        print(f"CurrentTime = {local_time}")
        print(f"{user_timezone} —[CURRENT_PERIOD]→ {time_period}")
        
        # Show query relationships
        print(f"Query —[REFERENCES]→ {classification}")
        print(f"{user_name} —[HAS_CLEARANCE]→ {classification}")
        
        # Show mission phase relationship
        print(f"{classification} —[RESTRICTED_IN]→ {mission_phase}")
        
        # Show timezone restriction
        print(f"{classification} —[BLOCKED_IN]→ {time_period}")
        
        # Show resolution logic
        if allowed:
            print(f"Resolution Logic:")
            print(f"User {user_name} ({user_email}) in {user_timezone}")
            print(f"Current time: {local_time} ({time_period})")
            print(f"Query involves {classification} data")
            print(f"Firewall allows access: \"{reason}\"")
        else:
            print(f"Resolution Logic:")
            print(f"User {user_name} ({user_email}) in {user_timezone}")
            print(f"Current time: {local_time} ({time_period})")
            print(f"Query involves {classification} data")
            print(f"Firewall blocks access: \"{reason}\"")
        
    except Exception as e:
        print(f"Error printing Graphiti Representation: {str(e)}")


async def main():
    """Main function to demonstrate the timezone-aware policy"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Timezone-Aware Policy demonstration")
    
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
        
        # Load policy configuration from JSON
        logger.info("Loading policy configuration...")
        config = load_policy_config()
        logger.info("Policy configuration loaded successfully")
        
        # Initialize the policy checker with configuration
        policy_checker = TimezoneAwarePolicy(graphiti, config, logger)
        logger.info("TimezoneAwarePolicy instance created successfully")
        
        # Add policies to the knowledge graph
        logger.info("Adding policies to knowledge graph...")
        await policy_checker.add_policies()
        
        # Test the policy with different scenarios
        logger.info("Starting policy tests...")
        print("\n" + "="*80)
        print("🌍 Testing Timezone-Aware Policy")
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
                result = await policy_checker.check_policy(user_id, query, mission_phase, current_time)
                
                # Print the dynamic Graphiti Representation
                print_graphiti_representation(user_id, query, result, description)
                
                if result["success"]:
                    print(f"   User: {result['user_name']} ({result['user_email']})")
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
        
        print("\n" + "="*80)
        print("🎉 Timezone-Aware Policy demonstration completed!")
        print("="*80)
        logger.info("Timezone-Aware Policy demonstration completed successfully")
        
    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}"
        logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        print("\nMake sure:")
        print("1. Neo4j is running and accessible")
        print("2. GROQ_API_KEY is set in your environment")
        print("3. Neo4j credentials are correct")
        print("4. pytz package is installed: pip install pytz")
        print("5. Check the log file 'timezone_aware_policy.log' for detailed error information")
        
    finally:
        # Close the connection
        if 'graphiti' in locals():
            try:
                logger.info("Closing Graphiti connection...")
                await graphiti.close()
                logger.info("Graphiti connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Graphiti connection: {str(e)}")
        
        logger.info("Timezone-Aware Policy demonstration finished")


if __name__ == "__main__":
    asyncio.run(main()) 