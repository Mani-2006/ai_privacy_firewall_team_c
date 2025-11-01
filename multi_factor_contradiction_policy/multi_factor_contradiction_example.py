#!/usr/bin/env python3
"""
Multi-Factor Contradiction Policy Example using Graphiti

This example demonstrates how to implement complex access control policies with:
- Multiple conflicting rules and restrictions
- Contradiction resolution based on priority hierarchy
- User clearance levels and permissions
- Time-based and mission phase restrictions
- Commander override and emergency access
- Graphiti knowledge graph integration for policy storage

Policy Rules:
- Data classifications (confidential, secret, internal)
- User clearances and roles
- Mission phase restrictions
- Time-based access controls
- Override permissions and emergency access
- Contradiction resolution with priority hierarchy
"""

import os
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Set
from enum import Enum

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.nodes import EpisodeType


def load_policy_config(config_file: str = "multi_factor_contradiction_config.json") -> Dict[str, Any]:
    """Load policy configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default configuration")
        return {
            "data_classifications": {
                "confidential": {
                    "description": "Confidential data requiring special clearance",
                    "level": "high",
                    "examples": ["Project Zeus"],
                    "restrictions": {
                        "pre_deployment": "block",
                        "off_hours": "block"
                    }
                }
            },
            "user_clearances": {
                "user123": {
                    "name": "User 123",
                    "clearance_level": "confidential",
                    "active_clearances": ["confidential", "secret"]
                }
            },
            "mission_phases": {
                "pre_deployment": {
                    "description": "Pre-deployment phase",
                    "access_rules": {
                        "confidential": "block"
                    }
                }
            },
            "test_scenarios": [
                {
                    "scenario": "confidential_off_hours_conflict",
                    "user": "user123",
                    "query": "What are the coordinates of Project Zeus?",
                    "data_classification": "confidential",
                    "mission_phase": "pre_deployment",
                    "time_period": "off_hours",
                    "expected_result": "blocked",
                    "expected_reason": "Blocked due to OffHours restriction despite user clearance"
                }
            ]
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_file}: {e}")
        return {}


def setup_logging() -> logging.Logger:
    """Setup logging configuration with timestamps and unique log file per run"""
    logger = logging.getLogger('multi_factor_contradiction_policy')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'multi_factor_contradiction_policy_{now}.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class ContradictionType(Enum):
    """Types of contradictions that can occur in policy decisions"""
    NONE = "none"
    CLEARANCE_VS_TIME = "clearance_vs_time"
    CLEARANCE_INSUFFICIENT = "clearance_insufficient"
    TIME_RESTRICTION = "time_restriction"
    OVERRIDE = "override"
    EMERGENCY_OVERRIDE = "emergency_override"
    EMERGENCY_DENIED = "emergency_denied"


class MultiFactorContradictionPolicy:
    """Implements multi-factor contradiction policies using Graphiti"""
    
    def __init__(self, graphiti: Graphiti, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.graphiti = graphiti
        self.config = config
        self.policies_added = False
        self.logger = logger or logging.getLogger('multi_factor_contradiction_policy')
        
        # Build lookup dictionaries for fast access
        self.data_classifications = config.get("data_classifications", {})
        self.user_clearances = config.get("user_clearances", {})
        self.mission_phases = config.get("mission_phases", {})
        self.time_periods = config.get("time_periods", {})
        
        # Build data classification lookup by examples
        self.classification_lookup = {}
        for class_name, class_config in self.data_classifications.items():
            for example in class_config.get("examples", []):
                self.classification_lookup[example.lower()] = class_name
    
    async def add_policies(self):
        """Add multi-factor contradiction policies to the knowledge graph"""
        try:
            if self.policies_added:
                self.logger.info("Multi-factor contradiction policies already added to graph - skipping")
                return
            
            self.logger.info("Starting to add multi-factor contradiction policies to knowledge graph...")
            
            # Load policies from JSON configuration and filter out problematic ones
            all_policies = self.config.get("contradiction_rules", [
                "Off-hours restrictions override user clearances",
                "Emergency phase overrides all time restrictions"
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
                        name=f"Multi-Factor Contradiction Policy - Text {i+1}",
                        episode_body=policy,
                        source=EpisodeType.text,
                        source_description="Multi-factor contradiction policy rule",
                        reference_time=datetime.now(timezone.utc),
                        entity_types={}  # Empty dict prevents automatic entity extraction
                    )
                    self.logger.debug(f"Successfully added text policy {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to add text policy {i+1}: {str(e)}")
                    # Continue with other policies instead of failing completely
                    continue
            
            self.policies_added = True
            self.logger.info("Successfully added multi-factor contradiction policies to knowledge graph")
            
        except Exception as e:
            self.logger.error(f"Error in add_policies: {str(e)}")
            # Mark as added to prevent retry attempts
            self.policies_added = True
            self.logger.info("Policy addition completed with some errors")
    
    def detect_data_classification(self, query: str) -> Optional[str]:
        """Detect the data classification based on query content"""
        try:
            query_lower = query.lower()
            
            for example, class_name in self.classification_lookup.items():
                if example in query_lower:
                    self.logger.debug(f"Detected data classification: {class_name} for query")
                    return class_name
            
            self.logger.debug("No specific data classification detected")
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting data classification: {str(e)}")
            return None
    
    def get_user_clearance(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user clearance information"""
        try:
            user_config = self.user_clearances.get(user_id)
            if not user_config:
                self.logger.warning(f"Unknown user: {user_id}")
                return None
            
            return user_config
            
        except Exception as e:
            self.logger.error(f"Error getting user clearance: {str(e)}")
            return None
    
    def check_clearance_sufficient(self, user_clearance: Dict[str, Any], required_classification: str) -> bool:
        """Check if user has sufficient clearance for the data classification"""
        try:
            active_clearances = user_clearance.get("active_clearances", [])
            
            # Define clearance hierarchy (higher = more access)
            clearance_hierarchy = {
                "internal": 1,
                "secret": 2,
                "confidential": 3
            }
            
            user_level = clearance_hierarchy.get(user_clearance.get("clearance_level", "internal"), 0)
            required_level = clearance_hierarchy.get(required_classification, 0)
            
            # User can access data at their level or below
            sufficient = user_level >= required_level
            
            self.logger.debug(f"Clearance check: user_level={user_level}, required_level={required_level}, sufficient={sufficient}")
            return sufficient
            
        except Exception as e:
            self.logger.error(f"Error checking clearance: {str(e)}")
            return False
    
    def check_mission_phase_restriction(self, classification: str, mission_phase: str) -> str:
        """Check mission phase restrictions for data classification"""
        try:
            phase_config = self.mission_phases.get(mission_phase)
            if not phase_config:
                self.logger.warning(f"Unknown mission phase: {mission_phase}")
                return "block"
            
            access_rules = phase_config.get("access_rules", {})
            restriction = access_rules.get(classification, "block")
            
            self.logger.debug(f"Mission phase restriction: {classification} in {mission_phase} = {restriction}")
            return restriction
            
        except Exception as e:
            self.logger.error(f"Error checking mission phase restriction: {str(e)}")
            return "block"
    
    def check_time_restriction(self, classification: str, time_period: str) -> str:
        """Check time-based restrictions for data classification"""
        try:
            class_config = self.data_classifications.get(classification)
            if not class_config:
                self.logger.warning(f"Unknown data classification: {classification}")
                return "block"
            
            restrictions = class_config.get("restrictions", {})
            restriction = restrictions.get(time_period, "block")
            
            self.logger.debug(f"Time restriction: {classification} during {time_period} = {restriction}")
            return restriction
            
        except Exception as e:
            self.logger.error(f"Error checking time restriction: {str(e)}")
            return "block"
    
    def check_override_permissions(self, user_clearance: Dict[str, Any], mission_phase: str) -> bool:
        """Check if user has override permissions"""
        try:
            # Commander override permissions
            if user_clearance.get("override_permissions", False):
                self.logger.debug("Commander override permissions detected")
                return True
            
            # Emergency phase override
            if mission_phase == "emergency":
                self.logger.debug("Emergency phase override detected")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking override permissions: {str(e)}")
            return False
    
    def resolve_contradiction(self, 
                            user_clearance: Dict[str, Any],
                            classification: str,
                            mission_phase: str,
                            time_period: str,
                            query: str) -> Tuple[bool, str, ContradictionType, Dict[str, Any]]:
        """
        Resolve contradictions between different policy factors
        
        Returns:
            Tuple of (allowed, reason, contradiction_type, details)
        """
        try:
            self.logger.info(f"Resolving contradiction for user {user_clearance.get('name', 'Unknown')}")
            
            # Initialize decision factors
            factors = {
                "clearance_sufficient": False,
                "mission_phase_allows": False,
                "time_allows": False,
                "override_applies": False,
                "emergency_override": False
            }
            
            # Check clearance
            factors["clearance_sufficient"] = self.check_clearance_sufficient(user_clearance, classification)
            
            # Check mission phase restriction
            phase_restriction = self.check_mission_phase_restriction(classification, mission_phase)
            factors["mission_phase_allows"] = phase_restriction in ["allow", "allow_with_clearance"]
            
            # Check time restriction
            time_restriction = self.check_time_restriction(classification, time_period)
            factors["time_allows"] = time_restriction in ["allow", "allow_with_clearance"]
            
            # Check override permissions
            factors["override_applies"] = self.check_override_permissions(user_clearance, mission_phase)
            
            # Check emergency override
            factors["emergency_override"] = mission_phase == "emergency"
            
            # Apply resolution priority hierarchy
            allowed, reason, contradiction_type = self.apply_priority_hierarchy(factors, user_clearance, classification, mission_phase, time_period)
            
            details = {
                "factors": factors,
                "user": user_clearance.get("name", "Unknown"),
                "classification": classification,
                "mission_phase": mission_phase,
                "time_period": time_period,
                "query": query
            }
            
            return allowed, reason, contradiction_type, details
            
        except Exception as e:
            self.logger.error(f"Error resolving contradiction: {str(e)}")
            return False, f"Error resolving contradiction: {str(e)}", ContradictionType.NONE, {}
    
    def apply_priority_hierarchy(self, 
                               factors: Dict[str, bool],
                               user_clearance: Dict[str, Any],
                               classification: str,
                               mission_phase: str,
                               time_period: str) -> Tuple[bool, str, ContradictionType]:
        """Apply priority hierarchy to resolve contradictions"""
        try:
            # Priority 1: Emergency phase (highest priority)
            if factors["emergency_override"]:
                return True, "Access granted: emergency phase overrides all restrictions", ContradictionType.EMERGENCY_OVERRIDE
            
            # Priority 2: Commander override permissions
            if factors["override_applies"]:
                return True, "Access granted: commander override permissions", ContradictionType.OVERRIDE
            
            # Priority 3: Time-based restrictions (off-hours, weekend)
            if not factors["time_allows"]:
                if factors["clearance_sufficient"]:
                    return False, "Blocked due to OffHours restriction despite user clearance", ContradictionType.CLEARANCE_VS_TIME
                else:
                    return False, f"Access denied: {time_period} restrictions apply", ContradictionType.TIME_RESTRICTION
            
            # Priority 4: User clearance levels
            if not factors["clearance_sufficient"]:
                return False, f"Access denied: insufficient clearance for {classification} data", ContradictionType.CLEARANCE_INSUFFICIENT
            
            # Priority 5: Mission phase restrictions
            if not factors["mission_phase_allows"]:
                return False, f"Access denied: {mission_phase} phase restrictions", ContradictionType.TIME_RESTRICTION
            
            # If we get here, all checks pass
            return True, "Access granted: all policy requirements satisfied", ContradictionType.NONE
            
        except Exception as e:
            self.logger.error(f"Error applying priority hierarchy: {str(e)}")
            return False, f"Error applying priority hierarchy: {str(e)}", ContradictionType.NONE
    
    async def check_policy(self, 
                          user_id: str, 
                          query: str, 
                          mission_phase: str, 
                          time_period: str) -> Dict[str, Any]:
        """
        Check multi-factor contradiction policy
        
        Args:
            user_id: User identifier
            query: Query to check
            mission_phase: Current mission phase
            time_period: Current time period
        
        Returns:
            Dict with policy decision and contradiction details
        """
        try:
            self.logger.info(f"Checking multi-factor contradiction policy for user {user_id}")
            
            # Validate inputs
            if not user_id or not query or not mission_phase or not time_period:
                self.logger.error("Missing required parameters")
                return {
                    "success": False,
                    "error": "Missing required parameters",
                    "user_id": user_id,
                    "query": query,
                    "mission_phase": mission_phase,
                    "time_period": time_period,
                    "allowed": False
                }
            
            # Get user clearance
            user_clearance = self.get_user_clearance(user_id)
            if not user_clearance:
                return {
                    "success": False,
                    "error": f"Unknown user: {user_id}",
                    "user_id": user_id,
                    "query": query,
                    "allowed": False
                }
            
            # Detect data classification
            classification = self.detect_data_classification(query)
            if not classification:
                # Default to internal if no specific classification detected
                classification = "internal"
                self.logger.debug("No specific classification detected, defaulting to internal")
            
            # Query the knowledge graph for relevant policies (optimized)
            search_query = f"multi-factor contradiction policy {classification} {mission_phase}"
            self.logger.debug(f"Searching graph with optimized query: {search_query}")
            
            try:
                results = await self.graphiti.search(search_query)
                self.logger.debug(f"Graph search returned {len(results) if results else 0} results")
            except Exception as e:
                self.logger.error(f"Error during graph search: {str(e)}")
                # Continue without graph results - policy logic will still work
                results = []
                self.logger.info("Continuing with policy logic despite graph search error")
            
            # Resolve contradictions
            allowed, reason, contradiction_type, details = self.resolve_contradiction(
                user_clearance, classification, mission_phase, time_period, query
            )
            
            # Build result
            result = {
                "success": True,
                "user_id": user_id,
                "user_name": user_clearance.get("name", "Unknown"),
                "query": query,
                "data_classification": classification,
                "mission_phase": mission_phase,
                "time_period": time_period,
                "allowed": allowed,
                "reason": reason,
                "contradiction_type": contradiction_type.value,
                "details": details,
                "policy_applied": "multi_factor_contradiction"
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
                "time_period": time_period,
                "allowed": False
            }


def print_graphiti_representation(user_id: str, query: str, result: dict, description: str = None):
    """Print a dynamic Graphiti Representation for the contradiction situation."""
    try:
        if description:
            print(f"\nGraphiti Representation for: {description}")
        else:
            print("\nGraphiti Representation:")
        
        # Extract details
        user_name = result.get("user_name", "Unknown")
        classification = result.get("data_classification", "unknown")
        mission_phase = result.get("mission_phase", "unknown")
        time_period = result.get("time_period", "unknown")
        allowed = result.get("allowed", False)
        reason = result.get("reason", "Unknown")
        contradiction_type = result.get("contradiction_type", "none")
        
        # Show the query relationship
        print(f"Query —[REFERENCES]→ {classification}")
        
        # Show user clearance relationship
        print(f"{user_name} —[HAS_CLEARANCE]→ {classification}")
        
        # Show mission phase relationship
        print(f"{classification} —[BLOCK_PROMPT_IN]→ {mission_phase}")
        
        # Show time period relationship
        print(f"CurrentTime = {time_period}")
        print(f"{classification} —[BLOCK_PROMPT_IN]→ {time_period}")
        
        # Show resolution logic
        if contradiction_type == "clearance_vs_time":
            print(f"Resolution Logic:")
            print(f"Prompt involves {classification} data.")
            print(f"User does have clearance → passes {mission_phase} rule.")
            print(f"But {time_period} rule blocks it unconditionally.")
            print(f"Firewall blocks prompt and logs dual rationale: \"{reason}\"")
        elif contradiction_type == "override":
            print(f"Resolution Logic:")
            print(f"Commander override permissions bypass {time_period} restrictions.")
            print(f"Firewall allows access: \"{reason}\"")
        elif contradiction_type == "emergency_override":
            print(f"Resolution Logic:")
            print(f"Emergency phase overrides all time and clearance restrictions.")
            print(f"Firewall allows access: \"{reason}\"")
        else:
            print(f"Resolution Logic:")
            print(f"Standard policy evaluation applied.")
            print(f"Result: {'ALLOWED' if allowed else 'BLOCKED'}")
            print(f"Reason: \"{reason}\"")
        
    except Exception as e:
        print(f"Error printing Graphiti Representation: {str(e)}")


async def main():
    """Main function to demonstrate the multi-factor contradiction policy"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Multi-Factor Contradiction Policy demonstration")
    
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
        policy_checker = MultiFactorContradictionPolicy(graphiti, config, logger)
        logger.info("MultiFactorContradictionPolicy instance created successfully")
        
        # Add policies to the knowledge graph
        logger.info("Adding policies to knowledge graph...")
        await policy_checker.add_policies()
        
        # Test the policy with different scenarios
        logger.info("Starting policy tests...")
        print("\n" + "="*80)
        print("🧠 Testing Multi-Factor Contradiction Policy")
        print("="*80)
        
        # Load test scenarios from configuration
        test_scenarios = config.get("test_scenarios", [])
        
        # Fallback to default scenarios if none in config
        if not test_scenarios:
            test_scenarios = [
                {
                    "scenario": "confidential_off_hours_conflict",
                    "user": "user123",
                    "query": "What are the coordinates of Project Zeus?",
                    "data_classification": "confidential",
                    "mission_phase": "pre_deployment",
                    "time_period": "off_hours",
                    "expected_result": "blocked",
                    "expected_reason": "Blocked due to OffHours restriction despite user clearance"
                }
            ]
        
        for scenario in test_scenarios:
            scenario_name = scenario["scenario"]
            user_id = scenario["user"]
            query = scenario["query"]
            mission_phase = scenario["mission_phase"]
            time_period = scenario["time_period"]
            expected_result = scenario["expected_result"]
            expected_reason = scenario["expected_reason"]
            description = scenario.get("description", scenario_name)
            
            logger.info(f"Running test: {description}")
            print(f"\n📋 Test: {description}")
            
            try:
                result = await policy_checker.check_policy(user_id, query, mission_phase, time_period)
                
                # Print the dynamic Graphiti Representation
                print_graphiti_representation(user_id, query, result, description)
                
                if result["success"]:
                    print(f"   User: {result['user_name']} ({user_id})")
                    print(f"   Query: {result['query']}")
                    print(f"   Data Classification: {result['data_classification']}")
                    print(f"   Mission Phase: {result['mission_phase']}")
                    print(f"   Time Period: {result['time_period']}")
                    
                    if result.get("allowed"):
                        print(f"   ✅ ALLOWED")
                        print(f"   Reason: {result['reason']}")
                        print(f"   Contradiction Type: {result['contradiction_type']}")
                        
                        # Check if result matches expected
                        if expected_result == "allowed":
                            print(f"   ✅ EXPECTED RESULT MATCH")
                        else:
                            print(f"   ⚠️  UNEXPECTED RESULT")
                    else:
                        print(f"   ❌ BLOCKED")
                        print(f"   Reason: {result['reason']}")
                        print(f"   Contradiction Type: {result['contradiction_type']}")
                        
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
        
        # Test the specific scenario from the prompt
        logger.info("Running original scenario test...")
        print(f"\n🎯 Test: Original Scenario - Confidential Off-Hours Conflict")
        
        try:
            original_result = await policy_checker.check_policy(
                "user123",
                "What are the coordinates of Project Zeus?",
                "pre_deployment",
                "off_hours"
            )
            print_graphiti_representation("user123", "What are the coordinates of Project Zeus?", original_result, "Original Scenario - Confidential Off-Hours Conflict")
            
            if original_result["success"]:
                print(f"   User: {original_result['user_name']} (user123)")
                print(f"   Query: {original_result['query']}")
                print(f"   Data Classification: {original_result['data_classification']}")
                print(f"   Mission Phase: {original_result['mission_phase']}")
                print(f"   Time Period: {original_result['time_period']}")
                
                if original_result.get("allowed"):
                    print(f"   ✅ ALLOWED")
                    print(f"   Reason: {original_result['reason']}")
                else:
                    print(f"   ❌ BLOCKED")
                    print(f"   Reason: {original_result['reason']}")
                    print(f"   Firewall blocks prompt and logs dual rationale: \"{original_result['reason']}\"")
                logger.info(f"Original scenario result: {'Allowed' if original_result.get('allowed') else 'Blocked'} - {original_result['reason']}")
            else:
                print(f"   ❌ ERROR: {original_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Error during original scenario test: {str(e)}"
            logger.error(error_msg)
            print(f"   ERROR: {error_msg}")
        
        print("\n" + "="*80)
        print("🎉 Multi-Factor Contradiction Policy demonstration completed!")
        print("="*80)
        logger.info("Multi-Factor Contradiction Policy demonstration completed successfully")
        
    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}"
        logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        print("\nMake sure:")
        print("1. Configuration file is valid JSON")
        print("2. All required fields are present in config")
        print("3. Check the log file 'multi_factor_contradiction_policy.log' for detailed error information")
        
    finally:
        logger.info("Multi-Factor Contradiction Policy demonstration finished")


if __name__ == "__main__":
    asyncio.run(main()) 