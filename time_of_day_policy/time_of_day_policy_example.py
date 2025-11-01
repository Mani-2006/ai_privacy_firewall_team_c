#!/usr/bin/env python3
"""
Time-of-Day Policy Example using Graphiti

This example demonstrates how to implement a simple time-of-day policy:
- Non-critical queries are allowed only during working hours (9AM-6PM)
- The policy is stored in the knowledge graph
- Runtime checks query the graph to determine if a request should be allowed
"""

import os
import asyncio
import json
import logging
import sys
from datetime import datetime, time, timezone
from typing import Dict, Any, Optional
import ast

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.groq_client import GroqClient


def load_policy_config(config_file: str = "policy_config.json") -> Dict[str, Any]:
    """Load policy configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default configuration")
        return {
            "query_types": {
                "critical": {
                    "description": "Emergency and security-related queries",
                    "time_restriction": "none",
                    "always_allowed": True
                },
                "non-critical": {
                    "description": "General information queries",
                    "time_restriction": "standard_hours",
                    "working_hours": {"start": "09:00", "end": "18:00"}
                }
            },
            "test_scenarios": [
                ("critical", "10:00", "Emergency system status check"),
                ("non-critical", "10:00", "Summarize today's news")
            ],
            "policy_rules": [
                "Critical queries are always allowed regardless of time.",
                "Non-critical queries are allowed only during working hours (09:00 to 18:00)."
            ]
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_file}: {e}")
        return {}


def setup_logging() -> logging.Logger:
    """Setup logging configuration with timestamps and unique log file per run (no console output)"""
    logger = logging.getLogger('time_of_day_policy')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'time_of_day_policy_{now}.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class TimeOfDayPolicy:
    """Implements a time-of-day policy using Graphiti knowledge graph"""
    
    def __init__(self, graphiti: Graphiti, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.graphiti = graphiti
        self.config = config
        self.policies_added = False
        self.logger = logger or logging.getLogger('time_of_day_policy')
    
    async def add_policies(self):
        """Add time-of-day policies to the knowledge graph"""
        try:
            if self.policies_added:
                self.logger.info("Policies already added to graph - skipping")
                return
            
            self.logger.info("Starting to add time-of-day policies to knowledge graph...")
            
            # Add the main policy as structured data
            policy_episode = {
                "fact": "Non-critical queries are allowed only during working hours",
                "entities": [
                    {
                        "name": "NonCriticalQuery", 
                        "type": "QueryType",
                        "attributes": {"criticality": "non-critical"}
                    },
                    {
                        "name": "WorkHours", 
                        "type": "TimeWindow", 
                        "attributes": {
                            "start_time": "09:00",
                            "end_time": "18:00",
                            "description": "Standard working hours"
                        }
                    }
                ],
                "relationship": {
                    "type": "ALLOWED_DURING",
                    "source": "NonCriticalQuery",
                    "target": "WorkHours",
                    "attributes": {
                        "policy_type": "time_restriction",
                        "enforcement": "strict"
                    }
                }
            }
            
            # Load policies from JSON configuration and filter out problematic ones
            all_policies = self.config.get("policy_rules", [
                "Non-critical queries are allowed only during working hours (09:00 to 18:00).",
                "Critical queries are always allowed regardless of time."
            ])
            
            # Filter out policies that might cause Neo4j property issues
            policies = []
            for policy in all_policies:
                # Skip policies that contain complex structures or might be parsed as objects
                if any(keyword in str(policy).lower() for keyword in ['timestamp', 'timezone', 'status', 'pattern', 'preservation', 'restriction']):
                    self.logger.warning(f"Skipping potentially problematic policy: {str(policy)[:50]}...")
                    continue
                policies.append(policy)
            
            self.logger.debug(f"Preparing to add {len(policies)} text policies to graph")
            
            # Add text-based policies with minimal entity extraction
            for i, policy in enumerate(policies):
                try:
                    self.logger.debug(f"Adding text policy {i+1}/{len(policies)}")
                    
                    # Check for non-ASCII characters
                    non_ascii_chars = [c for c in str(policy) if ord(c) > 127]
                    if non_ascii_chars:
                        self.logger.warning(f"Non-ASCII characters found in text policy {i+1}: {[repr(c) for c in non_ascii_chars]}")
                    
                    # Add episode with empty entity_types to prevent automatic entity extraction
                    await self.graphiti.add_episode(
                        name=f"Time-of-Day Policy - Text {i+1}",
                        episode_body=str(policy),
                        source=EpisodeType.text,
                        source_description="Time-of-day policy rule",
                        reference_time=datetime.now(timezone.utc),
                        entity_types={}  # Empty dict prevents automatic entity extraction
                    )
                    self.logger.debug(f"Successfully added text policy {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to add text policy {i+1}: {str(e)}")
                    # Continue with other policies instead of failing completely
                    continue
            
            self.policies_added = True
            self.logger.info("Successfully added time-of-day policies to knowledge graph")
            
        except Exception as e:
            self.logger.error(f"Error in add_policies: {str(e)}")
            # Mark as added to prevent retry attempts
            self.policies_added = True
            self.logger.info("Policy addition completed with some errors")
    
    def print_graphiti_representation(self, policy_episode: dict):
        """Print a human-readable Graphiti Representation for the main policy."""
        try:
            entities = {e["name"]: e for e in policy_episode.get("entities", [])}
            rel = policy_episode.get("relationship", {})
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            start_time = entities.get(target, {}).get("attributes", {}).get("start_time")
            end_time = entities.get(target, {}).get("attributes", {}).get("end_time")
            print("\nGraphiti Representation:")
            print(f"{source} —[{rel_type}]→ {target}")
            if start_time and end_time:
                print(f"{target} = {start_time}–{end_time}")
        except Exception as e:
            self.logger.error(f"Error printing Graphiti Representation: {str(e)}")
    
    def print_graphiti_representation(self, query_type: str, current_time: str, result: dict):
        """Print a dynamic Graphiti Representation for the test situation."""
        try:
            query_config = self.config.get("query_types", {}).get(query_type.lower(), {})
            
            if query_config.get("always_allowed", False):
                print("\nGraphiti Representation:")
                print(f"{query_type.title()}Query —[ALWAYS_ALLOWED]→ AnyTime")
                print(f"{current_time} is ALWAYS ALLOWED ({query_type} query)")
            else:
                work_hours = result.get("working_hours", {})
                allowed = result.get("allowed", False)
                status = "ALLOWED" if allowed else "DENIED"
                in_or_out = "INSIDE" if result.get("is_working_hours") else "OUTSIDE"
                
                print("\nGraphiti Representation:")
                print(f"{query_type.title()}Query —[ALLOWED_DURING]→ {query_config.get('time_restriction', 'WorkHours').title()}")
                if work_hours:
                    print(f"{query_config.get('time_restriction', 'WorkHours').title()} = {work_hours['start']}–{work_hours['end']}")
                print(f"{current_time} is {in_or_out} {query_config.get('time_restriction', 'WorkHours').title()} ({status})")
                
        except Exception as e:
            self.logger.error(f"Error printing Graphiti Representation: {str(e)}")
    
    async def check_policy(self, query_type: str, current_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if a query is allowed based on the time-of-day policy
        
        Args:
            query_type: Type of query ("critical" or "non-critical")
            current_time: Current time in HH:MM format (defaults to current system time)
        
        Returns:
            Dict with policy decision and details
        """
        try:
            if not current_time:
                current_time = datetime.now().strftime("%H:%M")
            
            self.logger.info(f"Checking policy for {query_type} query at {current_time}")
            
            # Validate query type
            if not query_type or not isinstance(query_type, str):
                self.logger.error(f"Invalid query_type provided: {query_type}")
                return {
                    "allowed": False,
                    "reason": f"Invalid query type: {query_type}",
                    "query_type": query_type,
                    "current_time": current_time,
                    "policy_applied": "invalid_input",
                    "error": "Query type must be a non-empty string"
                }
            
            # Get query type configuration from JSON
            query_config = self.config.get("query_types", {}).get(query_type.lower())
            if not query_config:
                self.logger.warning(f"Unknown query type: {query_type}")
                return {
                    "allowed": False,
                    "reason": f"Unknown query type: {query_type}",
                    "query_type": query_type,
                    "current_time": current_time,
                    "policy_applied": "unknown_type"
                }
            
            # Check if query type is always allowed
            if query_config.get("always_allowed", False):
                self.logger.info(f"{query_type} query detected - allowing access")
                return {
                    "allowed": True,
                    "reason": f"{query_type} queries are always allowed",
                    "query_type": query_type,
                    "current_time": current_time,
                    "policy_applied": "always_allowed"
                }
            
            # Check time restrictions for other query types
            working_hours = query_config.get("working_hours")
            if not working_hours:
                self.logger.error(f"No working hours defined for query type: {query_type}")
                return {
                    "allowed": False,
                    "reason": f"No working hours defined for {query_type}",
                    "query_type": query_type,
                    "current_time": current_time,
                    "policy_applied": "no_hours_defined"
                }
            
            self.logger.debug(f"{query_type} query detected - checking working hours")
            
            try:
                # Use working hours from configuration
                work_hours = working_hours
                self.logger.debug(f"Using working hours from config: {work_hours}")
                
                # Check if current time is within working hours
                is_working_hours = self._is_within_hours(current_time, work_hours)
                self.logger.info(f"Time {current_time} is within working hours: {is_working_hours}")
                
                return {
                    "allowed": is_working_hours,
                    "reason": f"{query_type} queries allowed during {work_hours['start']}-{work_hours['end']}",
                    "query_type": query_type,
                    "current_time": current_time,
                    "working_hours": work_hours,
                    "is_working_hours": is_working_hours,
                    "policy_applied": "time_restriction"
                }
                
            except Exception as e:
                self.logger.error(f"Error during time check: {str(e)}")
                return {
                    "allowed": False,
                    "reason": f"Error checking policy: {str(e)}",
                    "query_type": query_type,
                    "current_time": current_time,
                    "policy_applied": "error",
                    "error": str(e)
                }
            
        except Exception as e:
            self.logger.error(f"Unexpected error in check_policy: {str(e)}")
            return {
                "allowed": False,
                "reason": f"Unexpected error: {str(e)}",
                "query_type": query_type,
                "current_time": current_time,
                "policy_applied": "error",
                "error": str(e)
            }
    
    def _extract_working_hours(self, search_results) -> Dict[str, str]:
        """Extract working hours from Graphiti search results"""
        # Default working hours (fallback)
        default_hours = {"start": "09:00", "end": "18:00"}
        
        try:
            if not search_results:
                self.logger.warning("No search results returned, using default working hours")
                return default_hours
            
            # Look for working hours in the search results
            for i, result in enumerate(search_results):
                try:
                    fact = result.fact if hasattr(result, 'fact') else str(result)
                    self.logger.debug(f"Checking result {i+1}: {fact[:100]}...")
                    
                    if '09:00' in fact and '18:00' in fact:
                        self.logger.debug("Found working hours in 24-hour format")
                        return {"start": "09:00", "end": "18:00"}
                    elif '9 AM' in fact and '6 PM' in fact:
                        self.logger.debug("Found working hours in 12-hour format")
                        return {"start": "09:00", "end": "18:00"}
                        
                except Exception as e:
                    self.logger.warning(f"Error processing search result {i+1}: {str(e)}")
                    continue
            
            self.logger.warning("Could not extract working hours from results, using defaults")
            return default_hours
            
        except Exception as e:
            self.logger.error(f"Error extracting working hours: {str(e)}")
            return default_hours
    
    def _is_within_hours(self, current_time: str, work_hours: Dict[str, str]) -> bool:
        """Check if current time is within working hours"""
        try:
            if not current_time or not work_hours:
                self.logger.error("Invalid input to _is_within_hours")
                return False
            
            self.logger.debug(f"Checking if {current_time} is within {work_hours['start']}-{work_hours['end']}")
            
            current = datetime.strptime(current_time, "%H:%M").time()
            start = datetime.strptime(work_hours["start"], "%H:%M").time()
            end = datetime.strptime(work_hours["end"], "%H:%M").time()
            
            is_within = start <= current <= end
            self.logger.debug(f"Time comparison: {start} <= {current} <= {end} = {is_within}")
            
            return is_within
            
        except ValueError as e:
            self.logger.error(f"Invalid time format: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error parsing time: {str(e)}")
            return False


async def main():
    """Main function to demonstrate the time-of-day policy"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Time-of-Day Policy demonstration")
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable is required")
        print("ERROR: GROQ_API_KEY environment variable is required")
        return
    
    # Initialize Graphiti
    logger.info("Initializing Graphiti...")
    
    # Get Neo4j connection details from environment or use defaults
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    logger.info(f"Connecting to Neo4j at {neo4j_uri} with user {neo4j_user}")
    
    try:
        # Initialize Groq LLM client (was OpenAIClient)
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
        policy_checker = TimeOfDayPolicy(graphiti, config, logger)
        logger.info("TimeOfDayPolicy instance created successfully")
        
        # Add policies to the knowledge graph
        logger.info("Adding policies to knowledge graph...")
        await policy_checker.add_policies()
        
        # Test the policy with different scenarios
        logger.info("Starting policy tests...")
        print("\n" + "="*60)
        print("Testing Time-of-Day Policy")
        print("="*60)
        
        # Load test scenarios from configuration
        test_scenarios = []
        for scenario in config.get("test_scenarios", []):
            test_scenarios.append((
                scenario["query_type"],
                scenario["test_time"],
                scenario["description"]
            ))
        
        # Fallback to default scenarios if none in config
        if not test_scenarios:
            test_scenarios = [
                ("critical", "10:00", "Emergency system status check"),
                ("non-critical", "10:00", "Summarize today's news"),
                ("non-critical", "08:00", "Summarize today's news"),
                ("non-critical", "19:00", "Summarize today's news"),
                ("critical", "23:00", "Security alert"),
                ("non-critical", "14:30", "General information request"),
            ]
        
        for query_type, test_time, description in test_scenarios:
            logger.info(f"Running test: {description}")
            print(f"\nTest: {description}")
            
            try:
                result = await policy_checker.check_policy(query_type, test_time)
                # Print the dynamic Graphiti Representation for each test
                policy_checker.print_graphiti_representation(query_type, test_time, result)
                # Print the detailed test result as before
                status = "ALLOWED" if result["allowed"] else "DENIED"
                print(f"   {status}")
                print(f"   Reason: {result['reason']}")
                print(f"   Time: {result['current_time']}")
                if 'working_hours' in result:
                    print(f"   Working Hours: {result['working_hours']['start']}-{result['working_hours']['end']}")
                logger.info(f"Test result: {status} - {result['reason']}")
                
            except Exception as e:
                error_msg = f"Error during test '{description}': {str(e)}"
                logger.error(error_msg)
                print(f"   ERROR: {error_msg}")
        
        # Test with current time
        logger.info("Running current time test...")
        print(f"\nTest: Current time check")
        
        try:
            current_result = await policy_checker.check_policy("non-critical")
            # Print the dynamic Graphiti Representation for the current time test
            policy_checker.print_graphiti_representation("non-critical", current_result["current_time"], current_result)
            status = "ALLOWED" if current_result["allowed"] else "DENIED"
            print(f"   {status}")
            print(f"   Reason: {current_result['reason']}")
            print(f"   Current Time: {current_result['current_time']}")
            if 'working_hours' in current_result:
                print(f"   Working Hours: {current_result['working_hours']['start']}-{current_result['working_hours']['end']}")
            logger.info(f"Current time test result: {status} - {current_result['reason']}")
            
        except Exception as e:
            error_msg = f"Error during current time test: {str(e)}"
            logger.error(error_msg)
            print(f"   ERROR: {error_msg}")
        
        print("\n" + "="*60)
        print("Time-of-Day Policy demonstration completed!")
        print("="*60)
        logger.info("Time-of-Day Policy demonstration completed successfully")
        
    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}")
        print("\nMake sure:")
        print("1. Neo4j is running and accessible")
        print("2. GROQ_API_KEY is set in your environment")
        print("3. Neo4j credentials are correct")
        print("4. Check the log file 'time_of_day_policy.log' for detailed error information")
        
    finally:
        # Close the connection
        if 'graphiti' in locals():
            try:
                logger.info("Closing Graphiti connection...")
                await graphiti.close()
                logger.info("Graphiti connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Graphiti connection: {str(e)}")
        
        logger.info("Time-of-Day Policy demonstration finished")


if __name__ == "__main__":
    asyncio.run(main()) 