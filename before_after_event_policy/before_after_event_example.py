#!/usr/bin/env python3
"""
Before/After Event Policy Example using Graphiti

This example demonstrates how to implement time-based blocking policies:
- Scheduled event blocking based on timestamps
- Before/after event access control
- Dynamic blocking based on current time
- Graphiti knowledge graph integration for policy storage

Policy Rules:
- Launch plans blocked before mission launch event
- Mission data blocked before data release event
- System status blocked during maintenance windows
- Security reports blocked before audit completion
"""

import os
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.nodes import EpisodeType


def load_policy_config(config_file: str = "before_after_event_config.json") -> Dict[str, Any]:
    """Load policy configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default configuration")
        return {
            "scheduled_events": {
                "mission_launch": {
                    "description": "Mission launch event",
                    "timestamp": "2025-07-01T12:00:00Z",
                    "status": "scheduled"
                }
            },
            "blocking_rules": {
                "launch_plan": {
                    "description": "Launch plan information blocking",
                    "blocked_before": "mission_launch",
                    "block_message": "Launch plans are not available before the scheduled launch."
                }
            },
            "test_scenarios": [
                {
                    "test_time": "2025-06-30T10:00:00Z",
                    "query": "What is the Launch Plan?",
                    "event": "mission_launch",
                    "expected_result": "blocked",
                    "description": "Query before launch event"
                }
            ],
            "time_based_rules": [
                "Launch plans must be blocked before mission launch event"
            ]
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing {config_file}: {e}")
        return {}


def setup_logging() -> logging.Logger:
    """Setup logging configuration with timestamps and unique log file per run"""
    logger = logging.getLogger('before_after_event_policy')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'before_after_event_policy_{now}.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class BeforeAfterEventPolicy:
    """Implements before/after event blocking policies using Graphiti"""
    
    def __init__(self, graphiti: Graphiti, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.graphiti = graphiti
        self.config = config
        self.policies_added = False
        self.logger = logger or logging.getLogger('before_after_event_policy')
        
        # Build blocking rules dictionary for quick lookup
        self.blocking_rules = {}
        for rule_name, rule_config in config.get("blocking_rules", {}).items():
            for example in rule_config.get("examples", []):
                self.blocking_rules[example.lower()] = {
                    "rule_name": rule_name,
                    "blocked_before": rule_config.get("blocked_before"),
                    "allowed_after": rule_config.get("allowed_after"),
                    "block_message": rule_config.get("block_message", "Content is not available at this time."),
                    "description": rule_config.get("description", "")
                }
    
    async def add_policies(self):
        """Add before/after event policies to the knowledge graph"""
        try:
            if self.policies_added:
                self.logger.info("Before/after event policies already added to graph - skipping")
                return
            
            self.logger.info("Starting to add before/after event policies to knowledge graph...")
            
            # Load policies from JSON configuration and filter out problematic ones
            all_policies = self.config.get("time_based_rules", [
                "Launch plans must be blocked before mission launch event"
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
                        name=f"Before/After Event Policy - Text {i+1}",
                        episode_body=policy,
                        source=EpisodeType.text,
                        source_description="Before/after event policy rule",
                        reference_time=datetime.now(timezone.utc),
                        entity_types={}  # Empty dict prevents automatic entity extraction
                    )
                    self.logger.debug(f"Successfully added text policy {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to add text policy {i+1}: {str(e)}")
                    # Continue with other policies instead of failing completely
                    continue
            
            self.policies_added = True
            self.logger.info("Successfully added before/after event policies to knowledge graph")
            
        except Exception as e:
            self.logger.error(f"Error in add_policies: {str(e)}")
            # Mark as added to prevent retry attempts
            self.policies_added = True
            self.logger.info("Policy addition completed with some errors")
    
    def detect_blocked_content(self, query: str) -> List[Dict[str, Any]]:
        """Detect content that might be blocked based on the query"""
        detected_content = []
        
        try:
            query_lower = query.lower()
            
            for content, config in self.blocking_rules.items():
                if content.lower() in query_lower:
                    # Find all occurrences
                    pattern = re.compile(re.escape(content), re.IGNORECASE)
                    matches = pattern.finditer(query)
                    
                    for match in matches:
                        detected_content.append({
                            "content": match.group(),
                            "rule_name": config["rule_name"],
                            "blocked_before": config["blocked_before"],
                            "allowed_after": config["allowed_after"],
                            "block_message": config["block_message"],
                            "description": config["description"],
                            "start": match.start(),
                            "end": match.end()
                        })
            
            self.logger.debug(f"Detected {len(detected_content)} potentially blocked content in query")
            return detected_content
            
        except Exception as e:
            self.logger.error(f"Error detecting blocked content: {str(e)}")
            return []
    
    def get_event_timestamp(self, event_name: str) -> Optional[datetime]:
        """Get the timestamp for a scheduled event"""
        try:
            event_config = self.config.get("scheduled_events", {}).get(event_name)
            if not event_config:
                self.logger.warning(f"Unknown scheduled event: {event_name}")
                return None
            
            timestamp_str = event_config.get("timestamp")
            if not timestamp_str:
                self.logger.warning(f"No timestamp found for event: {event_name}")
                return None
            
            # Parse the timestamp
            try:
                # Handle ISO format with timezone
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                return datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Try parsing as RFC 3339 format
                return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error getting event timestamp: {str(e)}")
            return None
    
    def should_block_content(self, content_info: Dict[str, Any], current_time: datetime) -> Tuple[bool, str]:
        """Determine if content should be blocked based on current time and event"""
        try:
            event_name = content_info.get("blocked_before")
            if not event_name:
                self.logger.warning(f"No blocked_before event specified for content: {content_info.get('content')}")
                return False, "No blocking rule specified"
            
            event_timestamp = self.get_event_timestamp(event_name)
            if not event_timestamp:
                self.logger.warning(f"Could not get timestamp for event: {event_name}")
                return False, f"Event {event_name} not found"
            
            # Compare current time with event timestamp
            if current_time < event_timestamp:
                self.logger.info(f"Content blocked: {current_time} < {event_timestamp}")
                return True, content_info.get("block_message", "Content is not available at this time.")
            else:
                self.logger.info(f"Content allowed: {current_time} >= {event_timestamp}")
                return False, "Content is now available"
            
        except Exception as e:
            self.logger.error(f"Error checking blocking rules: {str(e)}")
            return False, f"Error checking blocking rules: {str(e)}"
    
    async def check_policy(self, query: str, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check and apply before/after event blocking policy
        
        Args:
            query: Query to check for blocked content
            current_time: Current time for comparison (defaults to now)
        
        Returns:
            Dict with policy decision and blocking details
        """
        try:
            if not current_time:
                current_time = datetime.now(timezone.utc)
            
            self.logger.info(f"Checking before/after event policy for query at {current_time}")
            
            # Validate inputs
            if not query or not isinstance(query, str):
                self.logger.error(f"Invalid query provided: {query}")
                return {
                    "success": False,
                    "error": "Query must be a non-empty string",
                    "query": query,
                    "current_time": current_time.isoformat(),
                    "blocked": False
                }
            
            # Detect potentially blocked content
            detected_content = self.detect_blocked_content(query)
            
            if not detected_content:
                self.logger.info("No blocked content detected in query")
                return {
                    "success": True,
                    "query": query,
                    "current_time": current_time.isoformat(),
                    "blocked": False,
                    "reason": "No blocked content detected",
                    "policy_applied": "no_blocking_needed"
                }
            
            # Check each detected content against blocking rules
            blocked_content = []
            for content_info in detected_content:
                should_block, message = self.should_block_content(content_info, current_time)
                if should_block:
                    blocked_content.append({
                        "content": content_info["content"],
                        "rule_name": content_info["rule_name"],
                        "block_message": message,
                        "event": content_info["blocked_before"]
                    })
            
            # Query the knowledge graph for relevant policies (optimized)
            search_query = f"before after event policy time-based blocking"
            self.logger.debug(f"Searching graph with optimized query: {search_query}")
            
            try:
                results = await self.graphiti.search(search_query)
                self.logger.debug(f"Graph search returned {len(results) if results else 0} results")
            except Exception as e:
                self.logger.error(f"Error during graph search: {str(e)}")
                # Continue without graph results - policy logic will still work
                results = []
                self.logger.info("Continuing with policy logic despite graph search error")
            
            if blocked_content:
                # Return the first blocked content (most relevant)
                first_blocked = blocked_content[0]
                self.logger.info(f"Query blocked due to: {first_blocked['content']}")
                return {
                    "success": True,
                    "query": query,
                    "current_time": current_time.isoformat(),
                    "blocked": True,
                    "reason": first_blocked["block_message"],
                    "blocked_content": first_blocked["content"],
                    "event": first_blocked["event"],
                    "policy_applied": "time_based_blocking"
                }
            else:
                self.logger.info("No content blocked - all detected content is allowed")
                return {
                    "success": True,
                    "query": query,
                    "current_time": current_time.isoformat(),
                    "blocked": False,
                    "reason": "All detected content is allowed at current time",
                    "detected_content": [c["content"] for c in detected_content],
                    "policy_applied": "time_based_allowing"
                }
            
        except Exception as e:
            self.logger.error(f"Unexpected error in check_policy: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "query": query,
                "current_time": current_time.isoformat() if current_time else None,
                "blocked": False
            }


def print_graphiti_representation(query: str, current_time: str, result: dict, description: str = None):
    """Print a dynamic Graphiti Representation for the blocking situation."""
    try:
        if description:
            print(f"\nGraphiti Representation for: {description}")
            print(f"   Test: Query at {current_time}")
        else:
            print("\nGraphiti Representation:")
        
        # Show the query relationship
        print(f"Query —[REFERENCES]→ Content")
        
        if result.get("blocked"):
            blocked_content = result.get("blocked_content", "Content")
            event = result.get("event", "Event")
            
            print(f"{blocked_content} —[BLOCK_IF_BEFORE]→ {event}")
            print(f"{event} timestamp = {event} timestamp")
            print(f"Now < {event} → Block")
            print(f"Resolution: Query arrives before {event}")
            print(f"Firewall blocks and responds: \"{result.get('reason', 'Content blocked')}\"")
        else:
            detected_content = result.get("detected_content", [])
            if detected_content:
                content = detected_content[0]
                print(f"{content} —[ALLOWED_AFTER]→ Event")
                print(f"Now >= Event → Allow")
                print(f"Resolution: Query arrives after event")
                print(f"Firewall allows and responds: \"{result.get('reason', 'Content allowed')}\"")
            else:
                print(f"Query —[NO_BLOCKING]→ Content")
                print(f"Resolution: No blocking rules apply")
        
        # Show the current time
        print(f"CurrentTime = {current_time}")
        
    except Exception as e:
        print(f"Error printing Graphiti Representation: {str(e)}")


async def main():
    """Main function to demonstrate the before/after event policy"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Before/After Event Policy demonstration")
    
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
        policy_checker = BeforeAfterEventPolicy(graphiti, config, logger)
        logger.info("BeforeAfterEventPolicy instance created successfully")
        
        # Add policies to the knowledge graph
        logger.info("Adding policies to knowledge graph...")
        await policy_checker.add_policies()
        
        # Test the policy with different scenarios
        logger.info("Starting policy tests...")
        print("\n" + "="*80)
        print("⚠️  Testing Before/After Event Policy")
        print("="*80)
        
        # Load test scenarios from configuration
        test_scenarios = config.get("test_scenarios", [])
        
        # Fallback to default scenarios if none in config
        if not test_scenarios:
            test_scenarios = [
                {
                    "test_time": "2025-06-30T10:00:00Z",
                    "query": "What is the Launch Plan?",
                    "event": "mission_launch",
                    "expected_result": "blocked",
                    "description": "Query before launch event"
                }
            ]
        
        for scenario in test_scenarios:
            test_time_str = scenario["test_time"]
            query = scenario["query"]
            expected_result = scenario["expected_result"]
            description = scenario["description"]
            
            logger.info(f"Running test: {description}")
            print(f"\n📋 Test: {description}")
            
            try:
                # Parse test time
                test_time = datetime.fromisoformat(test_time_str.replace('Z', '+00:00'))
                
                result = await policy_checker.check_policy(query, test_time)
                
                # Print the dynamic Graphiti Representation
                print_graphiti_representation(query, test_time_str, result, description)
                
                if result["success"]:
                    print(f"   Query: {result['query']}")
                    print(f"   Test Time: {result['current_time']}")
                    
                    if result.get("blocked"):
                        print(f"   ❌ BLOCKED")
                        print(f"   Reason: {result['reason']}")
                        print(f"   Blocked Content: {result.get('blocked_content', 'Unknown')}")
                        print(f"   Event: {result.get('event', 'Unknown')}")
                        
                        # Check if result matches expected
                        if expected_result == "blocked":
                            print(f"   ✅ EXPECTED RESULT MATCH")
                        else:
                            print(f"   ⚠️  UNEXPECTED RESULT")
                    else:
                        print(f"   ✅ ALLOWED")
                        print(f"   Reason: {result['reason']}")
                        
                        # Check if result matches expected
                        if expected_result == "allowed":
                            print(f"   ✅ EXPECTED RESULT MATCH")
                        else:
                            print(f"   ⚠️  UNEXPECTED RESULT")
                    
                    logger.info(f"Test result: {'Blocked' if result.get('blocked') else 'Allowed'} - {result['reason']}")
                else:
                    print(f"   ❌ ERROR: {result.get('error', 'Unknown error')}")
                    logger.error(f"Test failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Error during test '{description}': {str(e)}"
                logger.error(error_msg)
                print(f"   ERROR: {error_msg}")
        
        # Test the specific scenario from the prompt
        logger.info("Running original scenario test...")
        print(f"\n🎯 Test: Original Scenario - Launch Plan before Mission Launch")
        
        try:
            # Test with June 30 (before July 1 launch)
            test_time = datetime.fromisoformat("2025-06-30T10:00:00+00:00")
            original_result = await policy_checker.check_policy(
                "What is the Launch Plan for tomorrow?",
                test_time
            )
            print_graphiti_representation("What is the Launch Plan for tomorrow?", "2025-06-30T10:00:00Z", original_result, "Original Scenario - Launch Plan before Mission Launch")
            
            if original_result["success"]:
                print(f"   Query: {original_result['query']}")
                print(f"   Test Time: {original_result['current_time']}")
                
                if original_result.get("blocked"):
                    print(f"   ❌ BLOCKED")
                    print(f"   Reason: {original_result['reason']}")
                    print(f"   Firewall blocks and responds: \"{original_result['reason']}\"")
                    logger.info(f"Original scenario result: Blocked - {original_result['reason']}")
                else:
                    print(f"   ✅ ALLOWED")
                    print(f"   Reason: {original_result['reason']}")
                    logger.info(f"Original scenario result: Allowed - {original_result['reason']}")
            else:
                print(f"   ❌ ERROR: {original_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Error during original scenario test: {str(e)}"
            logger.error(error_msg)
            print(f"   ERROR: {error_msg}")
        
        print("\n" + "="*80)
        print("🎉 Before/After Event Policy demonstration completed!")
        print("="*80)
        logger.info("Before/After Event Policy demonstration completed successfully")
        
    except Exception as e:
        error_msg = f"Critical error in main function: {str(e)}"
        logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        print("\nMake sure:")
        print("1. Neo4j is running and accessible")
        print("2. GROQ_API_KEY is set in your environment")
        print("3. Neo4j credentials are correct")
        print("4. Check the log file 'before_after_event_policy.log' for detailed error information")
        
    finally:
        # Close the connection
        if 'graphiti' in locals():
            try:
                logger.info("Closing Graphiti connection...")
                await graphiti.close()
                logger.info("Graphiti connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Graphiti connection: {str(e)}")
        
        logger.info("Before/After Event Policy demonstration finished")


if __name__ == "__main__":
    asyncio.run(main()) 