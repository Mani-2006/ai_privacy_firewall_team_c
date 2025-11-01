#!/usr/bin/env python3
"""
Enhanced Graphiti Privacy Bridge with Timezone Awareness
======================================================

This module bridges Team C's privacy ontology with Graphiti knowledge graph storage.
Uses Graphiti's higher-level abstraction for natural language to Cypher translation
with proper timestamp handling and timezone awareness for global team integration.

Key Features:
- Timezone-aware timestamp formatting for Graphiti LLM processing
- Business hours consideration for policy enforcement
- Natural language episode content for better LLM translation
- Proper ISO 8601 timestamp formatting with Z suffix

Author: Team C Privacy Firewall
Date: 2024-12-30
"""

import sys
import os
import json
from pathlib import Path
import uuid
from datetime import datetime, timezone
import asyncio
from typing import Dict, List, Any, Optional

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Add Graphiti core path
graphiti_path = str(Path(__file__).parent.parent.parent / "graphiti_core")
sys.path.append(graphiti_path)

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.utils.datetime_utils import utc_now
    GRAPHITI_AVAILABLE = True
    print("✅ Graphiti core imported successfully")
except ImportError as e:
    print(f"⚠️  Graphiti core not available: {e}")
    print("   Falling back to direct Neo4j for now...")
    GRAPHITI_AVAILABLE = False

# Always import Neo4j for fallback
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Neo4j driver not available: {e}")
    NEO4J_AVAILABLE = False

# Import privacy ontology and timezone utilities
from ontology.privacy_ontology import AIPrivacyOntology
from integration.timezone_utils import TimezoneHandler
from integration.groq_llm_client import GroqLLMClient, GroqConfig, GroqClientManager

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not available, using environment variables directly")

class EnhancedGraphitiPrivacyBridge:
    """
    Enhanced privacy bridge with timezone awareness and proper timestamp handling.
    
    Uses Graphiti's higher-level abstraction with LLM-powered natural language
    to Cypher translation, ensuring proper temporal data for policy enforcement.
    """
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", neo4j_password="12345678"):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.ontology = AIPrivacyOntology()
        self.groq_client = None
        
        # Initialize Groq LLM client if API key available
        self._init_groq_client()
        
        if GRAPHITI_AVAILABLE:
            self._init_graphiti()
        else:
            self._init_neo4j_fallback()
    
    def _init_groq_client(self):
        """Initialize Groq LLM client for privacy decisions."""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                config = GroqConfig(
                    api_key=groq_api_key,
                    model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),  # Full Llama, not instant
                    temperature=0.1  # Consistent privacy decisions
                )
                self.groq_client = GroqLLMClient(config)
                print(f"✅ Groq LLM initialized with {config.model}")
                print("   Using Llama for privacy decision intelligence")
            else:
                print("⚠️  GROQ_API_KEY not found, using fallback decision logic")
        except Exception as e:
            print(f"⚠️  Groq initialization failed: {e}")
            print("   Using fallback decision logic")
    
    def _init_graphiti(self):
        """Initialize Graphiti with Groq masquerading as OpenAI."""
        try:
            # Check if we have the API key (now aliased as OPENAI_API_KEY)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  No OPENAI_API_KEY found, falling back to Neo4j")
                self._init_neo4j_fallback()
                return
            
            # Set Groq's base URL for OpenAI compatibility
            os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
            
            # Initialize Graphiti - it will use the Groq API endpoint
            self.graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password
            )
            self.use_graphiti = True
            print(f"✅ Graphiti initialized at {self.neo4j_uri}")
            print(f"   Using Groq Llama 3 70B via OpenAI-compatible API")
            print(f"   Base URL: https://api.groq.com/openai/v1")
            print(f"   API Key: {api_key[:20]}...")
        except Exception as e:
            print(f"⚠️  Graphiti initialization failed: {e}")
            print("   Falling back to Neo4j...")
            self._init_neo4j_fallback()
    
    def _init_neo4j_fallback(self):
        """Initialize Neo4j fallback for development."""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available and Graphiti initialization failed")
            
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.use_graphiti = False
        print(f"✅ Neo4j fallback initialized at {self.neo4j_uri}")
    
    async def create_privacy_decision_episode(self, privacy_request: dict):
        """
        Create privacy decision record with timezone-aware timestamps.
        
        Uses Graphiti's natural language processing and timing data for policy enforcement.
        Includes business hours and location context for global team integration.
        """
        
        # Make privacy decision using ontology
        decision = self.ontology.make_privacy_decision(
            requester=privacy_request["requester"],
            data_field=privacy_request["data_field"], 
            purpose=privacy_request["purpose"],
            context=privacy_request.get("context"),
            emergency=privacy_request.get("emergency", False)
        )
        
        if self.use_graphiti:
            return await self._create_episode_with_graphiti(privacy_request, decision)
        else:
            return await self._create_episode_neo4j_fallback(privacy_request, decision)
    
    async def _create_episode_with_graphiti(self, privacy_request: dict, decision: dict):
        """
        Create privacy decision episode using Graphiti's high-level abstraction.
        
        Uses natural language content with proper timestamp formatting for LLM-powered Cypher translation.
        Includes timezone awareness for global team integration.
        """
        try:
            episode_id = str(uuid.uuid4())
            
            # Get timezone-aware timestamp using Graphiti's datetime utilities
            current_time = utc_now() if GRAPHITI_AVAILABLE else TimezoneHandler.get_current_utc()
            requester_location = privacy_request.get('requester_location', 'utc')
            
            # Create properly formatted episode content following conversation pattern
            # This follows the shoe_conversation examples you provided
            formatted_timestamp = TimezoneHandler.format_for_graphiti(current_time, requester_location)
            
            episode_content = f"""PrivacyBot ({formatted_timestamp}): Privacy decision processed for data access request.

Requester ({formatted_timestamp}): {privacy_request['requester']} requested access to {privacy_request['data_field']} for {privacy_request['purpose']}

PrivacyBot ({formatted_timestamp}): Decision: {'ALLOWED' if decision.get('allowed', False) else 'DENIED'}
Reason: {decision.get('reason', 'No reason provided')}
Confidence: {decision.get('confidence', 0.0)}
Context: {privacy_request.get('context', 'General request')}
Emergency Override: {'Active' if privacy_request.get('emergency', False) else 'None'}

BusinessContext ({formatted_timestamp}): {TimezoneHandler.get_business_context(requester_location, current_time)}"""
            
            # Create EpisodicNode with timezone-aware timing data
            episode_node = EpisodicNode(
                name=f"Privacy Decision: {privacy_request['data_field']} at {formatted_timestamp}",
                content=episode_content,
                labels=["PrivacyDecision", "TeamC", "TimezoneAware"],
                uuid=episode_id,
                group_id="team_c_privacy",
                source=EpisodeType.message if GRAPHITI_AVAILABLE else "message",
                source_description="Team C Privacy Firewall Decision",
                created_at=current_time,
                valid_at=current_time  # When this privacy decision was made
            )
            
            # Add episode to Graphiti
            await self.graphiti.add_episodic_nodes([episode_node])
            
            # Create data entity for the requested field
            data_classification = self.ontology.classify_data_field(
                privacy_request["data_field"],
                privacy_request.get("context")
            )
            
            await self._create_data_entity_with_graphiti(
                privacy_request["data_field"], 
                data_classification,
                current_time
            )
            
            print(f"✅ Created Graphiti privacy decision episode: {episode_id}")
            print(f"   Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}")
            print(f"   Timestamp: {formatted_timestamp}")
            print(f"   Location context: {requester_location}")
            print(f"   Using Graphiti high-level abstraction with timing data")
            
            return decision
            
        except Exception as e:
            print(f"⚠️  Graphiti episode creation failed: {e}")
            print("   Falling back to Neo4j...")
            return await self._create_episode_neo4j_fallback(privacy_request, decision)
    
    async def _create_data_entity_with_graphiti(self, data_field: str, classification: dict, timestamp: datetime):
        """
        Create data entity using Graphiti's EntityNode abstraction.
        
        Uses timezone-aware descriptive content for LLM understanding and proper temporal tracking.
        """
        try:
            entity_id = str(uuid.uuid4())
            
            # Create descriptive entity content with timestamp following conversation pattern
            formatted_timestamp = TimezoneHandler.format_for_graphiti(timestamp)
            
            entity_summary = f"""DataClassifier ({formatted_timestamp}): Classified data field '{data_field}'

Classification Results ({formatted_timestamp}):
- Data Type: {classification.get('data_type', 'Unknown')}
- Sensitivity Level: {classification.get('sensitivity_level', 'Unknown')} 
- PII Status: {'Contains PII' if classification.get('is_pii', False) else 'No PII detected'}
- Confidence: {classification.get('confidence', 0.0)}
- Reasoning: {classification.get('reasoning', 'Automated classification')}

SystemNote ({formatted_timestamp}): This data asset has been processed by Team C's Privacy Ontology for access control and policy enforcement."""
            
            # Create EntityNode with timezone-aware descriptive content
            data_entity = EntityNode(
                name=f"{data_field}",
                summary=entity_summary,
                labels=["DataField", "ClassifiedAsset", "TimezoneAware", classification.get('data_type', 'Unknown')],
                uuid=entity_id,
                group_id="team_c_privacy",
                created_at=timestamp
            )
            
            # Add entity to Graphiti
            await self.graphiti.add_entity_nodes([data_entity])
            
            print(f"✅ Created Graphiti data entity: {data_field}")
            
        except Exception as e:
            print(f"⚠️  Graphiti data entity creation failed: {e}")
    
    async def _create_episode_neo4j_fallback(self, privacy_request: dict, decision: dict):
        """Fallback method using direct Neo4j access with timezone awareness."""
        current_time = TimezoneHandler.get_current_utc()
        formatted_timestamp = TimezoneHandler.format_for_graphiti(current_time)
        
        async with self.driver.session() as session:
            result = await session.run("""
                CREATE (e:PrivacyDecisionEpisode {
                    uuid: $uuid,
                    name: $name,
                    requester: $requester,
                    data_field: $data_field,
                    purpose: $purpose,
                    context: $context,
                    decision: $decision,
                    reason: $reason,
                    confidence: $confidence,
                    emergency: $emergency,
                    timestamp: $timestamp,
                    iso_timestamp: $iso_timestamp,
                    created_at: datetime($created_at),
                    team: 'C'
                })
                RETURN e.uuid as episode_id
            """, 
                uuid=str(uuid.uuid4()),
                name=f"Privacy Decision: {privacy_request['data_field']}",
                requester=privacy_request["requester"],
                data_field=privacy_request["data_field"],
                purpose=privacy_request["purpose"],
                context=privacy_request.get("context", ""),
                decision="ALLOWED" if decision["allowed"] else "DENIED",
                reason=decision["reason"],
                confidence=decision["confidence"],
                emergency=privacy_request.get("emergency", False),
                timestamp=formatted_timestamp,
                iso_timestamp=current_time.isoformat(),
                created_at=current_time.isoformat()
            )
            
            print(f"✅ Created Neo4j privacy decision (fallback)")
            print(f"   Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}")
            print(f"   Timestamp: {formatted_timestamp}")
            
            return decision
    
    async def classify_data_field(self, data_field: str, context: str = None):
        """
        Classify data field using Groq LLM intelligence or fallback logic.
        
        Args:
            data_field: The data field to classify
            context: Additional context about the data
            
        Returns:
            Classification result with data type and sensitivity
        """
        if self.groq_client:
            try:
                # Use Groq Llama for intelligent classification
                return await self.groq_client.classify_data_field(data_field, context)
            except Exception as e:
                print(f"⚠️  Groq classification failed: {e}")
                print("   Using fallback classification logic")
        
        # Fallback to ontology-based classification with timezone tracking
        current_time = TimezoneHandler.get_current_utc()
        classification = self.ontology.classify_data_field(data_field, context)
        
        if self.use_graphiti:
            await self._create_data_entity_with_graphiti(data_field, classification, current_time)
        
        return classification
    
    async def make_enhanced_privacy_decision(self, privacy_request: dict):
        """
        Make privacy decision using Groq LLM intelligence with timezone awareness.
        
        Combines Groq Llama reasoning with timezone-aware business logic
        for comprehensive privacy decisions.
        """
        if self.groq_client:
            try:
                # Use Groq for intelligent decision making
                llm_decision = await self.groq_client.make_privacy_decision(
                    requester=privacy_request["requester"],
                    data_field=privacy_request["data_field"],
                    purpose=privacy_request["purpose"],
                    context=privacy_request.get("context", ""),
                    emergency=privacy_request.get("emergency", False)
                )
                
                # Add data classification
                classification = await self.classify_data_field(
                    privacy_request["data_field"], 
                    privacy_request.get("context")
                )
                
                return {
                    "allowed": llm_decision["allowed"],
                    "reason": llm_decision["reason"],
                    "confidence": llm_decision["confidence"],
                    "data_classification": classification,
                    "emergency_used": llm_decision["emergency_used"],
                    "integration_ready": True,
                    "llm_powered": True
                }
                
            except Exception as e:
                print(f"⚠️  Groq decision failed: {e}")
                print("   Using fallback decision logic")
        
        # Fallback to ontology-based decision
        decision = self.ontology.make_privacy_decision(
            requester=privacy_request["requester"],
            data_field=privacy_request["data_field"],
            purpose=privacy_request["purpose"],
            context=privacy_request.get("context"),
            emergency=privacy_request.get("emergency", False)
        )
        
        classification = self.ontology.classify_data_field(
            privacy_request["data_field"], 
            privacy_request.get("context")
        )
        
        return {
            "allowed": decision["allowed"],
            "reason": decision["reason"],
            "confidence": decision["confidence"],
            "data_classification": classification,
            "emergency_used": decision.get("emergency_used", False),
            "integration_ready": True,
            "llm_powered": False
        }
    
    async def close(self):
        """Close connections properly."""
        # Close Groq client first
        if self.groq_client:
            try:
                await self.groq_client.close()
                print("✅ Groq client closed")
            except Exception as e:
                print(f"⚠️  Error closing Groq client: {e}")
        
        # Then close database connections
        if self.use_graphiti:
            try:
                await self.graphiti.close()
                print("✅ Graphiti connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Graphiti: {e}")
        else:
            try:
                await self.driver.close()
                print("✅ Neo4j connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Neo4j: {e}")

# Create instance for backward compatibility
GraphitiPrivacyBridge = EnhancedGraphitiPrivacyBridge