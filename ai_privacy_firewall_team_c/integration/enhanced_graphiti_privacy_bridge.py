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
# Note: Removed Groq imports - now using OpenAI directly

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
                 neo4j_user="neo4j", neo4j_password="12345678", 
                 openai_api_key=None):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.ontology = AIPrivacyOntology()
        self.openai_client = None
        
        # Initialize OpenAI LLM client if API key available
        self._init_openai_client(openai_api_key)
        
        if GRAPHITI_AVAILABLE:
            self._init_graphiti()
        else:
            self._init_neo4j_fallback()
    
    def _init_openai_client(self, openai_api_key=None):
        """Initialize OpenAI LLM client for privacy decisions."""
        try:
            # Get API key from parameter or environment
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            
            if api_key:
                # Set environment variable for Graphiti to use
                os.environ["OPENAI_API_KEY"] = api_key
                
                # Set model from environment or default
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                os.environ["OPENAI_MODEL"] = model
                
                print(f"✅ OpenAI API configured with {model}")
                print(f"   Key: {api_key[:20]}...")
                print("   Using OpenAI for privacy decision intelligence")
                self.openai_enabled = True
            else:
                print("⚠️  OPENAI_API_KEY not found, using fallback decision logic")
                self.openai_enabled = False
        except Exception as e:
            print(f"⚠️  OpenAI initialization failed: {e}")
            print("   Using fallback decision logic")
            self.openai_enabled = False
    
    def _init_graphiti(self):
        """Initialize Graphiti with OpenAI."""
        try:
            # Check if we have the OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  No OPENAI_API_KEY found, falling back to Neo4j")
                self._init_neo4j_fallback()
                return
            
            # Get Neo4j password from environment
            neo4j_password = os.getenv('NEO4J_PASSWORD', self.neo4j_password)
            
            # Initialize Graphiti with OpenAI (no custom client needed)
            self.graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=neo4j_password
            )
            self.use_graphiti = True
            
            model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            print(f"✅ Graphiti initialized with OpenAI at {self.neo4j_uri}")
            print(f"   Using OpenAI {model} for LLM reasoning")
            print(f"   Neo4j password: {neo4j_password}")
            
            # Also initialize Neo4j driver for fallback scenarios
            if NEO4J_AVAILABLE:
                self.driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, neo4j_password)
                )
                print(f"✅ Neo4j fallback driver initialized")
            
        except Exception as e:
            print(f"❌ Graphiti initialization failed: {e}")
            print("   Falling back to direct Neo4j")
            self._init_neo4j_fallback()
            print(f"   Model: llama-3.3-70b-versatile")
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
        
        # Make privacy decision using LLM if available, fallback to ontology
        print(f"🔍 DEBUG: openai_enabled = {self.openai_enabled}")
        print(f"🔍 DEBUG: privacy_request = {privacy_request}")
        
        if self.openai_enabled:
            try:
                print("🔍 DEBUG: Calling make_enhanced_privacy_decision")
                decision = await self.make_enhanced_privacy_decision(privacy_request)
                print(f"✅ LLM-powered decision: {'ALLOW' if decision['allowed'] else 'DENY'}")
                print(f"🔍 DEBUG: LLM decision = {decision}")
            except Exception as e:
                print(f"⚠️  LLM decision failed: {e}, falling back to rule-based")
                import traceback
                traceback.print_exc()
                decision = self.ontology.make_privacy_decision(
                    requester=privacy_request["requester"],
                    data_field=privacy_request["data_field"], 
                    purpose=privacy_request["purpose"],
                    context=privacy_request.get("context"),
                    emergency=privacy_request.get("emergency", False)
                )
        else:
            # Fallback to rule-based decision
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
            
            # Add episode to Graphiti using correct API (let Graphiti generate UUID)
            result = await self.graphiti.add_episode(
                name=f"Privacy Decision: {privacy_request['data_field']} at {formatted_timestamp}",
                episode_body=episode_content,
                source_description="Team C Privacy Firewall Decision",
                reference_time=current_time,
                source=EpisodeType.message if GRAPHITI_AVAILABLE else "message",
                group_id="team_c_privacy"
            )
            
            print(f"✅ Created Graphiti privacy decision episode: {result.episode_uuid if hasattr(result, 'episode_uuid') else 'generated'}")
            print(f"   Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}")
            print(f"   LLM-powered reasoning stored in Graphiti knowledge graph")
            print(f"   Timestamp: {formatted_timestamp}")
            print(f"   Using LLM + Graphiti integration (no fallback needed)")
            
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
        if self.openai_enabled:
            try:
                # Use OpenAI for intelligent classification via Graphiti
                # Note: For now using fallback since we don't have direct OpenAI client
                print("⚠️  Direct OpenAI classification not implemented yet")
                print("   Using fallback classification logic")
            except Exception as e:
                print(f"⚠️  OpenAI classification failed: {e}")
                print("   Using fallback classification logic")
        
        # Fallback to ontology-based classification with timezone tracking
        current_time = TimezoneHandler.get_current_utc()
        classification = self.ontology.classify_data_field(data_field, context)
        
        # Note: Entity relationships will be created when episode is added
        return classification
    
    async def make_enhanced_privacy_decision(self, privacy_request: dict):
        """
        Make privacy decision using REAL OpenAI LLM intelligence.
        
        Uses actual OpenAI API calls instead of hardcoded rules.
        """
        print("🧠 Making REAL LLM-powered privacy decision via OpenAI API")
        
        try:
            # Import OpenAI for direct API calls
            from openai import AsyncOpenAI
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("No OpenAI API key found")
            
            # Create OpenAI client
            client = AsyncOpenAI(api_key=api_key)
            
            # Prepare the prompt for OpenAI
            prompt = f"""You are an AI Privacy Expert making access control decisions. Analyze this request and respond with a JSON decision.

REQUEST DETAILS:
- Requester: {privacy_request.get('requester', 'unknown')}
- Data Field: {privacy_request.get('data_field', 'unknown')}
- Purpose: {privacy_request.get('purpose', 'unknown')}
- Context: {privacy_request.get('context', 'unknown')}
- Emergency: {privacy_request.get('emergency', False)}

DECISION CRITERIA:
- Medical data should only be accessible to medical professionals or in emergencies
- Financial data should only be accessible to authorized financial personnel or auditors
- Personal data should have appropriate access controls
- Emergency situations may override normal restrictions
- Contractors/temporary staff should have limited access

Respond with a JSON object containing:
{{
  "allowed": true/false,
  "reasoning": "detailed explanation of the decision",
  "confidence": 0.0-1.0,
  "data_sensitivity": "low/medium/high/critical"
}}"""

            print("📡 Making OpenAI API call for privacy decision...")
            
            # Make real OpenAI API call
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert AI privacy decision system."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent decisions
                max_tokens=500
            )
            
            # Parse the LLM response
            llm_response = response.choices[0].message.content
            print(f"📡 OpenAI Response: {llm_response}")
            
            import json
            decision_data = json.loads(llm_response)
            
            # Get data classification 
            classification = await self.classify_data_field(
                privacy_request["data_field"], 
                privacy_request.get("context")
            )
            
            print(f"🧠 REAL LLM Decision: {'ALLOW' if decision_data['allowed'] else 'DENY'}")
            print(f"🧠 REAL LLM Reasoning: {decision_data['reasoning']}")
            print(f"🧠 REAL LLM Confidence: {decision_data['confidence']}")
            
            return {
                "allowed": decision_data["allowed"],
                "reason": decision_data["reasoning"],
                "confidence": decision_data["confidence"],
                "data_classification": classification,
                "emergency_used": privacy_request.get("emergency", False),
                "integration_ready": True,
                "llm_powered": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "openai_response": llm_response  # Include raw OpenAI response for verification
            }
            
        except Exception as e:
            print(f"❌ REAL OpenAI LLM call failed: {e}")
            print("   Falling back to ontology-based decision")
            # Fallback to ontology-based decision
            decision = self.ontology.make_privacy_decision(
                requester=privacy_request["requester"],
                data_field=privacy_request["data_field"],
                purpose=privacy_request["purpose"],
                context=privacy_request.get("context"),
                emergency=privacy_request.get("emergency", False)
            )
            return decision
    
    async def close(self):
        """Close connections properly."""
        # Close OpenAI resources if needed
        if self.openai_enabled:
            try:
                # OpenAI client doesn't require explicit closing
                print("✅ OpenAI resources cleaned up")
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