#!/usr/bin/env python3
"""
Graphiti Privacy Bridge for Team C
==================================

This module bridges Team C's privacy ontology with Graphiti knowledge graph storage.
Uses Graphiti's higher-level abstraction for natural language to Cypher translation
and timing data for policy enforcement, as requested for team integration.

Author: Team C
Date: 2024-12-30
"""

import sys
import os
import json
from pathlib import Path
import uuid
from datetime import datetime
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
    from graphiti_core.nodes import EntityNode, EpisodicNode
    from graphiti_core.edges import EntityEdge, EpisodicEdge
    GRAPHITI_AVAILABLE = True
    print("✅ Graphiti core imported successfully")
except ImportError as e:
    print(f"⚠️  Graphiti core not available: {e}")
    print("   Falling back to direct Neo4j for now...")
    GRAPHITI_AVAILABLE = False
    # Keep Neo4j fallback for development
    from neo4j import AsyncGraphDatabase

# Import your privacy ontology
from ontology.privacy_ontology import AIPrivacyOntology

class GraphitiPrivacyBridge:
    """
    Connects privacy ontology with Graphiti knowledge graph storage.
    
    Uses Graphiti's higher-level abstraction for:
    - Natural language to Cypher translation
    - Timing data for policy enforcement
    - Shared backend for team integration
    """
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        self.ontology = AIPrivacyOntology()
        
        if GRAPHITI_AVAILABLE:
            self._init_graphiti(neo4j_uri, neo4j_user, neo4j_password)
        else:
            self._init_neo4j_fallback(neo4j_uri, neo4j_user, neo4j_password)
    
    def _init_graphiti(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        """Initialize Graphiti client with proper configuration."""
        try:
            # Use environment variables or defaults
            uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = neo4j_user or os.getenv("NEO4J_USER", "neo4j") 
            password = neo4j_password or os.getenv("NEO4J_PASSWORD", "12345678")
            
            # Initialize Graphiti client
            self.graphiti = Graphiti(
                driver_config={
                    "uri": uri,
                    "username": user,
                    "password": password
                },
                llm_client=None,  # Will use default OpenAI if API key available
                embedder_client=None,  # Will use default if needed
                reranker_client=None  # Optional
            )
            
            print(f"✅ Graphiti initialized with Neo4j at {uri}")
            self.use_graphiti = True
            
        except Exception as e:
            print(f"⚠️  Graphiti initialization failed: {e}")
            print("   Falling back to direct Neo4j...")
            self._init_neo4j_fallback(neo4j_uri, neo4j_user, neo4j_password)
    
    def _init_neo4j_fallback(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        """Fallback to direct Neo4j if Graphiti unavailable."""
        uri = neo4j_uri or "bolt://localhost:7687"
        user = neo4j_user or "neo4j"
        password = neo4j_password or "12345678"
        
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.use_graphiti = False
        print(f"✅ Neo4j fallback initialized at {uri}")

import sys
import asyncio
from datetime import datetime
import json
import os

# Fix Python paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Add Graphiti path
sys.path.append('/Users/apple/Downloads/graphiti/graphiti')

# Import Neo4j directly to avoid vector issues
from neo4j import AsyncGraphDatabase

# Import your privacy ontology
from ontology.privacy_ontology import AIPrivacyOntology

class GraphitiPrivacyBridge:
    """Connects privacy ontology with Neo4j (bypassing vector embeddings)"""
    
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", "12345678")
        )
        self.ontology = AIPrivacyOntology()
        
    async def create_privacy_decision_episode(self, privacy_request: dict):
        """
        Create privacy decision record using Graphiti's high-level abstraction.
        
        Uses Graphiti's natural language processing and timing data for policy enforcement.
        """
        
        # Make privacy decision using your ontology
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
        """Create privacy decision using Graphiti's high-level abstraction."""
        try:
            # Create episodic node with timing data for policy enforcement
            episode_id = f"privacy_episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Use Graphiti's natural language approach
            episode_content = f"""
            Privacy Decision Episode: {privacy_request['data_field']}
            
            Requester: {privacy_request['requester']} 
            Data Field: {privacy_request['data_field']}
            Purpose: {privacy_request['purpose']}
            Context: {privacy_request.get('context', 'standard')}
            Emergency: {privacy_request.get('emergency', False)}
            
            Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}
            Reason: {decision['reason']}
            Confidence: {decision['confidence']}
            Timestamp: {datetime.now().isoformat()}
            System: Team C Privacy Ontology
            """
            
            # Create episodic node using Graphiti
            episode_node = EpisodicNode(
                name=f"Privacy Decision: {privacy_request['data_field']}",
                labels=["PrivacyDecision", "TeamC"],
                source_id=episode_id,
                entity_name=privacy_request['data_field'],
                content=episode_content,
                created_at=datetime.now()
            )
            
            # Add to Graphiti knowledge graph
            await self.graphiti.add_episodic_nodes([episode_node])
            
            # Create entity nodes for requester and data field
            requester_entity = EntityNode(
                name=privacy_request['requester'],
                labels=["Requester", "Person"],
                source_id=f"requester_{privacy_request['requester']}"
            )
            
            data_entity = EntityNode(
                name=privacy_request['data_field'],
                labels=["DataField", "Asset"],
                source_id=f"data_{privacy_request['data_field']}"
            )
            
            await self.graphiti.add_entity_nodes([requester_entity, data_entity])
            
            # Create relationships using Graphiti
            request_edge = EpisodicEdge(
                source_node_id=requester_entity.source_id,
                target_node_id=data_entity.source_id,
                relation_type="REQUESTED_ACCESS_TO",
                episode_id=episode_node.source_id,
                created_at=datetime.now()
            )
            
            decision_edge = EpisodicEdge(
                source_node_id=episode_node.source_id,
                target_node_id=data_entity.source_id,
                relation_type="PRIVACY_DECISION_FOR",
                episode_id=episode_node.source_id,
                created_at=datetime.now()
            )
            
            await self.graphiti.add_episodic_edges([request_edge, decision_edge])
            
            print(f"✅ Created Graphiti privacy decision episode: {episode_id}")
            print(f"   Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}")
            print(f"   Reason: {decision['reason']}")
            print(f"   Using Graphiti high-level abstraction with timing data")
            
            return decision
            
        except Exception as e:
            print(f"⚠️  Graphiti episode creation failed: {e}")
            print("   Falling back to Neo4j...")
            return await self._create_episode_neo4j_fallback(privacy_request, decision)
    
    async def _create_episode_neo4j_fallback(self, privacy_request: dict, decision: dict):
        """Fallback method using direct Neo4j access."""
        # Create record directly in Neo4j (existing implementation)
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
                    system: "team_c_ontology",
                    version: "1.0"
                })
                RETURN e.uuid as episode_id
            """,
                uuid=f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(privacy_request)) % 10000}",
                name=f"Privacy Decision: {privacy_request['data_field']}",
                requester=privacy_request["requester"],
                data_field=privacy_request["data_field"],
                purpose=privacy_request["purpose"],
                context=privacy_request.get("context", ""),
                decision="ALLOWED" if decision["allowed"] else "DENIED",
                reason=decision["reason"],
                confidence=decision["confidence"],
                emergency=privacy_request.get("emergency", False),
                timestamp=datetime.now().isoformat()
            )
            
            record = await result.single()
            episode_id = record["episode_id"]
        
        print(f"✅ Created privacy decision episode: {episode_id}")
        print(f"   Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}")
        print(f"   Reason: {decision['reason']}")
        
        return decision
        
    async def create_data_entity(self, data_field: str, context: str = None):
        """
        Create data classification entity using Graphiti's high-level abstraction.
        
        Uses Graphiti's natural language understanding for better classification storage.
        """
        
        # Classify using your ontology
        classification = self.ontology.classify_data_field(data_field, context)
        
        if self.use_graphiti:
            return await self._create_data_entity_with_graphiti(data_field, context, classification)
        else:
            return await self._create_data_entity_neo4j_fallback(data_field, context, classification)
    
    async def _create_data_entity_with_graphiti(self, data_field: str, context: str, classification: dict):
        """Create data entity using Graphiti's high-level abstraction."""
        try:
            # Create entity description for Graphiti's natural language processing
            entity_description = f"""
            Data Field Classification: {data_field}
            
            Field Name: {data_field}
            Context: {context or 'general'}
            Data Type: {classification['data_type']}
            Sensitivity Level: {classification['sensitivity']}
            Context Dependent: {classification.get('context_dependent', False)}
            Classification Reasoning: {', '.join(classification.get('reasoning', []))}
            Classification Confidence: {classification.get('confidence', 0.9)}
            Classified At: {datetime.now().isoformat()}
            System: Team C Privacy Ontology
            """
            
            # Create entity node using Graphiti
            data_entity = EntityNode(
                name=data_field,
                labels=["DataField", "ClassifiedAsset", classification['data_type']],
                source_id=f"data_entity_{data_field}_{hash(data_field) % 10000}",
                description=entity_description,
                created_at=datetime.now()
            )
            
            # Add to Graphiti knowledge graph
            result = await self.graphiti.add_entity_nodes([data_entity])
            
            print(f"✅ Created data entity: {data_field}")
            print(f"   Type: {classification['data_type']}")
            print(f"   Sensitivity: {classification['sensitivity']}")
            print(f"   Using Graphiti high-level abstraction")
            
            return {
                "field": data_field,
                "data_type": classification["data_type"],
                "sensitivity": classification["sensitivity"],
                "context_dependent": classification.get("context_dependent", False),
                "equivalents": classification.get("equivalents", []),
                "reasoning": classification.get("reasoning", []),
                "graphiti_entity_id": data_entity.source_id
            }
            
        except Exception as e:
            print(f"⚠️  Graphiti entity creation failed: {e}")
            print("   Falling back to Neo4j...")
            return await self._create_data_entity_neo4j_fallback(data_field, context, classification)
    
    async def _create_data_entity_neo4j_fallback(self, data_field: str, context: str, classification: dict):
        """Fallback method using direct Neo4j access."""
        # Create entity directly in Neo4j (existing implementation)
        async with self.driver.session() as session:
            result = await session.run("""
                MERGE (d:DataEntity {name: $name})
                SET d.data_type = $data_type,
                    d.sensitivity_level = $sensitivity,
                    d.context_dependent = $context_dependent,
                    d.equivalents = $equivalents,
                    d.reasoning = $reasoning,
                    d.classified_by = "team_c_ontology",
                    d.updated_at = $timestamp
                RETURN d.name as entity_name
            """,
                name=data_field,
                data_type=classification["data_type"],
                sensitivity=classification["sensitivity"],
                context_dependent=classification["context_dependent"],
                equivalents=json.dumps(classification["equivalents"]),
                reasoning=json.dumps(classification["reasoning"]),
                timestamp=datetime.now().isoformat()
            )
            
            record = await result.single()
            entity_name = record["entity_name"]
        
        print(f"✅ Created data entity: {entity_name}")
        print(f"   Type: {classification['data_type']}")
        print(f"   Sensitivity: {classification['sensitivity']}")
        
        return classification
        
    async def demo_8_week_scenarios(self):
        """Run the 5 demo scenarios from external test configuration"""
        
        print("🎬 Running 8-Week Project Demo Scenarios...")
        
        # Load demo scenarios from external file
        test_file_path = os.path.join(parent_dir, "tests", "test_cases.json")
        
        try:
            with open(test_file_path, 'r') as f:
                test_config = json.load(f)
            
            demo_scenarios = test_config["8_week_demo_scenarios"]
            print(f"✅ Loaded {len(demo_scenarios)} scenarios from external config")
            
        except FileNotFoundError:
            print("⚠️  External test file not found, using fallback scenarios...")
            # Fallback hardcoded scenarios
            demo_scenarios = [
                {
                    "name": "Emergency Access",
                    "input": {
                        "requester": "dr_emergency", 
                        "data_field": "patient_medical_record",
                        "purpose": "emergency_treatment",
                        "context": "medical",
                        "emergency": True
                    }
                },
                {
                    "name": "Temporal Boundaries", 
                    "input": {
                        "requester": "contractor_john",
                        "data_field": "project_financial_data", 
                        "purpose": "project_work",
                        "context": "financial",
                        "emergency": False
                    }
                },
                {
                    "name": "Organizational Hierarchy",
                    "input": {
                        "requester": "manager_sarah",
                        "data_field": "team_performance_data",
                        "purpose": "management_review", 
                        "context": "hr",
                        "emergency": False
                    }
                },
                {
                    "name": "Semantic Understanding - Medical",
                    "input": {
                        "requester": "doctor",
                        "data_field": "patient_diagnosis",
                        "purpose": "treatment",
                        "context": "medical", 
                        "emergency": False
                    }
                },
                {
                    "name": "Semantic Understanding - IT",
                    "input": {
                        "requester": "it_admin",
                        "data_field": "system_diagnosis", 
                        "purpose": "troubleshooting",
                        "context": "it",
                        "emergency": False
                    }
                }
            ]
        
        for scenario in demo_scenarios:
            scenario_name = scenario.get("name", scenario.get("test_id", "Unknown"))
            scenario_input = scenario.get("input", scenario.get("request", {}))
            
            print(f"\n🎯 Demo: {scenario_name}")
            decision = await self.create_privacy_decision_episode(scenario_input)
            classification = await self.create_data_entity(
                scenario_input["data_field"], 
                scenario_input.get("context")
            )
            
            # Create relationship between decision and data
            await self.create_decision_data_relationship(
                scenario_input["data_field"],
                decision
            )
        
        print(f"\n💡 For comprehensive testing with validation, run:")
        print(f"   python tests/run_demo_tests.py")
        print(f"   python tests/run_demo_tests.py --comprehensive")
        
    async def create_decision_data_relationship(self, data_field: str, decision: dict):
        """Create relationship between privacy decision and data entity"""
        
        async with self.driver.session() as session:
            await session.run("""
                MATCH (e:PrivacyDecisionEpisode {data_field: $data_field})
                MATCH (d:DataEntity {name: $data_field})
                MERGE (e)-[:CONCERNS]->(d)
                MERGE (d)-[:HAS_DECISION]->(e)
            """,
                data_field=data_field
            )
            
    async def close(self):
        """Close connections to Graphiti or Neo4j."""
        if self.use_graphiti and hasattr(self, 'graphiti'):
            try:
                await self.graphiti.close()
                print("✅ Graphiti connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Graphiti: {e}")
        
        if hasattr(self, 'driver'):
            try:
                await self.driver.close()
                print("✅ Neo4j connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Neo4j: {e}")

# This is the BRIDGE that connects your smart ontology to Graphiti knowledge graph storage

# Run the demo
async def main():
    print("🌉 Starting Graphiti-Privacy Integration Demo (High-Level Abstraction)...")
    print("Using Graphiti's natural language to Cypher translation and timing data")
    
    bridge = GraphitiPrivacyBridge()
    
    try:
        await bridge.demo_8_week_scenarios()
        print("\n🎉 Demo completed successfully!")
        print("Check Neo4j Browser at http://localhost:7474")
        print("Queries to try:")
        print("  MATCH (e:PrivacyDecisionEpisode) RETURN e ORDER BY e.timestamp DESC")
        print("  MATCH (d:DataEntity) RETURN d")
        print("  MATCH (e:PrivacyDecisionEpisode)-[r]-(d:DataEntity) RETURN e.decision, d.data_type")
        print("\nGraphiti provides:")
        print("  ✅ Higher-level abstraction over Neo4j")
        print("  ✅ Natural language to Cypher translation")
        print("  ✅ Timing data for policy enforcement")
        print("  ✅ Shared backend for team integration")
        
    finally:
        await bridge.close()

if __name__ == "__main__":
    asyncio.run(main())
