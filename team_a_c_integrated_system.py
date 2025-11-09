#!/usr/bin/env python3
"""
Team A + Team C Integrated Privacy & Temporal Framework
========================================================

This module integrates Team A's temporal framework with Team C's privacy firewall,
creating a comprehensive privacy decision system with temporal context awareness.

Integration Features:
- Team A's 6-tuple temporal context + Team C's privacy ontology intelligence
- Time-aware privacy decisions with emergency overrides
- Combined Neo4j knowledge graph storage
- Enhanced FastAPI service with temporal endpoints
- Unified testing and validation

Author: Team C (Privacy Firewall) + Team A (Temporal Framework)
Date: 2024-12-30
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import asyncio
import logging

# Add Team A's temporal framework to path
TEAM_A_PATH = Path(__file__).parent.parent / "ai_temporal_framework"
sys.path.insert(0, str(TEAM_A_PATH))

# Team C imports (our privacy system)
from ontology.privacy_ontology import AIPrivacyOntology
from integration.enhanced_graphiti_privacy_bridge import EnhancedGraphitiPrivacyBridge

# Team A imports (temporal framework)
try:
    from core.tuples import EnhancedContextualIntegrityTuple, TemporalContext, TimeWindow
    from core.policy_engine import TemporalPolicyEngine
    from core.evaluator import TemporalEvaluator
    from core.enricher import TemporalEnricher
    from core.graphiti_manager import TemporalGraphitiManager, GraphitiConfig
    TEAM_A_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Team A temporal framework not fully available: {e}")
    print("   Running in Team C standalone mode...")
    TEAM_A_AVAILABLE = False

class IntegratedPrivacyTemporalSystem:
    """
    Integrated system combining Team A's temporal framework with Team C's privacy firewall.
    
    This class provides:
    1. Enhanced privacy decisions with temporal context
    2. Emergency override capabilities
    3. Time-aware access control
    4. Combined audit logging
    5. Unified Neo4j knowledge graph storage
    """
    
    def __init__(
        self, 
        neo4j_uri: str = "bolt://ssh.phorena.com:57687",
        neo4j_user: str = "llm_security",
        neo4j_password: Optional[str] = None,
        enable_temporal: bool = True
    ):
        """Initialize integrated system with both Team A and Team C components."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.enable_temporal = enable_temporal and TEAM_A_AVAILABLE
        
        # Initialize Team C privacy components
        self.privacy_ontology = AIPrivacyOntology()
        self.privacy_bridge = EnhancedGraphitiPrivacyBridge(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=self.neo4j_password
        )
        
        # Initialize Team A temporal components (if available)
        self.temporal_manager = None
        self.temporal_policy_engine = None
        self.temporal_evaluator = None
        
        if self.enable_temporal:
            try:
                self._initialize_temporal_components()
                print("✅ Team A + Team C integration successful!")
            except Exception as e:
                print(f"⚠️  Temporal components initialization failed: {e}")
                print("   Running in Team C privacy-only mode...")
                self.enable_temporal = False
        else:
            print("🔒 Running Team C privacy firewall in standalone mode")
    
    def _initialize_temporal_components(self):
        """Initialize Team A's temporal framework components."""
        if not TEAM_A_AVAILABLE:
            raise ImportError("Team A temporal framework not available")
        
        # Set up Graphiti manager for Team A
        config = GraphitiConfig(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
            team_namespace="llm_security"
        )
        
        self.temporal_manager = TemporalGraphitiManager(config)
        self.temporal_policy_engine = TemporalPolicyEngine(graphiti_manager=self.temporal_manager)
        self.temporal_evaluator = TemporalEvaluator(policy_engine=self.temporal_policy_engine)
    
    def make_enhanced_privacy_decision(
        self,
        data_field: str,
        requester_role: str,
        context: str,
        organizational_context: Optional[str] = None,
        temporal_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make enhanced privacy decision combining Team A temporal + Team C privacy intelligence.
        
        Args:
            data_field: The data field being accessed
            requester_role: Role of the person requesting access
            context: Current situational context
            organizational_context: Additional org context (Team C)
            temporal_context: Time-based context (Team A)
        
        Returns:
            Enhanced decision dictionary with temporal and privacy analysis
        """
        print(f"\n🔒 Enhanced Privacy Decision: {data_field}")
        print("=" * 60)
        
        # Step 1: Team C Privacy Classification and Decision
        privacy_classification = self.privacy_ontology.classify_data_field(data_field, context)
        privacy_decision = self.privacy_ontology.make_privacy_decision(
            data_field=data_field,
            requester_role=requester_role,
            context=context,
            organizational_context=organizational_context
        )
        
        print(f"🧠 Privacy Classification: {privacy_classification}")
        print(f"🔐 Base Privacy Decision: {privacy_decision['decision']}")
        
        # Step 2: Team A Temporal Analysis (if enabled)
        temporal_analysis = None
        if self.enable_temporal and temporal_context:
            temporal_analysis = self._analyze_temporal_context(
                data_field=data_field,
                requester_role=requester_role,
                context=context,
                temporal_context=temporal_context
            )
            print(f"⏰ Temporal Analysis: {temporal_analysis['decision']}")
        
        # Step 3: Combined Decision Logic
        final_decision = self._combine_privacy_and_temporal_decisions(
            privacy_decision=privacy_decision,
            temporal_analysis=temporal_analysis
        )
        
        # Step 4: Store Combined Decision in Neo4j
        self._store_integrated_decision(
            data_field=data_field,
            requester_role=requester_role,
            context=context,
            privacy_decision=privacy_decision,
            temporal_analysis=temporal_analysis,
            final_decision=final_decision
        )
        
        print(f"✅ Final Decision: {final_decision['decision']}")
        print(f"📊 Confidence: {final_decision['confidence']:.2f}")
        
        return final_decision
    
    def _analyze_temporal_context(
        self,
        data_field: str,
        requester_role: str,
        context: str,
        temporal_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal context using Team A's framework."""
        try:
            # Create enhanced 6-tuple for Team A
            enhanced_tuple = EnhancedContextualIntegrityTuple(
                data_subject=data_field,
                data_sender=temporal_context.get("data_sender", "system"),
                data_recipient=requester_role,
                data_type=temporal_context.get("data_type", "sensitive"),
                transmission_principle=temporal_context.get("transmission_principle", "secure"),
                temporal_context=TemporalContext(
                    situation=context,
                    urgency_level=temporal_context.get("urgency_level", "normal"),
                    time_window=TimeWindow(
                        start=datetime.now(timezone.utc),
                        end=datetime.now(timezone.utc) + timedelta(hours=1),
                        window_type="access_window"
                    )
                )
            )
            
            # Evaluate using Team A's temporal engine
            temporal_result = self.temporal_evaluator.evaluate(enhanced_tuple)
            
            return {
                "decision": temporal_result.get("decision", "DENY"),
                "confidence": temporal_result.get("confidence", 0.5),
                "reasoning": temporal_result.get("reasoning", "Temporal evaluation"),
                "emergency_override": temporal_result.get("emergency_override", False),
                "time_window_valid": temporal_result.get("time_window_valid", True)
            }
            
        except Exception as e:
            print(f"⚠️  Temporal analysis error: {e}")
            return {
                "decision": "DENY",
                "confidence": 0.0,
                "reasoning": f"Temporal analysis failed: {e}",
                "emergency_override": False,
                "time_window_valid": False
            }
    
    def _combine_privacy_and_temporal_decisions(
        self,
        privacy_decision: Dict[str, Any],
        temporal_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine Team A temporal + Team C privacy decisions using intelligent logic."""
        
        # Base decision from Team C privacy
        base_decision = privacy_decision["decision"]
        base_confidence = privacy_decision["confidence"]
        base_reasoning = privacy_decision["reasoning"]
        
        # If temporal analysis is not available, return privacy decision
        if not temporal_analysis:
            return {
                "decision": base_decision,
                "confidence": base_confidence,
                "reasoning": f"Privacy only: {base_reasoning}",
                "components": {
                    "privacy": privacy_decision,
                    "temporal": None
                },
                "integration_method": "privacy_only"
            }
        
        # Extract temporal components
        temporal_decision = temporal_analysis["decision"]
        temporal_confidence = temporal_analysis["confidence"]
        emergency_override = temporal_analysis.get("emergency_override", False)
        
        # Combined decision logic
        if emergency_override and base_decision == "DENY":
            # Emergency override can upgrade DENY to ALLOW
            final_decision = "ALLOW"
            final_confidence = max(0.7, temporal_confidence)
            reasoning = f"Emergency override: {temporal_analysis['reasoning']}"
            integration_method = "emergency_override"
            
        elif base_decision == "ALLOW" and temporal_decision == "ALLOW":
            # Both systems agree - high confidence
            final_decision = "ALLOW"
            final_confidence = min(1.0, (base_confidence + temporal_confidence) / 2 + 0.1)
            reasoning = f"Privacy + Temporal agreement: {base_reasoning}"
            integration_method = "consensus_allow"
            
        elif base_decision == "DENY" or temporal_decision == "DENY":
            # Either system denies - err on side of security
            final_decision = "DENY"
            final_confidence = max(base_confidence, temporal_confidence)
            reasoning = f"Security priority: {base_reasoning} | {temporal_analysis['reasoning']}"
            integration_method = "security_priority"
            
        else:
            # Default to privacy decision with temporal context
            final_decision = base_decision
            final_confidence = (base_confidence + temporal_confidence) / 2
            reasoning = f"Privacy-guided: {base_reasoning} (temporal: {temporal_analysis['reasoning']})"
            integration_method = "privacy_guided"
        
        return {
            "decision": final_decision,
            "confidence": final_confidence,
            "reasoning": reasoning,
            "components": {
                "privacy": privacy_decision,
                "temporal": temporal_analysis
            },
            "integration_method": integration_method,
            "emergency_override_used": emergency_override and final_decision == "ALLOW"
        }
    
    def _store_integrated_decision(
        self,
        data_field: str,
        requester_role: str,
        context: str,
        privacy_decision: Dict[str, Any],
        temporal_analysis: Optional[Dict[str, Any]],
        final_decision: Dict[str, Any]
    ):
        """Store integrated decision in Neo4j knowledge graph."""
        try:
            # Create decision record combining both Team A and Team C data
            decision_data = {
                "data_field": data_field,
                "requester_role": requester_role,
                "context": context,
                "privacy_classification": privacy_decision.get("data_classification"),
                "privacy_decision": privacy_decision["decision"],
                "privacy_confidence": privacy_decision["confidence"],
                "temporal_decision": temporal_analysis["decision"] if temporal_analysis else None,
                "temporal_confidence": temporal_analysis["confidence"] if temporal_analysis else None,
                "emergency_override": temporal_analysis.get("emergency_override", False) if temporal_analysis else False,
                "final_decision": final_decision["decision"],
                "final_confidence": final_decision["confidence"],
                "integration_method": final_decision["integration_method"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "team_integration": "A+C_temporal_privacy"
            }
            
            # Store in Team C's privacy bridge
            self.privacy_bridge.create_privacy_decision_episode(
                requester_role=requester_role,
                data_entity=data_field,
                context=context,
                decision=final_decision["decision"],
                reasoning=final_decision["reasoning"],
                confidence=final_decision["confidence"]
            )
            
            # Store in Team A's temporal manager (if available)
            if self.enable_temporal and self.temporal_manager:
                entity_id = self.temporal_manager.create_entity(
                    entity_type="IntegratedDecision",
                    properties=decision_data
                )
                print(f"📊 Stored integrated decision: {entity_id}")
            
        except Exception as e:
            print(f"⚠️  Error storing integrated decision: {e}")

def demo_integrated_system():
    """Demonstrate the integrated Team A + Team C system."""
    print("🚀 Team A + Team C Integrated Privacy & Temporal Framework")
    print("=" * 70)
    print("Combining:")
    print("  🔒 Team C: AI Privacy Firewall with OWL/RDF Ontology")
    print("  ⏰ Team A: 6-Tuple Temporal Framework with Emergency Override")
    print("  🧠 Shared: Neo4j Knowledge Graph Integration")
    print()
    
    # Initialize integrated system
    system = IntegratedPrivacyTemporalSystem()
    
    # Test scenarios combining both frameworks
    test_scenarios = [
        {
            "name": "Emergency Medical Access",
            "data_field": "patient_medical_record",
            "requester_role": "emergency_doctor",
            "context": "emergency_room",
            "temporal_context": {
                "urgency_level": "critical",
                "data_type": "medical",
                "transmission_principle": "emergency_override"
            }
        },
        {
            "name": "After-hours Financial Data",
            "data_field": "customer_financial_data",
            "requester_role": "financial_analyst",
            "context": "after_hours",
            "temporal_context": {
                "urgency_level": "normal",
                "data_type": "financial",
                "transmission_principle": "secure"
            }
        },
        {
            "name": "Weekend Code Repository Access",
            "data_field": "source_code",
            "requester_role": "software_engineer",
            "context": "weekend_maintenance",
            "temporal_context": {
                "urgency_level": "high",
                "data_type": "intellectual_property",
                "transmission_principle": "authorized"
            }
        }
    ]
    
    results = []
    for scenario in test_scenarios:
        print(f"\n🧪 Scenario: {scenario['name']}")
        print("-" * 50)
        
        result = system.make_enhanced_privacy_decision(
            data_field=scenario["data_field"],
            requester_role=scenario["requester_role"],
            context=scenario["context"],
            temporal_context=scenario["temporal_context"]
        )
        
        results.append({
            "scenario": scenario["name"],
            "decision": result["decision"],
            "confidence": result["confidence"],
            "integration_method": result["integration_method"]
        })
    
    # Summary report
    print("\n📊 Integration Test Summary")
    print("=" * 50)
    for result in results:
        print(f"  {result['scenario']}: {result['decision']} "
              f"(confidence: {result['confidence']:.2f}, "
              f"method: {result['integration_method']})")
    
    print(f"\n✅ Completed {len(results)} integrated privacy + temporal decisions")
    return results

if __name__ == "__main__":
    demo_integrated_system()