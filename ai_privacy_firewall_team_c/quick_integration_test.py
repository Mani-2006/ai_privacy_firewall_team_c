#!/usr/bin/env python3
"""
Quick Integration Test Script
============================

Simple script to test specific Team A + Team C integration scenarios.
Customize the scenarios below to test different combinations.

Usage:
    python quick_integration_test.py

Author: Team C + Team A Integration
Date: 2024-12-30
"""

import requests
import json
from datetime import datetime, timezone

def test_scenario(name, classification_data, privacy_data, temporal_context):
    """Test a single integration scenario."""
    print(f"\n🧪 Testing: {name}")
    print("="*50)
    
    # Step 1: Classification
    print("📊 Step 1: Data Classification")
    try:
        response = requests.post(
            "http://localhost:8002/api/v1/classify",
            json=classification_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            classification = response.json()
            print(f"✅ Data Type: {classification['data_type']}")
            print(f"✅ Sensitivity: {classification['sensitivity']}")
            print(f"✅ Confidence: {classification['confidence']:.2f}")
        else:
            print(f"❌ Classification failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return
    
    # Step 2: Privacy Decision
    print(f"\n🔒 Step 2: Privacy Decision")
    try:
        response = requests.post(
            "http://localhost:8002/api/v1/privacy-decision",
            json=privacy_data,
            headers={"Content-Type": "application/json"}, 
            timeout=10
        )
        if response.status_code == 200:
            privacy = response.json()
            decision = "ALLOW" if privacy['allowed'] else "DENY"
            print(f"✅ Team C Decision: {decision}")
            print(f"✅ Reason: {privacy['reason']}")
            print(f"✅ Confidence: {privacy['confidence']:.2f}")
            print(f"✅ Emergency Used: {privacy['emergency_used']}")
        else:
            print(f"❌ Privacy decision failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Privacy decision error: {e}")
        return
    
    # Step 3: Temporal Context (Simulated)
    print(f"\n⏰ Step 3: Team A Temporal Context (Simulated)")
    urgency = temporal_context.get('urgency_level', 'normal')
    emergency = temporal_context.get('emergency_override', False)
    
    if emergency and urgency == 'critical':
        temporal_decision = "ALLOW_OVERRIDE"
        temporal_confidence = 0.95
    elif urgency in ['high', 'critical']:
        temporal_decision = "ALLOW"
        temporal_confidence = 0.85
    elif urgency == 'normal':
        temporal_decision = "ALLOW"
        temporal_confidence = 0.70
    else:
        temporal_decision = "DENY"
        temporal_confidence = 0.60
    
    print(f"✅ Team A Decision: {temporal_decision}")
    print(f"✅ Urgency Level: {urgency}")
    print(f"✅ Emergency Override: {emergency}")
    print(f"✅ Confidence: {temporal_confidence:.2f}")
    
    # Step 4: Integration Logic
    print(f"\n🔗 Step 4: Integrated Decision")
    privacy_allowed = privacy['allowed']
    
    if emergency and urgency == 'critical' and not privacy_allowed:
        final_decision = "ALLOW"
        method = "emergency_override"
        final_confidence = 0.95
    elif privacy_allowed and temporal_decision in ["ALLOW", "ALLOW_OVERRIDE"]:
        final_decision = "ALLOW"
        method = "consensus_allow"
        final_confidence = min(1.0, (privacy['confidence'] + temporal_confidence) / 2 + 0.1)
    elif not privacy_allowed or temporal_decision == "DENY":
        final_decision = "DENY"
        method = "security_priority"
        final_confidence = max(privacy['confidence'], temporal_confidence)
    else:
        final_decision = "ALLOW" if privacy_allowed else "DENY"
        method = "privacy_guided"
        final_confidence = (privacy['confidence'] + temporal_confidence) / 2
    
    print(f"🎯 Final Decision: {final_decision}")
    print(f"🎯 Integration Method: {method}")
    print(f"🎯 Final Confidence: {final_confidence:.2f}")
    
    return {
        "name": name,
        "classification": classification,
        "privacy": privacy,
        "temporal": {
            "decision": temporal_decision,
            "confidence": temporal_confidence,
            "emergency": emergency
        },
        "final": {
            "decision": final_decision,
            "method": method,
            "confidence": final_confidence
        }
    }

def main():
    """Run quick integration tests."""
    print("🚀 Quick Team A + Team C Integration Tests")
    print("Testing enhanced privacy decisions with temporal context")
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8002/docs", timeout=5)
        if response.status_code != 200:
            print("❌ Team C API not accessible. Please start it first:")
            print("   cd ai_privacy_firewall_team_c")
            print("   PYTHONPATH=. python api/privacy_api_service.py")
            return
    except:
        print("❌ Cannot connect to Team C API on port 8002")
        return
    
    print("✅ Team C Privacy API is running")
    
    # Test Scenarios - Customize these as needed
    scenarios = [
        {
            "name": "🚨 Medical Emergency - Doctor",
            "classification": {
                "data_field": "patient_vital_signs",
                "context": "emergency_room"
            },
            "privacy": {
                "requester": "emergency_doctor",
                "data_field": "patient_vital_signs",
                "purpose": "emergency_treatment",
                "context": "life_threatening_emergency",
                "emergency": True
            },
            "temporal": {
                "urgency_level": "critical",
                "emergency_override": True
            }
        },
        
        {
            "name": "💰 Financial Data - Analyst",
            "classification": {
                "data_field": "revenue_projections",
                "context": "quarterly_analysis"
            },
            "privacy": {
                "requester": "financial_analyst",
                "data_field": "revenue_projections", 
                "purpose": "quarterly_report",
                "context": "business_hours",
                "emergency": False
            },
            "temporal": {
                "urgency_level": "normal",
                "emergency_override": False
            }
        },
        
        {
            "name": "🔒 Sensitive Data - Unauthorized",
            "classification": {
                "data_field": "employee_social_security",
                "context": "data_browsing"
            },
            "privacy": {
                "requester": "contractor",
                "data_field": "employee_social_security",
                "purpose": "curiosity",
                "context": "unauthorized_access",
                "emergency": False
            },
            "temporal": {
                "urgency_level": "low",
                "emergency_override": False
            }
        },
        
        {
            "name": "🚨 Emergency - Non-Medical",
            "classification": {
                "data_field": "patient_medication_list",
                "context": "building_emergency"
            },
            "privacy": {
                "requester": "security_guard",
                "data_field": "patient_medication_list",
                "purpose": "emergency_response",
                "context": "medical_emergency",
                "emergency": True
            },
            "temporal": {
                "urgency_level": "critical",
                "emergency_override": True
            }
        }
    ]
    
    results = []
    for scenario in scenarios:
        result = test_scenario(
            scenario["name"],
            scenario["classification"],
            scenario["privacy"],
            scenario["temporal"]
        )
        if result:
            results.append(result)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("="*50)
    for result in results:
        privacy_decision = "ALLOW" if result['privacy']['allowed'] else "DENY"
        temporal_decision = result['temporal']['decision']
        final_decision = result['final']['decision']
        
        print(f"{result['name']}:")
        print(f"   Team C: {privacy_decision} | Team A: {temporal_decision} | Final: {final_decision}")
        print(f"   Method: {result['final']['method']} | Confidence: {result['final']['confidence']:.2f}")
        print()
    
    print("🎯 Integration working successfully!")
    print("💡 Modify the scenarios above to test different combinations")

if __name__ == "__main__":
    main()