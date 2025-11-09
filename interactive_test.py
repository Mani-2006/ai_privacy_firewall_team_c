#!/usr/bin/env python3
"""
Interactive Integration Tester
==============================

Interactive script to test Team A + Team C integration with custom scenarios.
Easily modify parameters to see how different combinations affect decisions.

Usage:
    python interactive_test.py

Author: Team C + Team A Integration
Date: 2024-12-30
"""

import requests
import json

def test_custom_scenario():
    """Test a custom scenario with user-defined parameters."""
    print("\n🧪 Custom Integration Test")
    print("="*40)
    
    # Easy customization - just change these values!
    test_config = {
        "data_field": "patient_allergy_information",  # Change this
        "context": "emergency_room",                  # Change this
        "requester": "emergency_nurse",               # Change this  
        "purpose": "medication_safety_check",         # Change this
        "emergency": True,                            # Change this
        "urgency_level": "critical",                  # low, normal, high, critical
        "emergency_override": True                    # Change this
    }
    
    print(f"🔍 Testing Configuration:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")
    
    # Step 1: Classification
    print(f"\n📊 Step 1: Data Classification")
    classification_data = {
        "data_field": test_config["data_field"],
        "context": test_config["context"]
    }
    
    try:
        response = requests.post(
            "http://localhost:8002/api/v1/classify",
            json=classification_data,
            timeout=10
        )
        
        if response.status_code == 200:
            classification = response.json()
            print(f"✅ Data Type: {classification['data_type']}")
            print(f"✅ Sensitivity: {classification['sensitivity']}")
            print(f"✅ Reasoning: {classification.get('reasoning', [])}")
        else:
            print(f"❌ Classification failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Step 2: Privacy Decision
    print(f"\n🔒 Step 2: Privacy Decision")
    privacy_data = {
        "requester": test_config["requester"],
        "data_field": test_config["data_field"],
        "purpose": test_config["purpose"],
        "context": test_config["context"],
        "emergency": test_config["emergency"]
    }
    
    try:
        response = requests.post(
            "http://localhost:8002/api/v1/privacy-decision",
            json=privacy_data,
            timeout=10
        )
        
        if response.status_code == 200:
            privacy = response.json()
            print(f"✅ Team C Decision: {'ALLOW' if privacy['allowed'] else 'DENY'}")
            print(f"✅ Reason: {privacy['reason']}")
            print(f"✅ Confidence: {privacy['confidence']:.2f}")
            print(f"✅ Emergency Used: {privacy['emergency_used']}")
        else:
            print(f"❌ Privacy decision failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Step 3: Temporal Analysis (Simulated)
    print(f"\n⏰ Step 3: Team A Temporal Analysis")
    urgency = test_config["urgency_level"]
    emergency_override = test_config["emergency_override"]
    
    # Temporal decision logic
    if emergency_override and urgency == "critical":
        temporal_decision = "ALLOW_OVERRIDE"
        temporal_confidence = 0.95
    elif urgency == "critical":
        temporal_decision = "ALLOW"
        temporal_confidence = 0.90
    elif urgency == "high":
        temporal_decision = "ALLOW"
        temporal_confidence = 0.85
    elif urgency == "normal":
        temporal_decision = "ALLOW"
        temporal_confidence = 0.70
    else:  # low urgency
        temporal_decision = "DENY"
        temporal_confidence = 0.60
    
    print(f"✅ Team A Decision: {temporal_decision}")
    print(f"✅ Urgency Level: {urgency}")
    print(f"✅ Emergency Override: {emergency_override}")
    print(f"✅ Temporal Confidence: {temporal_confidence:.2f}")
    
    # Step 4: Integration
    print(f"\n🔗 Step 4: Integrated Decision Logic")
    privacy_allowed = privacy['allowed']
    
    # Integration logic
    if emergency_override and urgency == "critical" and not privacy_allowed:
        final_decision = "ALLOW"
        method = "emergency_override"
        final_confidence = 0.95
        explanation = "Team A emergency override superseded Team C denial"
        
    elif privacy_allowed and temporal_decision in ["ALLOW", "ALLOW_OVERRIDE"]:
        final_decision = "ALLOW"
        method = "consensus_allow"
        final_confidence = min(1.0, (privacy['confidence'] + temporal_confidence) / 2 + 0.1)
        explanation = "Both Team A and Team C approved access"
        
    elif not privacy_allowed or temporal_decision == "DENY":
        final_decision = "DENY"
        method = "security_priority"
        final_confidence = max(privacy['confidence'], temporal_confidence)
        explanation = "Security priority - at least one system denied access"
        
    else:
        final_decision = "ALLOW" if privacy_allowed else "DENY"
        method = "privacy_guided"
        final_confidence = (privacy['confidence'] + temporal_confidence) / 2
        explanation = "Team C privacy decision with temporal context"
    
    print(f"🎯 Final Decision: {final_decision}")
    print(f"🎯 Integration Method: {method}")
    print(f"🎯 Final Confidence: {final_confidence:.2f}")
    print(f"🎯 Explanation: {explanation}")
    
    # Summary
    print(f"\n📋 Decision Summary")
    print(f"="*40)
    print(f"Data: {test_config['data_field']}")
    print(f"Requester: {test_config['requester']}")
    print(f"Context: {test_config['context']}")
    print(f"Emergency: {test_config['emergency']}")
    print(f"")
    print(f"Team C (Privacy): {'ALLOW' if privacy_allowed else 'DENY'} ({privacy['confidence']:.2f})")
    print(f"Team A (Temporal): {temporal_decision} ({temporal_confidence:.2f})")
    print(f"Integrated: {final_decision} ({final_confidence:.2f}) via {method}")

def run_predefined_tests():
    """Run a series of predefined test cases."""
    print("\n🎯 Predefined Integration Tests")
    print("="*40)
    
    test_cases = [
        {
            "name": "🚨 Life-threatening Emergency",
            "data_field": "patient_medication_allergies",
            "context": "intensive_care_unit",
            "requester": "icu_physician", 
            "purpose": "prevent_allergic_reaction",
            "emergency": True,
            "urgency_level": "critical",
            "emergency_override": True
        },
        {
            "name": "📊 Business Intelligence Access",
            "data_field": "customer_purchase_analytics",
            "context": "quarterly_review",
            "requester": "data_scientist",
            "purpose": "trend_analysis",
            "emergency": False,
            "urgency_level": "normal", 
            "emergency_override": False
        },
        {
            "name": "🔒 Unauthorized Attempt",
            "data_field": "executive_compensation_data",
            "context": "employee_curiosity",
            "requester": "junior_developer",
            "purpose": "salary_comparison",
            "emergency": False,
            "urgency_level": "low",
            "emergency_override": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        # Quick test
        classification_data = {
            "data_field": test_case["data_field"],
            "context": test_case["context"]
        }
        
        privacy_data = {
            "requester": test_case["requester"],
            "data_field": test_case["data_field"],
            "purpose": test_case["purpose"],
            "context": test_case["context"],
            "emergency": test_case["emergency"]
        }
        
        try:
            # Classification
            class_resp = requests.post("http://localhost:8002/api/v1/classify", json=classification_data, timeout=10)
            privacy_resp = requests.post("http://localhost:8002/api/v1/privacy-decision", json=privacy_data, timeout=10)
            
            if class_resp.status_code == 200 and privacy_resp.status_code == 200:
                classification = class_resp.json()
                privacy = privacy_resp.json()
                
                print(f"📊 Classification: {classification['data_type']} - {classification['sensitivity']}")
                print(f"🔒 Team C: {'ALLOW' if privacy['allowed'] else 'DENY'} (confidence: {privacy['confidence']:.2f})")
                print(f"⏰ Team A: Urgency {test_case['urgency_level']}, Override: {test_case['emergency_override']}")
                
                # Quick integration logic
                if test_case['emergency_override'] and test_case['urgency_level'] == 'critical':
                    final = "ALLOW (Emergency Override)"
                elif privacy['allowed'] and test_case['urgency_level'] in ['normal', 'high', 'critical']:
                    final = "ALLOW (Consensus)"
                elif not privacy['allowed'] or test_case['urgency_level'] == 'low':
                    final = "DENY (Security Priority)"
                else:
                    final = "ALLOW (Standard)"
                
                print(f"🎯 Integrated Decision: {final}")
            else:
                print(f"❌ API call failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main interactive menu."""
    print("🚀 Interactive Team A + Team C Integration Tester")
    print("="*50)
    
    # Check API connectivity
    try:
        response = requests.get("http://localhost:8002/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Team C Privacy API is accessible")
        else:
            print("❌ Team C API issues detected")
            return
    except:
        print("❌ Cannot connect to Team C API on port 8002")
        print("Please start the API first:")
        print("  cd ai_privacy_firewall_team_c")
        print("  PYTHONPATH=. python api/privacy_api_service.py")
        return
    
    print("\nChoose your test:")
    print("1. 🧪 Custom Scenario Test (modify code above)")
    print("2. 🎯 Predefined Test Cases")
    print("3. 📖 View Current Custom Configuration")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_custom_scenario()
    elif choice == "2":
        run_predefined_tests()
    elif choice == "3":
        print("\n📖 To customize scenarios:")
        print("   1. Edit the 'test_config' dictionary in test_custom_scenario()")
        print("   2. Change data_field, requester, emergency, urgency_level, etc.")
        print("   3. Run option 1 to see the results")
        print("\n💡 Example modifications:")
        print('   data_field: "customer_credit_cards" → "source_code_repository"')
        print('   requester: "intern" → "ceo"')
        print('   emergency: False → True')
        print('   urgency_level: "low" → "critical"')
    else:
        print("Invalid choice. Run the script again.")

if __name__ == "__main__":
    main()