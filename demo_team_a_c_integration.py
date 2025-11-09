#!/usr/bin/env python3
"""
Team A + Team C Integration Demo
===============================

Quick demonstration of the integrated temporal + privacy framework.

This demo shows:
1. How Team A's temporal context enhances Team C's privacy decisions
2. Emergency override capabilities
3. Time-aware access control
4. Combined audit trails

Run this after starting the enhanced API service on port 8003.

Author: Team C + Team A
Date: 2024-12-30
"""

import requests
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

def demo_api_call(endpoint: str, data: Dict[str, Any], description: str) -> Dict[str, Any]:
    """Make an API call and display results."""
    print(f"\n🔍 {description}")
    print("-" * 50)
    print(f"Endpoint: POST {endpoint}")
    print(f"Request: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(
            f"http://localhost:8003{endpoint}",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Response ({response.status_code}):")
            print(f"Decision: {result.get('decision', 'UNKNOWN')}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"Integration Method: {result.get('integration_method', 'unknown')}")
            print(f"Emergency Override: {result.get('emergency_override_used', False)}")
            print(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")
            
            if 'audit_trail' in result:
                print(f"\nAudit Trail:")
                for i, entry in enumerate(result['audit_trail'], 1):
                    print(f"  {i}. {entry}")
            
            return result
        else:
            print(f"\n❌ Error ({response.status_code}):")
            try:
                error_detail = response.json()
                print(f"Details: {error_detail}")
            except:
                print(f"Details: {response.text}")
            return {"error": True, "status_code": response.status_code}
            
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        return {"error": True, "exception": str(e)}

def main():
    """Run Team A + Team C integration demo."""
    print("🚀 Team A + Team C Integration Demo")
    print("=" * 60)
    print("Demonstrating enhanced privacy decisions with temporal context")
    print("Combining Team A's 6-tuple temporal framework with Team C's privacy firewall")
    print()
    
    # Check API health
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"Integration Mode: {health['components']['integration_mode']}")
            print(f"Team A Temporal: {health['components']['team_a_temporal']}")
            print(f"Team C Privacy: {health['components']['team_c_privacy']}")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API at http://localhost:8003")
        print(f"Error: {e}")
        print("\nPlease start the enhanced API service first:")
        print("python enhanced_privacy_api_service.py")
        return
    
    # Demo Scenarios
    
    # 1. Emergency Medical Access (should allow with override)
    demo_api_call(
        "/temporal-privacy-decision",
        {
            "data_field": "patient_cardiac_monitor_data",
            "requester_role": "emergency_cardiologist", 
            "context": "emergency_room_critical_care",
            "temporal_context": {
                "situation": "cardiac_arrest_resuscitation",
                "urgency_level": "critical",
                "data_type": "medical_critical",
                "transmission_principle": "emergency_override",
                "emergency_override_requested": True,
                "time_window_start": datetime.now(timezone.utc).isoformat(),
                "time_window_end": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
            }
        },
        "🚨 Emergency Medical Access (Critical Urgency)"
    )
    
    # 2. After-hours Financial Analysis (should allow with temporal context)
    demo_api_call(
        "/temporal-privacy-decision",
        {
            "data_field": "quarterly_revenue_data",
            "requester_role": "senior_financial_analyst",
            "context": "quarterly_report_deadline",
            "organizational_context": "finance_team",
            "temporal_context": {
                "situation": "regulatory_deadline_tomorrow",
                "urgency_level": "high",
                "data_type": "financial_reporting",
                "transmission_principle": "secure_authorized",
                "emergency_override_requested": False,
                "time_window_start": datetime.now(timezone.utc).isoformat(),
                "time_window_end": (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()
            }
        },
        "💰 After-hours Financial Analysis (High Priority)"
    )
    
    # 3. Unauthorized Access Attempt (should deny despite temporal context)
    demo_api_call(
        "/temporal-privacy-decision",
        {
            "data_field": "employee_personal_data",
            "requester_role": "temp_contractor",
            "context": "curiosity_browse",
            "temporal_context": {
                "situation": "casual_browsing",
                "urgency_level": "low",
                "data_type": "personal_confidential",
                "transmission_principle": "unauthorized",
                "emergency_override_requested": False
            }
        },
        "🚫 Unauthorized Access Attempt (Should Deny)"
    )
    
    # 4. Executive Access (should allow with org context)
    demo_api_call(
        "/privacy-decision",
        {
            "data_field": "company_strategic_roadmap",
            "requester_role": "chief_executive_officer",
            "context": "board_presentation_prep",
            "organizational_context": "c_suite_executive",
            "temporal_context": {
                "situation": "board_meeting_preparation",
                "urgency_level": "high",
                "data_type": "strategic_confidential",
                "transmission_principle": "executive_privilege"
            }
        },
        "👔 Executive Strategic Access (Organizational Override)"
    )
    
    # 5. Emergency Override Test
    demo_api_call(
        "/emergency-override",
        {
            "data_field": "patient_allergy_medication_data",
            "requester_role": "emergency_nurse",
            "emergency_situation": "anaphylactic_shock_treatment",
            "justification": "Patient unconscious, need allergy info to prevent fatal medication interaction",
            "expected_duration_minutes": 15
        },
        "🆘 Emergency Override (Life-threatening Situation)"
    )
    
    # 6. Classification with Temporal Context
    demo_api_call(
        "/classify",
        {
            "data_field": "research_experiment_data",
            "context": "academic_research_publication",
            "temporal_context": {
                "situation": "peer_review_publication",
                "urgency_level": "normal",
                "data_type": "research_intellectual_property"
            }
        },
        "🔬 Research Data Classification (Academic Context)"
    )
    
    print(f"\n🎯 Demo Complete!")
    print("=" * 60)
    print("Key Integration Features Demonstrated:")
    print("✅ Emergency override capabilities from Team A")
    print("✅ Temporal context awareness in privacy decisions")
    print("✅ Combined Team A + Team C reasoning")
    print("✅ Organizational context integration")
    print("✅ Comprehensive audit trails")
    print("✅ Time-window based access control")
    print("\nFor comprehensive testing, run:")
    print("python test_team_a_c_integration.py")

if __name__ == "__main__":
    main()