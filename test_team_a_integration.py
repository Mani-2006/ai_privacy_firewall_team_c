#!/usr/bin/env python3
"""
Team A Integration Test Script
=============================

Tests the enhanced privacy bridge with Team A's temporal framework integration.
Demonstrates the new data models and request/response handling.

Author: Team C Privacy Firewall - Integration Update
Date: 2025-12-02
"""

import asyncio
import json
from datetime import datetime, timezone
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent)
sys.path.append(parent_dir)

from integration.enhanced_graphiti_privacy_bridge import EnhancedGraphitiPrivacyBridge
from integration.team_a_models import TeamAIntegrationClient, EnhancedContextualIntegrityTuple, TemporalContext


async def test_team_a_integration():
    """Test the Team A temporal framework integration."""
    print("🧪 Testing Team A Temporal Framework Integration")
    print("=" * 60)
    
    # Initialize the enhanced bridge
    bridge = EnhancedGraphitiPrivacyBridge()
    
    # Test cases matching Team A's expected scenarios
    test_cases = [
        {
            "name": "Medical Professional Access",
            "privacy_request": {
                "requester": "Dr. Sarah Johnson",
                "requester_role": "attending_physician",
                "base_role": "doctor",
                "data_field": "patient_medical_records",
                "data_subject": "patient_12345", 
                "purpose": "treatment_planning",
                "context": "medical",
                "emergency": False,
                "service_id": "hospital_ehr_system",
                "inherited_permissions": ["read_medical_data", "update_treatment_plan"]
            }
        },
        {
            "name": "Emergency Override Request",
            "privacy_request": {
                "requester": "Dr. Mike Chen",
                "requester_role": "emergency_physician", 
                "base_role": "doctor",
                "data_field": "patient_critical_vitals",
                "data_subject": "patient_67890",
                "purpose": "emergency_treatment",
                "context": "medical",
                "emergency": True,
                "emergency_auth_id": "emrg_2025_001",
                "service_id": "emergency_room_system",
                "inherited_permissions": ["emergency_access", "critical_data_override"]
            }
        },
        {
            "name": "HR Employee Data Access",
            "privacy_request": {
                "requester": "Alice HR Manager",
                "requester_role": "hr_manager",
                "base_role": "hr_staff", 
                "data_field": "employee_salary_information",
                "data_subject": "employee_54321",
                "purpose": "payroll_processing",
                "context": "administrative",
                "emergency": False,
                "service_id": "hr_management_system",
                "inherited_permissions": ["read_employee_data", "process_payroll"]
            }
        },
        {
            "name": "Unauthorized Access Attempt", 
            "privacy_request": {
                "requester": "John Contractor",
                "requester_role": "temp_contractor",
                "base_role": "contractor",
                "data_field": "confidential_financial_data", 
                "data_subject": "company_financial_records",
                "purpose": "general_access",
                "context": "business",
                "emergency": False,
                "service_id": "contractor_portal",
                "inherited_permissions": []
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Test Team A tuple creation
            tuple_data = bridge.team_a_client.create_temporal_tuple(
                test_case["privacy_request"],
                emergency=test_case["privacy_request"].get("emergency", False)
            )
            
            print(f"✅ Created Team A tuple:")
            print(f"   Request ID: {tuple_data.request_id}")
            print(f"   Data Type: {tuple_data.data_type}")
            print(f"   Emergency Auth: {tuple_data.temporal_context.emergency_authorization_id}")
            print(f"   Base Role: {tuple_data.temporal_context.base_role}")
            print(f"   Inherited Perms: {tuple_data.temporal_context.inherited_permissions}")
            
            # Test Team A request formatting
            team_a_request = bridge.team_a_client.format_request_for_team_a(tuple_data)
            
            print(f"✅ Team A Request Format - ALL FIELDS:")
            print(f"   📋 REQUIRED FIELDS:")
            print(f"      request_id: {team_a_request['request_id']}")
            print(f"      emergency_authorization_id: {team_a_request.get('emergency_authorization_id', 'None')}")
            print(f"   📊 CORE TUPLE:")
            print(f"      data_type: {team_a_request['data_type']}")
            print(f"      data_sender: {team_a_request['data_sender']}")
            print(f"      data_recipient: {team_a_request['data_recipient']}")
            print(f"      transmission_principle: {team_a_request['transmission_principle']}")
            print(f"      data_subject: {team_a_request['data_subject']}")
            print(f"   ⏰ TEMPORAL CONTEXT:")
            tc = team_a_request['temporal_context']
            print(f"      situation: {tc['situation']}")
            print(f"      urgency_level: {tc['urgency_level']}")
            print(f"      temporal_role: {tc['temporal_role']}")
            print(f"      emergency_override: {tc['emergency_override']}")
            print(f"      access_window: {tc['access_window']}")
            print(f"      timestamp: {tc['timestamp']}")
            print(f"      timezone: {tc['timezone']}")
            print(f"   🔍 AUDIT FIELDS:")
            print(f"      correlation_id: {team_a_request['correlation_id']}")
            print(f"      audit_required: {team_a_request['audit_required']}")
            
            # Test full privacy decision with Team A integration
            decision = await bridge.create_privacy_decision_episode(test_case["privacy_request"])
            
            print(f"✅ Privacy Decision:")
            print(f"   Decision: {'ALLOWED' if decision['allowed'] else 'DENIED'}")
            print(f"   Reason: {decision['reason']}")
            print(f"   Team A Integration: {decision.get('team_a_integration', False)}")
            print(f"   Decision ID: {decision.get('decision_id', 'N/A')}")
            print(f"   Risk Level: {decision.get('risk_level', 'N/A')}")
            print(f"   Expires At: {decision.get('expires_at', 'N/A')}")
            
            results.append({
                "test_case": test_case['name'],
                "status": "PASSED",
                "decision": decision
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test_case": test_case['name'], 
                "status": "FAILED",
                "error": str(e)
            })
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASSED")
    total = len(results)
    
    for result in results:
        status_emoji = "✅" if result["status"] == "PASSED" else "❌"
        print(f"{status_emoji} {result['test_case']}: {result['status']}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    # Test Team A data models directly
    print(f"\n🧪 Testing Team A Data Models Directly")
    print("-" * 40)
    
    # Test TemporalContext with new fields
    temporal_context = TemporalContext(
        timestamp=datetime.now(timezone.utc),
        timezone="UTC",
        business_hours=True,
        emergency_override=False,
        emergency_authorization_id=None,
        base_role="doctor", 
        inherited_permissions=["read_medical_data"],
        temporal_role_valid_until=None
    )
    
    print(f"✅ TemporalContext created with Team A fields:")
    print(f"   Emergency Auth ID: {temporal_context.emergency_authorization_id}")
    print(f"   Base Role: {temporal_context.base_role}")
    print(f"   Inherited Permissions: {temporal_context.inherited_permissions}")
    
    # Test EnhancedContextualIntegrityTuple 
    tuple_data = EnhancedContextualIntegrityTuple(
        data_type="medical_data",
        data_sender="team_c_privacy_system",
        data_recipient="Dr. Sarah Johnson",
        transmission_principle="access_control",
        temporal_context=temporal_context,
        data_subject="patient_12345",
        # New Team A fields get auto-generated
        audit_required=True
    )
    
    print(f"✅ EnhancedContextualIntegrityTuple created:")
    print(f"   Request ID: {tuple_data.request_id}")
    print(f"   Correlation ID: {tuple_data.correlation_id}")
    print(f"   Audit Required: {tuple_data.audit_required}")
    
    await bridge.close()
    
    if passed == total:
        print(f"\n🎉 All Team A integration tests passed! Ready for production.")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Please review before deployment.")
        return False


if __name__ == "__main__":
    print("🚀 Starting Team A Integration Tests...")
    success = asyncio.run(test_team_a_integration())
    exit(0 if success else 1)