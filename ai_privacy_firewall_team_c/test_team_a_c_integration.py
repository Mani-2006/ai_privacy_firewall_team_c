#!/usr/bin/env python3
"""
Team A + Team C Integration Test Suite
======================================

Comprehensive test suite for the integrated temporal + privacy framework.

This test runner validates:
1. Team C privacy decisions (standalone)
2. Team A temporal decisions (standalone) 
3. Integrated Team A + C decisions
4. Emergency override scenarios
5. API endpoint functionality
6. Edge cases and error handling

Author: Team C + Team A Integration
Date: 2024-12-30
"""

import sys
import os
import json
import time
import asyncio
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add paths for both Team A and Team C
TEAM_C_PATH = Path(__file__).parent
TEAM_A_PATH = Path(__file__).parent.parent / "ai_temporal_framework"

sys.path.insert(0, str(TEAM_C_PATH))
if TEAM_A_PATH.exists():
    sys.path.insert(0, str(TEAM_A_PATH))

class IntegratedTestRunner:
    """Comprehensive test runner for Team A + C integration."""
    
    def __init__(self, api_base_url: str = "http://localhost:8003"):
        """Initialize test runner with API endpoint."""
        self.api_base_url = api_base_url
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        # Test scenarios
        self.test_scenarios = self._load_test_scenarios()
    
    def _load_test_scenarios(self) -> List[Dict[str, Any]]:
        """Load comprehensive test scenarios for integration testing."""
        return [
            {
                "name": "Emergency Medical Access - Critical",
                "category": "emergency_override",
                "data": {
                    "data_field": "patient_medical_record",
                    "requester_role": "emergency_doctor",
                    "context": "emergency_room",
                    "temporal_context": {
                        "situation": "cardiac_arrest_patient",
                        "urgency_level": "critical",
                        "data_type": "medical",
                        "transmission_principle": "emergency_override",
                        "emergency_override_requested": True
                    }
                },
                "expected": {
                    "decision": "ALLOW",
                    "emergency_override_used": True,
                    "confidence_min": 0.7
                }
            },
            {
                "name": "After-hours Financial Data - High Urgency",
                "category": "temporal_privacy",
                "data": {
                    "data_field": "customer_financial_data",
                    "requester_role": "financial_analyst",
                    "context": "after_hours_analysis",
                    "temporal_context": {
                        "situation": "regulatory_deadline",
                        "urgency_level": "high",
                        "data_type": "financial",
                        "transmission_principle": "secure",
                        "emergency_override_requested": False
                    }
                },
                "expected": {
                    "decision": "ALLOW",
                    "emergency_override_used": False,
                    "confidence_min": 0.6
                }
            },
            {
                "name": "Weekend Source Code Access - Normal",
                "category": "temporal_privacy",
                "data": {
                    "data_field": "source_code",
                    "requester_role": "software_engineer",
                    "context": "weekend_maintenance",
                    "temporal_context": {
                        "situation": "scheduled_maintenance",
                        "urgency_level": "normal",
                        "data_type": "intellectual_property",
                        "transmission_principle": "authorized",
                        "emergency_override_requested": False
                    }
                },
                "expected": {
                    "decision": "ALLOW",
                    "emergency_override_used": False,
                    "confidence_min": 0.5
                }
            },
            {
                "name": "Unauthorized HR Data Access - Low Priority",
                "category": "privacy_deny",
                "data": {
                    "data_field": "employee_salary_data",
                    "requester_role": "intern",
                    "context": "routine_analysis",
                    "temporal_context": {
                        "situation": "learning_exercise",
                        "urgency_level": "low",
                        "data_type": "confidential",
                        "transmission_principle": "unauthorized",
                        "emergency_override_requested": False
                    }
                },
                "expected": {
                    "decision": "DENY",
                    "emergency_override_used": False,
                    "confidence_min": 0.6
                }
            },
            {
                "name": "Executive Override - Administrative",
                "category": "organizational",
                "data": {
                    "data_field": "company_strategic_plan",
                    "requester_role": "ceo",
                    "context": "board_meeting",
                    "organizational_context": "executive_level",
                    "temporal_context": {
                        "situation": "strategic_planning",
                        "urgency_level": "high",
                        "data_type": "strategic",
                        "transmission_principle": "executive_access",
                        "emergency_override_requested": False
                    }
                },
                "expected": {
                    "decision": "ALLOW",
                    "emergency_override_used": False,
                    "confidence_min": 0.8
                }
            },
            {
                "name": "Medical Emergency - Non-Medical Personnel",
                "category": "emergency_mixed",
                "data": {
                    "data_field": "patient_medical_record",
                    "requester_role": "security_guard",
                    "context": "medical_emergency",
                    "temporal_context": {
                        "situation": "medical_emergency_response",
                        "urgency_level": "critical",
                        "data_type": "medical",
                        "transmission_principle": "emergency_assistance",
                        "emergency_override_requested": True
                    }
                },
                "expected": {
                    "decision": "DENY",  # Even with emergency, non-medical personnel shouldn't access medical records
                    "emergency_override_used": False,
                    "confidence_min": 0.7
                }
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🚀 Team A + Team C Integration Test Suite")
        print("=" * 60)
        
        # Test API health
        await self._test_api_health()
        
        # Test API contracts
        await self._test_api_contracts()
        
        # Test basic privacy classification
        await self._test_privacy_classification()
        
        # Test enhanced privacy decisions
        await self._test_enhanced_privacy_decisions()
        
        # Test temporal privacy decisions
        await self._test_temporal_privacy_decisions()
        
        # Test emergency override
        await self._test_emergency_override()
        
        # Test error handling
        await self._test_error_handling()
        
        # Generate summary report
        return self._generate_test_report()
    
    async def _test_api_health(self):
        """Test API health endpoint."""
        print("\n🔍 Testing API Health & Integration Status")
        print("-" * 40)
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Health: {health_data['status']}")
                print(f"   Integration Mode: {health_data['components']['integration_mode']}")
                print(f"   Team C Privacy: {health_data['components']['team_c_privacy']}")
                print(f"   Team A Temporal: {health_data['components']['team_a_temporal']}")
                
                self._record_test_result("API Health Check", True, "API accessible and components loaded")
            else:
                print(f"❌ API Health Check failed: {response.status_code}")
                self._record_test_result("API Health Check", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ API Health Check error: {e}")
            self._record_test_result("API Health Check", False, str(e))
    
    async def _test_api_contracts(self):
        """Test API contracts endpoint."""
        print("\n📋 Testing API Contracts")
        print("-" * 40)
        
        try:
            response = requests.get(f"{self.api_base_url}/contracts", timeout=10)
            
            if response.status_code == 200:
                contracts = response.json()
                
                # Check for Team A integration contracts
                if "team_a_temporal_integration" in contracts:
                    print("✅ Team A temporal integration contracts found")
                    temporal_endpoints = contracts["team_a_temporal_integration"]["endpoints"]
                    print(f"   Temporal endpoints: {len(temporal_endpoints)}")
                    
                    self._record_test_result("API Contracts - Team A Integration", True, "Temporal contracts available")
                else:
                    print("⚠️  Team A temporal integration contracts not found")
                    self._record_test_result("API Contracts - Team A Integration", False, "Temporal contracts missing")
                
                if "team_c_privacy" in contracts:
                    print("✅ Team C privacy contracts found")
                    self._record_test_result("API Contracts - Team C Privacy", True, "Privacy contracts available")
                else:
                    print("❌ Team C privacy contracts not found")
                    self._record_test_result("API Contracts - Team C Privacy", False, "Privacy contracts missing")
                    
            else:
                print(f"❌ API Contracts failed: {response.status_code}")
                self._record_test_result("API Contracts", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ API Contracts error: {e}")
            self._record_test_result("API Contracts", False, str(e))
    
    async def _test_privacy_classification(self):
        """Test enhanced privacy classification."""
        print("\n🧠 Testing Enhanced Privacy Classification")
        print("-" * 40)
        
        test_cases = [
            {
                "data_field": "patient_medical_record",
                "context": "hospital_emergency",
                "temporal_context": {
                    "situation": "emergency_treatment",
                    "urgency_level": "critical"
                }
            },
            {
                "data_field": "customer_financial_data", 
                "context": "financial_analysis",
                "temporal_context": {
                    "situation": "quarterly_report",
                    "urgency_level": "normal"
                }
            }
        ]
        
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.api_base_url}/classify",
                    json=test_case,
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    classification = result.get("classification", "unknown")
                    integration_mode = result.get("integration_mode", "unknown")
                    
                    print(f"✅ Classification for {test_case['data_field']}: {classification}")
                    print(f"   Integration Mode: {integration_mode}")
                    
                    self._record_test_result(
                        f"Classification - {test_case['data_field']}", 
                        True, 
                        f"Classified as {classification} in {integration_mode} mode"
                    )
                else:
                    print(f"❌ Classification failed for {test_case['data_field']}: {response.status_code}")
                    self._record_test_result(
                        f"Classification - {test_case['data_field']}", 
                        False, 
                        f"HTTP {response.status_code}"
                    )
                    
            except Exception as e:
                print(f"❌ Classification error for {test_case['data_field']}: {e}")
                self._record_test_result(
                    f"Classification - {test_case['data_field']}", 
                    False, 
                    str(e)
                )
    
    async def _test_enhanced_privacy_decisions(self):
        """Test enhanced privacy decisions with temporal context."""
        print("\n🔐 Testing Enhanced Privacy Decisions")
        print("-" * 40)
        
        for scenario in self.test_scenarios[:3]:  # Test first 3 scenarios
            await self._test_single_privacy_decision(scenario, "/privacy-decision")
    
    async def _test_temporal_privacy_decisions(self):
        """Test dedicated temporal privacy decisions."""
        print("\n⏰ Testing Temporal Privacy Decisions")
        print("-" * 40)
        
        for scenario in self.test_scenarios[2:5]:  # Test middle scenarios
            await self._test_single_privacy_decision(scenario, "/temporal-privacy-decision")
    
    async def _test_emergency_override(self):
        """Test emergency override functionality."""
        print("\n🚨 Testing Emergency Override")
        print("-" * 40)
        
        emergency_request = {
            "data_field": "patient_critical_care_data",
            "requester_role": "emergency_physician",
            "emergency_situation": "cardiac_arrest_resuscitation",
            "justification": "Life-threatening emergency requiring immediate access to patient history",
            "expected_duration_minutes": 30
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/emergency-override",
                json=emergency_request,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                decision = result.get("decision", "UNKNOWN")
                emergency_used = result.get("emergency_override_used", False)
                
                print(f"✅ Emergency Override: {decision}")
                print(f"   Override Used: {emergency_used}")
                print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
                
                # Emergency should typically allow access
                expected_success = decision == "ALLOW" and emergency_used
                self._record_test_result(
                    "Emergency Override", 
                    expected_success, 
                    f"Decision: {decision}, Override: {emergency_used}"
                )
            else:
                print(f"❌ Emergency Override failed: {response.status_code}")
                self._record_test_result("Emergency Override", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Emergency Override error: {e}")
            self._record_test_result("Emergency Override", False, str(e))
    
    async def _test_single_privacy_decision(self, scenario: Dict[str, Any], endpoint: str):
        """Test a single privacy decision scenario."""
        try:
            response = requests.post(
                f"{self.api_base_url}{endpoint}",
                json=scenario["data"],
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                decision = result.get("decision", "UNKNOWN")
                confidence = result.get("confidence", 0.0)
                integration_method = result.get("integration_method", "unknown")
                
                # Check expectations
                expected = scenario["expected"]
                decision_correct = decision == expected["decision"]
                confidence_adequate = confidence >= expected.get("confidence_min", 0.5)
                emergency_correct = result.get("emergency_override_used", False) == expected.get("emergency_override_used", False)
                
                success = decision_correct and confidence_adequate and emergency_correct
                
                print(f"{'✅' if success else '❌'} {scenario['name']}: {decision}")
                print(f"   Confidence: {confidence:.2f} (min: {expected.get('confidence_min', 0.5)})")
                print(f"   Integration: {integration_method}")
                print(f"   Emergency Override: {result.get('emergency_override_used', False)}")
                
                self._record_test_result(
                    f"{endpoint} - {scenario['name']}", 
                    success, 
                    f"Decision: {decision}, Confidence: {confidence:.2f}, Method: {integration_method}"
                )
            else:
                print(f"❌ {scenario['name']} failed: {response.status_code}")
                self._record_test_result(
                    f"{endpoint} - {scenario['name']}", 
                    False, 
                    f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            print(f"❌ {scenario['name']} error: {e}")
            self._record_test_result(
                f"{endpoint} - {scenario['name']}", 
                False, 
                str(e)
            )
    
    async def _test_error_handling(self):
        """Test error handling and edge cases."""
        print("\n⚠️  Testing Error Handling")
        print("-" * 40)
        
        error_test_cases = [
            {
                "name": "Missing Required Fields",
                "endpoint": "/privacy-decision",
                "data": {"data_field": "test"},  # Missing required fields
                "expected_status": 422
            },
            {
                "name": "Invalid Temporal Context",
                "endpoint": "/temporal-privacy-decision", 
                "data": {
                    "data_field": "test_data",
                    "requester_role": "test_role",
                    "context": "test_context",
                    "temporal_context": {
                        "situation": "test",
                        "urgency_level": "invalid_level"  # Invalid urgency level
                    }
                },
                "expected_status": 422
            }
        ]
        
        for test_case in error_test_cases:
            try:
                response = requests.post(
                    f"{self.api_base_url}{test_case['endpoint']}",
                    json=test_case["data"],
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                status_correct = response.status_code == test_case["expected_status"]
                
                print(f"{'✅' if status_correct else '❌'} {test_case['name']}: HTTP {response.status_code}")
                
                self._record_test_result(
                    f"Error Handling - {test_case['name']}", 
                    status_correct, 
                    f"Expected {test_case['expected_status']}, got {response.status_code}"
                )
                
            except Exception as e:
                print(f"❌ {test_case['name']} error: {e}")
                self._record_test_result(
                    f"Error Handling - {test_case['name']}", 
                    False, 
                    str(e)
                )
    
    def _record_test_result(self, test_name: str, passed: bool, details: str):
        """Record a test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        
        self.test_results.append({
            "test_name": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"\n📊 Integration Test Report")
        print("=" * 50)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Categorize results
        failed_tests = [r for r in self.test_results if not r["passed"]]
        
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   - {test['test_name']}: {test['details']}")
        else:
            print(f"\n✅ All tests passed!")
        
        return {
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": success_rate
            },
            "test_results": self.test_results,
            "failed_tests": failed_tests,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

async def run_integration_tests():
    """Run complete integration test suite."""
    print("🔗 Team A + Team C Integration Test Suite")
    print("Testing enhanced privacy decisions with temporal context")
    print()
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8003/health", timeout=5)
        if response.status_code != 200:
            print("❌ API not accessible at http://localhost:8003")
            print("   Please start the enhanced API service first:")
            print("   python enhanced_privacy_api_service.py")
            return
    except Exception:
        print("❌ API not running at http://localhost:8003")
        print("   Please start the enhanced API service first:")
        print("   python enhanced_privacy_api_service.py")
        return
    
    # Run tests
    test_runner = IntegratedTestRunner()
    report = await test_runner.run_all_tests()
    
    # Save report
    report_path = Path(__file__).parent / "integration_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    asyncio.run(run_integration_tests())