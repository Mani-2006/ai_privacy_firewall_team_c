"""
Complete Test Runner for AI Privacy Firewall - Team C
Integrated demo tests and comprehensive test runner
Team C: Ken & Nibin
"""

import sys
import asyncio
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Fix Python paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from integration.graphiti_privacy_bridge import GraphitiPrivacyBridge
from ontology.privacy_ontology import AIPrivacyOntology

class PrivacyTestRunner:
    """Complete test runner for privacy ontology and bridge integration"""
    
    def __init__(self):
        self.ontology = AIPrivacyOntology()
        self.bridge = None
        
        # Load test configuration
        test_file_path = os.path.join(current_dir, "test_cases.json")
        
        try:
            with open(test_file_path, 'r') as f:
                self.test_config = json.load(f)
        except FileNotFoundError:
            print("⚠️  Test cases file not found. Using fallback scenarios...")
            self.test_config = self._get_fallback_config()
            
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "categories": {},
            "start_time": None,
            "end_time": None
        }
    
    def _get_fallback_config(self):
        """Fallback test configuration if file is missing"""
        return {
            "8_week_demo_scenarios": [
                {
                    "test_id": "DEMO_001",
                    "name": "Emergency Access",
                    "input": {
                        "requester": "dr_emergency", 
                        "data_field": "patient_medical_record",
                        "purpose": "emergency_treatment",
                        "context": "medical",
                        "emergency": True
                    },
                    "expected_result": {
                        "allowed": True,
                        "should_contain_reason": ["emergency override"],
                        "min_confidence": 0.95
                    }
                },
                {
                    "test_id": "DEMO_002",
                    "name": "Temporal Boundaries", 
                    "input": {
                        "requester": "contractor_john",
                        "data_field": "project_financial_data", 
                        "purpose": "project_work",
                        "context": "financial",
                        "emergency": False
                    },
                    "expected_result": {
                        "allowed": False,
                        "should_contain_reason": ["financial data requires", "HR authorization"],
                        "min_confidence": 0.85
                    }
                },
                {
                    "test_id": "DEMO_003",
                    "name": "Organizational Hierarchy",
                    "input": {
                        "requester": "manager_sarah",
                        "data_field": "team_performance_data",
                        "purpose": "management_review", 
                        "context": "hr",
                        "emergency": False
                    },
                    "expected_result": {
                        "allowed": True,
                        "should_contain_reason": ["standard access policy"],
                        "min_confidence": 0.75
                    }
                },
                {
                    "test_id": "DEMO_004",
                    "name": "Semantic Understanding - Medical",
                    "input": {
                        "requester": "doctor",
                        "data_field": "patient_diagnosis",
                        "purpose": "treatment",
                        "context": "medical", 
                        "emergency": False
                    },
                    "expected_result": {
                        "allowed": True,
                        "should_contain_reason": ["medical professional", "medical treatment purpose"],
                        "min_confidence": 0.90
                    }
                },
                {
                    "test_id": "DEMO_005",
                    "name": "Semantic Understanding - IT",
                    "input": {
                        "requester": "it_admin",
                        "data_field": "system_diagnosis", 
                        "purpose": "troubleshooting",
                        "context": "it",
                        "emergency": False
                    },
                    "expected_result": {
                        "allowed": True,
                        "should_contain_reason": ["IT professional", "system data"],
                        "min_confidence": 0.85
                    }
                }
            ]
        }
    
    async def setup_bridge(self):
        """Setup bridge for integration tests"""
        try:
            self.bridge = GraphitiPrivacyBridge()
            return True
        except Exception as e:
            print(f"⚠️  Bridge setup failed: {e}")
            return False
    
    def run_ontology_test(self, test_case: Dict) -> Dict:
        """Run a single ontology test case"""
        test_input = test_case["input"]
        expected = test_case["expected_result"]
        
        # Execute the test
        try:
            result = self.ontology.make_privacy_decision(
                requester=test_input["requester"],
                data_field=test_input["data_field"],
                purpose=test_input["purpose"],
                context=test_input.get("context"),
                emergency=test_input.get("emergency", False)
            )
            
            # Validate results
            test_result = {
                "test_id": test_case["test_id"],
                "name": test_case["name"],
                "passed": True,
                "issues": [],
                "actual_result": result
            }
            
            # Check allowed/denied
            if "allowed" in expected:
                if result["allowed"] != expected["allowed"]:
                    test_result["passed"] = False
                    test_result["issues"].append(f"Expected allowed={expected['allowed']}, got {result['allowed']}")
            
            # Check confidence threshold
            if "min_confidence" in expected:
                if result["confidence"] < expected["min_confidence"]:
                    test_result["passed"] = False
                    test_result["issues"].append(f"Confidence {result['confidence']:.2f} below minimum {expected['min_confidence']}")
            
            # Check reason contains expected text
            if "should_contain_reason" in expected:
                reason_lower = result["reason"].lower()
                for expected_text in expected["should_contain_reason"]:
                    if expected_text.lower() not in reason_lower:
                        test_result["passed"] = False
                        test_result["issues"].append(f"Reason should contain '{expected_text}'")
            
            return test_result
            
        except Exception as e:
            return {
                "test_id": test_case["test_id"],
                "name": test_case["name"],
                "passed": False,
                "issues": [f"Exception during test: {str(e)}"],
                "actual_result": None
            }
    
    async def run_bridge_test(self, test_case: Dict) -> Dict:
        """Run a test case that requires bridge integration"""
        if not self.bridge:
            return {
                "test_id": test_case["test_id"],
                "name": test_case["name"],
                "passed": False,
                "issues": ["Bridge not available for integration test"],
                "actual_result": None
            }
        
        test_input = test_case["input"]
        
        try:
            # Run through bridge
            decision = await self.bridge.create_privacy_decision_episode(test_input)
            classification = await self.bridge.create_data_entity(
                test_input["data_field"],
                test_input.get("context")
            )
            
            return {
                "test_id": test_case["test_id"],
                "name": test_case["name"],
                "passed": True,
                "issues": [],
                "actual_result": {
                    "decision": decision,
                    "classification": classification
                }
            }
            
        except Exception as e:
            return {
                "test_id": test_case["test_id"],
                "name": test_case["name"],
                "passed": False,
                "issues": [f"Bridge test exception: {str(e)}"],
                "actual_result": None
            }
    
    async def run_demo_scenarios(self, include_bridge: bool = False):
        """Run the 5 demo scenarios with validation"""
        
        print("🧪 Testing Your 5 Demo Scenarios...")
        print("=" * 50)
        
        demo_scenarios = self.test_config.get("8_week_demo_scenarios", [])
        
        if include_bridge:
            bridge_ready = await self.setup_bridge()
            if not bridge_ready:
                print("⚠️  Running ontology tests only (bridge unavailable)")
                include_bridge = False
        
        results = []
        
        for i, test_case in enumerate(demo_scenarios, 1):
            print(f"\n{i}. Testing: {test_case['name']}")
            
            if include_bridge:
                result = await self.run_bridge_test(test_case)
            else:
                result = self.run_ontology_test(test_case)
            
            # Check if result matches expected
            expected = test_case["expected_result"]
            
            if result["passed"]:
                if include_bridge:
                    actual_result = "ALLOWED" if result["actual_result"]["decision"]["allowed"] else "DENIED"
                else:
                    actual_result = "ALLOWED" if result["actual_result"]["allowed"] else "DENIED"
                    
                expected_result = "ALLOWED" if expected["allowed"] else "DENIED"
                
                if actual_result == expected_result:
                    print(f"   ✅ TEST PASSED: {actual_result}")
                    results.append((test_case['name'], True))
                else:
                    print(f"   ❌ TEST FAILED: Expected {expected_result}, got {actual_result}")
                    results.append((test_case['name'], False))
            else:
                print(f"   ❌ TEST ERROR: {', '.join(result['issues'])}")
                results.append((test_case['name'], False))
        
        # Summary
        print("\n" + "=" * 50)
        print("🎯 TEST RESULTS:")
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🏆 Overall: {passed}/{len(results)} scenarios passed ({passed/len(results)*100:.0f}%)")
        
        if self.bridge:
            await self.bridge.close()
        
        return passed == len(results)
    
    async def run_comprehensive_tests(self, include_bridge: bool = False):
        """Run all test categories comprehensively"""
        self.results["start_time"] = datetime.now()
        
        print("🚀 Starting Comprehensive AI Privacy Firewall Test Suite")
        print(f"📋 Test Suite: AI Privacy Firewall - Team C")
        print("=" * 80)
        
        # Setup bridge if needed
        if include_bridge:
            bridge_ready = await self.setup_bridge()
            if not bridge_ready:
                print("⚠️  Running ontology tests only (bridge unavailable)")
                include_bridge = False
        
        # Run demo scenarios
        demo_scenarios = self.test_config.get("8_week_demo_scenarios", [])
        
        print(f"\n🧪 Running {len(demo_scenarios)} Demo Scenarios:")
        print("-" * 60)
        
        category_results = {"passed": 0, "failed": 0, "tests": []}
        
        for test_case in demo_scenarios:
            if include_bridge:
                result = await self.run_bridge_test(test_case)
            else:
                result = self.run_ontology_test(test_case)
            
            # Update counters
            if result["passed"]:
                category_results["passed"] += 1
                print(f"✅ {result['test_id']}: {result['name']}")
            else:
                category_results["failed"] += 1
                print(f"❌ {result['test_id']}: {result['name']}")
                for issue in result["issues"]:
                    print(f"   💥 {issue}")
            
            category_results["tests"].append(result)
            self.results["total_tests"] += 1
        
        self.results["categories"]["demo_scenarios"] = category_results
        self.results["passed"] += category_results["passed"]
        self.results["failed"] += category_results["failed"]
        
        self.results["end_time"] = datetime.now()
        
        # Print summary
        self.print_summary()
        
        if self.bridge:
            await self.bridge.close()
        
        return self.results
    
    def print_summary(self):
        """Print test results summary"""
        total = self.results["total_tests"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        
        duration = (self.results["end_time"] - self.results["start_time"]).total_seconds()
        
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        print("\n📋 Results by Category:")
        for category, result in self.results["categories"].items():
            total_cat = result["passed"] + result["failed"]
            pass_rate = result["passed"] / total_cat * 100 if total_cat > 0 else 0
            print(f"  {category}: {result['passed']}/{total_cat} ({pass_rate:.1f}%)")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Your privacy ontology is working perfectly!")
        else:
            print(f"\n⚠️  {failed} tests failed. Review the issues above.")

# Main execution functions
async def run_demo_tests():
    """Run the 5 demo scenarios and validate results (simple version)"""
    runner = PrivacyTestRunner()
    return await runner.run_demo_scenarios(include_bridge=False)

async def run_bridge_demo_tests():
    """Run the 5 demo scenarios with bridge integration"""
    runner = PrivacyTestRunner()
    return await runner.run_demo_scenarios(include_bridge=True)

async def run_comprehensive_tests():
    """Run comprehensive test suite"""
    runner = PrivacyTestRunner()
    return await runner.run_comprehensive_tests(include_bridge=False)

async def run_comprehensive_bridge_tests():
    """Run comprehensive test suite with bridge integration"""
    runner = PrivacyTestRunner()
    return await runner.run_comprehensive_tests(include_bridge=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Privacy Firewall Test Runner")
    parser.add_argument("--bridge", action="store_true", help="Include bridge integration tests")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        if args.bridge:
            asyncio.run(run_comprehensive_bridge_tests())
        else:
            asyncio.run(run_comprehensive_tests())
    else:
        if args.bridge:
            asyncio.run(run_bridge_demo_tests())
        else:
            asyncio.run(run_demo_tests())
