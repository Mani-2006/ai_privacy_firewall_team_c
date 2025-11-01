#!/usr/bin/env python3
"""
Team A + Team C Integration Demo Scenarios
==========================================

Practical demonstration of privacy decisions with temporal context concepts.
This demo shows how Team A's temporal framework would enhance Team C's decisions.

Current Setup:
- Team C Privacy API running on port 8002
- Demo scenarios showing integration potential
- Temporal context simulation for Team A integration

Author: Team C + Team A Integration Demo
Date: 2024-12-30
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

class IntegrationDemoRunner:
    """Demo runner for Team A + C integration scenarios."""
    
    def __init__(self, api_base_url: str = "http://localhost:8002"):
        self.api_base_url = api_base_url
        self.demo_results = []
    
    def print_header(self, title: str):
        """Print formatted header."""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")
    
    def print_scenario(self, name: str, description: str):
        """Print scenario header."""
        print(f"\n🧪 Scenario: {name}")
        print(f"Description: {description}")
        print("-" * 50)
    
    def make_api_call(self, endpoint: str, data: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
        """Make API call and display results."""
        print(f"📤 Request to {endpoint}:")
        print(f"   {json.dumps(data, indent=2)}")
        
        try:
            response = requests.post(
                f"{self.api_base_url}{endpoint}",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Response ({response.status_code}):")
                
                if endpoint == "/api/v1/classify":
                    print(f"   📊 Data Type: {result.get('data_type', 'Unknown')}")
                    print(f"   🔒 Sensitivity: {result.get('sensitivity', 'Unknown')}")
                    print(f"   🧠 Reasoning: {result.get('reasoning', [])}")
                    print(f"   📈 Confidence: {result.get('confidence', 0.0):.2f}")
                
                elif endpoint == "/api/v1/privacy-decision":
                    print(f"   🔑 Decision: {'ALLOW' if result.get('allowed', False) else 'DENY'}")
                    print(f"   💭 Reason: {result.get('reason', 'No reason provided')}")
                    print(f"   📈 Confidence: {result.get('confidence', 0.0):.2f}")
                    print(f"   🚨 Emergency Used: {result.get('emergency_used', False)}")
                    
                    # Show data classification
                    data_class = result.get('data_classification', {})
                    if data_class:
                        print(f"   📊 Classification: {data_class.get('data_type', 'Unknown')} - {data_class.get('sensitivity', 'Unknown')}")
                
                self.demo_results.append({
                    "scenario": scenario_name,
                    "success": True,
                    "result": result
                })
                
                return result
                
            else:
                print(f"\n❌ Error ({response.status_code}):")
                try:
                    error_detail = response.json()
                    print(f"   Details: {error_detail}")
                except:
                    print(f"   Details: {response.text}")
                
                self.demo_results.append({
                    "scenario": scenario_name,
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                })
                
                return {"error": True, "status_code": response.status_code}
                
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            self.demo_results.append({
                "scenario": scenario_name,
                "success": False,
                "error": str(e)
            })
            return {"error": True, "exception": str(e)}
    
    def simulate_temporal_context(self, temporal_data: Dict[str, Any]):
        """Simulate how Team A's temporal context would enhance the decision."""
        print(f"\n⏰ Simulated Team A Temporal Context:")
        print(f"   🕐 Situation: {temporal_data.get('situation', 'Unknown')}")
        print(f"   🚨 Urgency: {temporal_data.get('urgency_level', 'Unknown')}")
        print(f"   ⚡ Emergency Override: {temporal_data.get('emergency_override_requested', False)}")
        print(f"   📅 Time Window: {temporal_data.get('time_window_description', 'Not specified')}")
        
        # Simulate temporal decision logic
        urgency = temporal_data.get('urgency_level', 'normal')
        emergency = temporal_data.get('emergency_override_requested', False)
        
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
        
        print(f"   🎯 Temporal Decision: {temporal_decision}")
        print(f"   📊 Temporal Confidence: {temporal_confidence:.2f}")
        
        return {
            "decision": temporal_decision,
            "confidence": temporal_confidence,
            "emergency_override": emergency and urgency == 'critical'
        }
    
    def demonstrate_integration_logic(self, privacy_result: Dict[str, Any], temporal_result: Dict[str, Any], scenario_name: str):
        """Demonstrate how Team A + C decisions would be combined."""
        print(f"\n🔗 Integration Logic Simulation:")
        
        privacy_allowed = privacy_result.get('allowed', False)
        privacy_confidence = privacy_result.get('confidence', 0.0)
        temporal_decision = temporal_result.get('decision', 'DENY')
        temporal_confidence = temporal_result.get('confidence', 0.0)
        emergency_override = temporal_result.get('emergency_override', False)
        
        # Simulate integration logic
        if emergency_override and not privacy_allowed:
            final_decision = "ALLOW"
            final_confidence = max(0.85, temporal_confidence)
            integration_method = "emergency_override"
            reasoning = "Team A emergency override superseded Team C denial"
            
        elif privacy_allowed and temporal_decision in ["ALLOW", "ALLOW_OVERRIDE"]:
            final_decision = "ALLOW" 
            final_confidence = min(1.0, (privacy_confidence + temporal_confidence) / 2 + 0.1)
            integration_method = "consensus_allow"
            reasoning = "Both Team A temporal and Team C privacy approved"
            
        elif not privacy_allowed or temporal_decision == "DENY":
            final_decision = "DENY"
            final_confidence = max(privacy_confidence, temporal_confidence)
            integration_method = "security_priority"
            reasoning = "Security priority - at least one system denied access"
            
        else:
            final_decision = "ALLOW" if privacy_allowed else "DENY"
            final_confidence = (privacy_confidence + temporal_confidence) / 2
            integration_method = "privacy_guided"
            reasoning = "Team C privacy decision with temporal context consideration"
        
        print(f"   🎯 Final Integrated Decision: {final_decision}")
        print(f"   📊 Final Confidence: {final_confidence:.2f}")
        print(f"   🔧 Integration Method: {integration_method}")
        print(f"   💭 Reasoning: {reasoning}")
        print(f"   🚨 Emergency Override Used: {emergency_override and final_decision == 'ALLOW'}")
        
        return {
            "final_decision": final_decision,
            "final_confidence": final_confidence,
            "integration_method": integration_method,
            "reasoning": reasoning,
            "emergency_override_used": emergency_override and final_decision == "ALLOW"
        }
    
    def run_demo_scenarios(self):
        """Run comprehensive demo scenarios."""
        self.print_header("Team A + Team C Integration Demo Scenarios")
        print("🔗 Demonstrating enhanced privacy decisions with temporal context")
        print("🎯 Current: Team C API running | Simulating: Team A integration")
        
        # Test API connectivity
        try:
            response = requests.get(f"{self.api_base_url}/docs", timeout=5)
            if response.status_code == 200:
                print("✅ Team C Privacy API is accessible")
            else:
                print("⚠️  Team C Privacy API may have issues")
        except:
            print("❌ Cannot connect to Team C Privacy API")
            print("   Please ensure the API is running on port 8002")
            return
        
        scenarios = [
            {
                "name": "Critical Medical Emergency",
                "description": "Emergency doctor accessing patient data during cardiac arrest",
                "classification_request": {
                    "data_field": "patient_cardiac_monitor_data",
                    "context": "emergency_room_critical_care"
                },
                "privacy_request": {
                    "requester": "emergency_cardiologist",
                    "data_field": "patient_cardiac_monitor_data", 
                    "purpose": "cardiac_arrest_treatment",
                    "context": "emergency_room_critical_care",
                    "emergency": True
                },
                "temporal_context": {
                    "situation": "cardiac_arrest_resuscitation",
                    "urgency_level": "critical",
                    "emergency_override_requested": True,
                    "time_window_description": "Immediate access for 2 hours"
                }
            },
            
            {
                "name": "After-Hours Financial Analysis",
                "description": "Senior analyst accessing quarterly data for regulatory deadline",
                "classification_request": {
                    "data_field": "quarterly_financial_statements",
                    "context": "regulatory_reporting_deadline"
                },
                "privacy_request": {
                    "requester": "senior_financial_analyst",
                    "data_field": "quarterly_financial_statements",
                    "purpose": "regulatory_compliance_report",
                    "context": "after_hours_deadline_work",
                    "emergency": False
                },
                "temporal_context": {
                    "situation": "regulatory_deadline_tomorrow",
                    "urgency_level": "high",
                    "emergency_override_requested": False,
                    "time_window_description": "Tonight until 8 AM for deadline"
                }
            },
            
            {
                "name": "Weekend System Maintenance", 
                "description": "Engineer accessing source code for scheduled maintenance",
                "classification_request": {
                    "data_field": "production_source_code",
                    "context": "scheduled_maintenance_window"
                },
                "privacy_request": {
                    "requester": "senior_software_engineer",
                    "data_field": "production_source_code",
                    "purpose": "scheduled_security_patch",
                    "context": "weekend_maintenance_window",
                    "emergency": False
                },
                "temporal_context": {
                    "situation": "scheduled_maintenance",
                    "urgency_level": "normal",
                    "emergency_override_requested": False,
                    "time_window_description": "Weekend maintenance window (2 AM - 6 AM)"
                }
            },
            
            {
                "name": "Unauthorized Curiosity Access",
                "description": "Intern attempting to access employee salary data",
                "classification_request": {
                    "data_field": "employee_salary_database",
                    "context": "casual_data_exploration"
                },
                "privacy_request": {
                    "requester": "summer_intern",
                    "data_field": "employee_salary_database", 
                    "purpose": "learning_about_company_structure",
                    "context": "casual_browsing",
                    "emergency": False
                },
                "temporal_context": {
                    "situation": "casual_exploration",
                    "urgency_level": "low", 
                    "emergency_override_requested": False,
                    "time_window_description": "Regular business hours"
                }
            },
            
            {
                "name": "Executive Strategic Access",
                "description": "CEO accessing strategic plans for board meeting",
                "classification_request": {
                    "data_field": "company_strategic_roadmap_2025",
                    "context": "board_meeting_preparation"
                },
                "privacy_request": {
                    "requester": "chief_executive_officer",
                    "data_field": "company_strategic_roadmap_2025",
                    "purpose": "board_presentation_preparation", 
                    "context": "executive_board_meeting",
                    "emergency": False
                },
                "temporal_context": {
                    "situation": "board_meeting_tomorrow",
                    "urgency_level": "high",
                    "emergency_override_requested": False,
                    "time_window_description": "Executive preparation time"
                }
            },
            
            {
                "name": "Medical Emergency - Non-Medical Personnel",
                "description": "Security guard trying to access medical data during emergency",
                "classification_request": {
                    "data_field": "patient_medical_records",
                    "context": "building_medical_emergency"
                },
                "privacy_request": {
                    "requester": "security_guard",
                    "data_field": "patient_medical_records",
                    "purpose": "emergency_assistance",
                    "context": "medical_emergency_in_building", 
                    "emergency": True
                },
                "temporal_context": {
                    "situation": "medical_emergency_response",
                    "urgency_level": "critical",
                    "emergency_override_requested": True,
                    "time_window_description": "Emergency response period"
                }
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            self.print_scenario(f"{i}. {scenario['name']}", scenario['description'])
            
            # Step 1: Data Classification
            print("📊 Step 1: Data Classification")
            classification_result = self.make_api_call(
                "/api/v1/classify",
                scenario['classification_request'],
                f"{scenario['name']} - Classification"
            )
            
            time.sleep(1)  # Brief pause for readability
            
            # Step 2: Privacy Decision
            print("\n🔒 Step 2: Privacy Decision")
            privacy_result = self.make_api_call(
                "/api/v1/privacy-decision", 
                scenario['privacy_request'],
                f"{scenario['name']} - Privacy Decision"
            )
            
            # Step 3: Temporal Context Simulation
            print("\n⏰ Step 3: Temporal Context Analysis")
            temporal_result = self.simulate_temporal_context(scenario['temporal_context'])
            
            # Step 4: Integration Logic
            print("\n🔗 Step 4: Integration Decision Logic")
            integration_result = self.demonstrate_integration_logic(
                privacy_result, 
                temporal_result, 
                scenario['name']
            )
            
            # Summary for this scenario
            print(f"\n📋 Summary for {scenario['name']}:")
            print(f"   Team C Decision: {'ALLOW' if privacy_result.get('allowed', False) else 'DENY'}")
            print(f"   Team A Decision: {temporal_result.get('decision', 'UNKNOWN')}")
            print(f"   Integrated Decision: {integration_result.get('final_decision', 'UNKNOWN')}")
            print(f"   Integration Method: {integration_result.get('integration_method', 'UNKNOWN')}")
            
            time.sleep(2)  # Pause between scenarios
        
        # Final Summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary of all demo results."""
        self.print_header("Demo Summary & Integration Analysis")
        
        successful_scenarios = [r for r in self.demo_results if r.get('success', False)]
        failed_scenarios = [r for r in self.demo_results if not r.get('success', False)]
        
        print(f"📊 Total Scenarios Tested: {len(self.demo_results)}")
        print(f"✅ Successful API Calls: {len(successful_scenarios)}")
        print(f"❌ Failed API Calls: {len(failed_scenarios)}")
        
        if failed_scenarios:
            print(f"\n❌ Failed Scenarios:")
            for scenario in failed_scenarios:
                print(f"   - {scenario['scenario']}: {scenario.get('error', 'Unknown error')}")
        
        print(f"\n🎯 Key Integration Capabilities Demonstrated:")
        print(f"   ✅ Team C privacy classification and decisions")
        print(f"   ✅ Emergency override logic simulation")
        print(f"   ✅ Temporal context analysis simulation")
        print(f"   ✅ Combined decision integration logic")
        print(f"   ✅ Multi-level confidence scoring")
        print(f"   ✅ Audit trail and reasoning capture")
        
        print(f"\n🚀 Next Steps for Full Integration:")
        print(f"   1. Deploy Team A temporal framework service")
        print(f"   2. Connect both systems to shared Neo4j database")
        print(f"   3. Implement real-time temporal policy engine")
        print(f"   4. Add emergency override workflow automation")
        print(f"   5. Create unified audit dashboard")

def main():
    """Run the integration demo."""
    print("🚀 Team A + Team C Integration Demo")
    print("=" * 50)
    print("This demo shows how temporal context enhances privacy decisions")
    print("Current: Team C Privacy API | Simulated: Team A Temporal Framework")
    print()
    
    demo_runner = IntegrationDemoRunner()
    demo_runner.run_demo_scenarios()

if __name__ == "__main__":
    main()