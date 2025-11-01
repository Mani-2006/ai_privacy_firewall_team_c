"""
AI Privacy Firewall - FIXED Privacy Ontology
Team C: Ken Mani Binu & Nibin Thomas
"""

from owlready2 import *
from typing import Dict, List
import json

class AIPrivacyOntology:
    """Core ontology for contextual privacy decisions - FIXED VERSION"""
    
    def __init__(self):
        self.onto = get_ontology("http://ai-privacy-firewall.org/ontology/")
        self.create_base_ontology()
        
    def create_base_ontology(self):
        """Create the foundational privacy ontology structure"""
        
        with self.onto:
            # === CORE CLASSES ===
            
            # Data Categories
            class DataType(Thing): pass
            class MedicalData(DataType): pass
            class FinancialData(DataType): pass  
            class PersonalData(DataType): pass
            class IdentificationData(DataType): pass
            
            # Context Types
            class Context(Thing): pass
            class MedicalContext(Context): pass
            class ITContext(Context): pass
            class HRContext(Context): pass
            class LegalContext(Context): pass
            
            # Actor Types
            class Actor(Thing): pass
            class Doctor(Actor): pass
            class Nurse(Actor): pass
            class ITAdmin(Actor): pass
            class HRStaff(Actor): pass
            class Patient(Actor): pass
            
            # Purpose Categories
            class Purpose(Thing): pass
            class Treatment(Purpose): pass
            class Diagnosis(Purpose): pass
            class Billing(Purpose): pass
            class Research(Purpose): pass
            class ITTroubleshooting(Purpose): pass
            
            # Sensitivity Levels
            class SensitivityLevel(Thing): pass
            class PublicData(SensitivityLevel): pass
            class InternalData(SensitivityLevel): pass
            class ConfidentialData(SensitivityLevel): pass
            class RestrictedData(SensitivityLevel): pass
            
        print("✅ Base privacy ontology created")
        
    def classify_data_field(self, field_name: str, context: str = None) -> Dict:
        """Classify a data field using ontological reasoning"""
        
        field_lower = field_name.lower()
        
        # Basic classification rules
        classification = {
            "field": field_name,
            "data_type": "PersonalData",
            "sensitivity": "InternalData",
            "context_dependent": False,
            "equivalents": [],
            "reasoning": []
        }
        
        # Medical data detection
        medical_keywords = ["patient", "diagnosis", "medical", "health", "symptom", "treatment"]
        if any(keyword in field_lower for keyword in medical_keywords):
            classification["data_type"] = "MedicalData"
            classification["sensitivity"] = "RestrictedData"
            classification["reasoning"].append("Contains medical terminology")
            
        # Financial data detection  
        financial_keywords = ["salary", "payment", "billing", "account", "financial", "credit"]
        if any(keyword in field_lower for keyword in financial_keywords):
            classification["data_type"] = "FinancialData"
            classification["sensitivity"] = "ConfidentialData"
            classification["reasoning"].append("Contains financial terminology")
            
        # Identification data
        id_keywords = ["ssn", "social security", "id", "identifier", "employee_id", "badge"]
        if any(keyword in field_lower for keyword in id_keywords):
            classification["data_type"] = "IdentificationData" 
            classification["sensitivity"] = "RestrictedData"
            classification["reasoning"].append("Contains identification data")
            
        # Context-dependent disambiguation
        if context:
            if context.lower() == "medical" and "diagnosis" in field_lower:
                classification["data_type"] = "MedicalData"
                classification["context_dependent"] = True
                classification["reasoning"].append("Medical context: clinical diagnosis")
                
            elif context.lower() == "it" and "diagnosis" in field_lower:
                classification["data_type"] = "PersonalData"  # IT troubleshooting data
                classification["sensitivity"] = "InternalData"
                classification["context_dependent"] = True
                classification["reasoning"].append("IT context: system troubleshooting")
                
        return classification
        
    def make_privacy_decision(self, requester: str, data_field: str, 
                            purpose: str, context: str = None, 
                            emergency: bool = False) -> Dict:
        """Make a privacy decision using ontological reasoning - FIXED VERSION"""
        
        # Step 1: Classify the data
        data_classification = self.classify_data_field(data_field, context)
        
        # Step 2: Determine access decision
        allowed = True
        reason = []
        confidence = 0.8
        
        # Define role categories
        medical_roles = ["doctor", "dr", "physician", "nurse", "md", "medical"]
        it_roles = ["it", "admin", "tech", "support", "engineer"]
        hr_roles = ["hr", "manager", "supervisor"]
        
        # Check requester role
        requester_lower = requester.lower()
        is_medical = any(role in requester_lower for role in medical_roles)
        is_it = any(role in requester_lower for role in it_roles)
        is_hr = any(role in requester_lower for role in hr_roles)
        
        # FIXED: Medical data access logic
        if data_classification["data_type"] == "MedicalData":
            if emergency:
                reason.append("Emergency override for medical data")
                confidence = 0.95
                allowed = True
            elif is_medical:
                reason.append("Medical professional accessing medical data")
                confidence = 0.90
                allowed = True
            else:
                reason.append("Medical data requires medical professional access")
                confidence = 0.95
                allowed = False
                
        # Financial data access logic
        elif data_classification["data_type"] == "FinancialData":
            if is_hr and purpose.lower() in ["review", "management", "audit"]:
                reason.append("HR staff accessing financial data for authorized purpose")
                confidence = 0.85
                allowed = True
            elif emergency:
                reason.append("Emergency override for financial data")
                confidence = 0.90
                allowed = True
            else:
                reason.append("Financial data requires HR authorization")
                confidence = 0.90
                allowed = False
                
        # IT data access logic
        elif data_classification["data_type"] == "PersonalData" and context == "it":
            if is_it:
                reason.append("IT professional accessing system data")
                confidence = 0.85
                allowed = True
            else:
                reason.append("IT system data requires IT role")
                confidence = 0.80
                allowed = False
                
        # Context-specific purpose validation
        if allowed and context == "medical" and purpose.lower() == "treatment":
            reason.append("Valid medical treatment purpose")
            confidence = min(confidence + 0.05, 1.0)
            
        if not reason:
            reason.append("Standard access policy applied")
            
        return {
            "allowed": allowed,
            "reason": "; ".join(reason),
            "confidence": confidence,
            "data_classification": data_classification,
            "emergency_used": emergency,
            "requester_roles": {
                "medical": is_medical,
                "it": is_it, 
                "hr": is_hr
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }

# This is the BRAIN of your privacy system - makes intelligent decisions about data access

# Test the FIXED ontology
if __name__ == "__main__":
    print("🧠 Testing FIXED AI Privacy Ontology...")
    
    ontology = AIPrivacyOntology()
    
    # Load test cases from external file
    import os
    import json
    
    test_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "test_cases.json")
    
    try:
        with open(test_file_path, 'r') as f:
            test_config = json.load(f)
        
        # Run basic ontology tests
        basic_tests = test_config["test_categories"]["basic_ontology_tests"]
        
        print(f"\n🎯 Running {len(basic_tests)} Basic Ontology Tests:")
        
        for test_case in basic_tests:
            test_input = test_case["input"]
            decision = ontology.make_privacy_decision(
                requester=test_input["requester"],
                data_field=test_input["data_field"],
                purpose=test_input["purpose"],
                context=test_input.get("context"),
                emergency=test_input.get("emergency", False)
            )
            
            print(f"\n📋 {test_case['test_id']}: {test_case['name']}")
            print(f"   Input: {test_input['requester']} → {test_input['data_field']}")
            print(f"   Decision: {'✅ ALLOWED' if decision['allowed'] else '❌ DENIED'}")
            print(f"   Reason: {decision['reason']}")
            print(f"   Confidence: {decision['confidence']:.2f}")
        
        print(f"\n💡 For comprehensive testing, run: python tests/test_runner.py")
        
    except FileNotFoundError:
        print("⚠️  Test cases file not found. Create tests/test_cases.json first.")
        print("💡 Using fallback inline tests...")
        
        # Fallback to minimal inline tests
        fallback_tests = [
            ("dr_jones", "medical_diagnosis", "treatment", "medical", False),
            ("it_admin", "system_diagnosis", "troubleshooting", "it", False),
        ]
        
        for requester, data_field, purpose, context, emergency in fallback_tests:
            decision = ontology.make_privacy_decision(requester, data_field, purpose, context, emergency)
            print(f"\n📋 {requester} → {data_field}: {'✅ ALLOWED' if decision['allowed'] else '❌ DENIED'}")
