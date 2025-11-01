#!/usr/bin/env python3
"""
Security-Fixed Privacy Ontology (No OWL Dependencies)
====================================================

This module provides privacy classification without owlready2 dependencies,
eliminating macOS security warnings while maintaining full functionality.

Author: Team C Privacy Firewall
Date: 2024-12-30
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class SecurityFixedPrivacyOntology:
    """Privacy ontology without owlready2 dependencies - no security warnings"""
    
    def __init__(self):
        """Initialize with predefined classification rules"""
        self.data_types = {
            "medical": ["patient", "diagnosis", "medical", "health", "symptom", "treatment", 
                       "blood", "cardiac", "mental", "therapy", "prescription", "surgery"],
            "financial": ["salary", "payment", "billing", "account", "financial", "credit",
                         "transaction", "invoice", "compensation", "revenue", "cost"],
            "identification": ["ssn", "social security", "id", "identifier", "employee_id", 
                              "badge", "license", "passport", "driver", "certificate"],
            "personal": ["name", "email", "phone", "address", "contact", "profile", "bio"]
        }
        
        self.sensitivity_levels = {
            "public": 0,
            "internal": 1, 
            "confidential": 2,
            "restricted": 3
        }
        
        self.role_permissions = {
            "doctor": ["medical", "personal", "identification"],
            "nurse": ["medical", "personal"],
            "hr_manager": ["financial", "personal", "identification"],
            "auditor": ["financial", "personal"],
            "it_admin": ["personal"],
            "patient": ["personal"],
            "manager": ["personal", "financial"],
            "employee": ["personal"]
        }
        
        print("✅ Security-fixed privacy ontology initialized")
        
    def classify_data_field(self, field_name: str, context: str = None) -> Dict[str, Any]:
        """Classify data field using rule-based approach"""
        field_lower = field_name.lower()
        
        # Default classification
        classification = {
            "field": field_name,
            "data_type": "PersonalData",
            "sensitivity": "InternalData", 
            "confidence": 0.5,
            "reasoning": [],
            "context_dependent": False,
            "equivalents": []
        }
        
        # Classify by keyword matching
        for data_type, keywords in self.data_types.items():
            if any(keyword in field_lower for keyword in keywords):
                if data_type == "medical":
                    classification.update({
                        "data_type": "MedicalData",
                        "sensitivity": "RestrictedData",
                        "confidence": 0.9,
                        "reasoning": ["Contains medical terminology"]
                    })
                elif data_type == "financial":
                    classification.update({
                        "data_type": "FinancialData", 
                        "sensitivity": "ConfidentialData",
                        "confidence": 0.85,
                        "reasoning": ["Contains financial terminology"]
                    })
                elif data_type == "identification":
                    classification.update({
                        "data_type": "IdentificationData",
                        "sensitivity": "RestrictedData",
                        "confidence": 0.95,
                        "reasoning": ["Contains identification data"]
                    })
                elif data_type == "personal":
                    classification.update({
                        "data_type": "PersonalData",
                        "sensitivity": "InternalData",
                        "confidence": 0.7,
                        "reasoning": ["Contains personal information"]
                    })
                break
        
        # Context-based refinement
        if context:
            context_lower = context.lower()
            if "emergency" in context_lower:
                classification["context_dependent"] = True
                classification["reasoning"].append("Emergency context detected")
            elif "audit" in context_lower:
                classification["context_dependent"] = True
                classification["reasoning"].append("Audit context detected")
                
        return classification
        
    def check_role_access(self, requester_role: str, data_type: str) -> Dict[str, Any]:
        """Check if role can access data type"""
        role_lower = requester_role.lower()
        data_type_key = self._map_data_type_to_key(data_type)
        
        allowed_types = self.role_permissions.get(role_lower, ["personal"])
        access_allowed = data_type_key in allowed_types
        
        return {
            "access_allowed": access_allowed,
            "role": requester_role,
            "data_type": data_type,
            "reasoning": f"Role '{requester_role}' {'can' if access_allowed else 'cannot'} access {data_type}",
            "allowed_types": allowed_types
        }
        
    def _map_data_type_to_key(self, data_type: str) -> str:
        """Map ontology data type to permission key"""
        mapping = {
            "MedicalData": "medical",
            "FinancialData": "financial", 
            "IdentificationData": "identification",
            "PersonalData": "personal"
        }
        return mapping.get(data_type, "personal")
        
    def get_privacy_decision(self, field_name: str, requester_role: str, context: str = None) -> Dict[str, Any]:
        """Get comprehensive privacy decision"""
        classification = self.classify_data_field(field_name, context)
        role_check = self.check_role_access(requester_role, classification["data_type"])
        
        # Calculate confidence
        base_confidence = classification["confidence"]
        role_confidence = 0.8 if role_check["access_allowed"] else 0.2
        final_confidence = (base_confidence + role_confidence) / 2
        
        decision = "ALLOW" if role_check["access_allowed"] else "DENY"
        
        return {
            "decision": decision,
            "confidence": round(final_confidence, 2),
            "classification": classification,
            "role_analysis": role_check,
            "reasoning": f"{role_check['reasoning']}. {', '.join(classification['reasoning'])}",
            "timestamp": datetime.now().isoformat()
        }

# Compatibility alias for existing code
AIPrivacyOntology = SecurityFixedPrivacyOntology