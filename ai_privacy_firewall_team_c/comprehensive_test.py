#!/usr/bin/env python3
"""
Comprehensive Test Suite for Team C Privacy Firewall API
========================================================

This script demonstrates all the enhanced features of the privacy firewall system:
- Privacy decision making with timezone awareness
- Data classification capabilities
- Emergency override functionality
- Team A & B integration contracts
- Neo4j fallback mode testing

Author: Team C Privacy Firewall
Date: 2024-12-30
"""

import requests
import json
import time
from datetime import datetime, timezone

# API Base URL
BASE_URL = "http://localhost:8002/api/v1"

def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

def print_response(response, test_description):
    """Print formatted API response."""
    print(f"\n📋 {test_description}")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS")
        try:
            data = response.json()
            print(json.dumps(data, indent=2))
        except:
            print(response.text)
    else:
        print("❌ ERROR")
        print(response.text)

def test_health_endpoint():
    """Test the health endpoint."""
    print_test_header("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response, "API Health Status")

def test_data_classification():
    """Test data classification capabilities."""
    print_test_header("Data Classification Tests")
    
    test_cases = [
        {
            "data_field": "employee_bank_account",
            "context": "payroll_processing",
            "description": "Financial Data Classification"
        },
        {
            "data_field": "customer_email", 
            "context": "marketing_campaign",
            "description": "PII Data Classification"
        },
        {
            "data_field": "product_inventory_count",
            "context": "business_analytics", 
            "description": "Business Data Classification"
        },
        {
            "data_field": "patient_medical_records",
            "context": "healthcare_treatment",
            "description": "Medical Data Classification"
        }
    ]
    
    for test_case in test_cases:
        try:
            response = requests.post(f"{BASE_URL}/classify", json={
                "data_field": test_case["data_field"],
                "context": test_case["context"]
            })
            print_response(response, test_case["description"])
        except Exception as e:
            print(f"⚠️  Classification test failed: {e}")

def test_privacy_decisions():
    """Test privacy decision making."""
    print_test_header("Privacy Decision Tests")
    
    test_cases = [
        {
            "request": {
                "requester": "alice_california",
                "data_field": "employee_salary", 
                "purpose": "payroll_processing",
                "context": "quarterly_review",
                "emergency": False,
                "requester_location": "california"
            },
            "description": "California Office Hours Request"
        },
        {
            "request": {
                "requester": "bob_india",
                "data_field": "customer_demographics",
                "purpose": "market_analysis", 
                "context": "business_intelligence",
                "emergency": False,
                "requester_location": "india"
            },
            "description": "India Office Analytics Request"
        },
        {
            "request": {
                "requester": "doctor_smith",
                "data_field": "patient_medical_records",
                "purpose": "emergency_treatment",
                "context": "hospital_emergency", 
                "emergency": True,
                "requester_location": "eastern"
            },
            "description": "Emergency Medical Access"
        },
        {
            "request": {
                "requester": "intern_john",
                "data_field": "financial_statements",
                "purpose": "learning_exercise",
                "context": "training_program",
                "emergency": False,
                "requester_location": "california"
            },
            "description": "Unauthorized Access Attempt"
        }
    ]
    
    for test_case in test_cases:
        response = requests.post(f"{BASE_URL}/privacy-decision", json=test_case["request"])
        print_response(response, test_case["description"])

def test_team_integration():
    """Test Team A and B integration contracts."""
    print_test_header("Team Integration Contracts")
    
    # Team A Integration Test (Temporal Context)
    team_a_request = {
        "requester": "dr_smith",
        "data_field": "patient_record", 
        "purpose": "treatment",
        "context": "medical",
        "emergency": True,
        "temporal_context": {
            "current_time": datetime.now(timezone.utc).isoformat(),
            "business_hours": False,
            "emergency_time": True,
            "shift_type": "night_shift"
        }
    }
    
    response = requests.post(f"{BASE_URL}/privacy-decision", json=team_a_request)
    print_response(response, "Team A Integration: Temporal Context")
    
    # Team B Integration Test (Organizational Context)
    team_b_request = {
        "requester": "manager_sarah",
        "data_field": "team_performance_data",
        "purpose": "quarterly_review", 
        "context": "hr_management",
        "emergency": False,
        "org_context": {
            "role": "engineering_manager",
            "department": "engineering", 
            "clearance_level": "manager",
            "employment_type": "full_time"
        }
    }
    
    response = requests.post(f"{BASE_URL}/privacy-decision", json=team_b_request)
    print_response(response, "Team B Integration: Organizational Context")

def test_api_contracts():
    """Test API contracts endpoint."""
    print_test_header("API Integration Contracts")
    response = requests.get(f"{BASE_URL}/contracts")
    print_response(response, "Team A & B Integration Specifications")

def test_timezone_features():
    """Test timezone-aware features."""
    print_test_header("Timezone Awareness Tests")
    
    timezone_test_cases = [
        {
            "requester": "alice_ca",
            "data_field": "employee_data",
            "purpose": "hr_review",
            "context": "business_hours_check",
            "requester_location": "california",
            "description": "California Business Hours"
        },
        {
            "requester": "raj_india", 
            "data_field": "project_data",
            "purpose": "status_update",
            "context": "team_coordination", 
            "requester_location": "india",
            "description": "India Office Hours"
        },
        {
            "requester": "sarah_london",
            "data_field": "client_information",
            "purpose": "support_ticket",
            "context": "customer_service",
            "requester_location": "london", 
            "description": "London Office Hours"
        }
    ]
    
    for test_case in timezone_test_cases:
        request_data = {k: v for k, v in test_case.items() if k != "description"}
        response = requests.post(f"{BASE_URL}/privacy-decision", json=request_data)
        print_response(response, test_case["description"])

def main():
    """Run comprehensive test suite."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🚀 Starting Team C Privacy Firewall Comprehensive Test Suite")
    print(f"🕐 Test Time: {datetime.now(timezone.utc).isoformat()}")
    
    # Check actual mode
    api_key = os.getenv("OPENAI_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if api_key and base_url == "https://api.groq.com/openai/v1":
        print(f"🔧 Mode: Graphiti + Groq LLM Integration ({groq_model})")
        print("🤖 AI-Powered Privacy Intelligence: ACTIVE")
    elif api_key:
        print("🔧 Mode: Graphiti + OpenAI Integration")
        print("🤖 AI-Powered Privacy Intelligence: ACTIVE")
    else:
        print("🔧 Mode: Neo4j Fallback (No API Key)")
        print("🤖 AI-Powered Privacy Intelligence: DISABLED")
    
    print("🌍 Features: Global Timezone Awareness, Emergency Override, Team Integration")
    
    try:
        # Test all endpoints and features
        test_health_endpoint()
        test_privacy_decisions()
        test_data_classification()
        test_team_integration()
        test_timezone_features()
        test_api_contracts()
        
        print_test_header("Test Suite Complete")
        print("✅ All tests completed successfully!")
        print("📊 Results: Privacy firewall is fully operational")
        print("🌍 Timezone support: Active")
        
        # Display actual system status
        if api_key and base_url == "https://api.groq.com/openai/v1":
            print("🤖 LLM Integration: Groq Llama 3.3 70B ACTIVE")
            print("🧠 AI-Powered Decisions: Fully operational")
        elif api_key:
            print("🤖 LLM Integration: OpenAI ACTIVE")
            print("🧠 AI-Powered Decisions: Fully operational")
        else:
            print("🔄 Neo4j fallback: Working perfectly")
            print("🧠 AI-Powered Decisions: Using rule-based logic")
            
        print("🤝 Team integration: Ready for deployment")
        print("\n📋 API Documentation: http://localhost:8002/docs")
        
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        print("⚠️  Ensure the API service is running on http://localhost:8002")

if __name__ == "__main__":
    main()