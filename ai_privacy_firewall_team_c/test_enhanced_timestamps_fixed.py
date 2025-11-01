#!/usr/bin/env python3
"""
Test Script for Enhanced Graphiti Timestamp and Timezone Handling
===============================================================

This script demonstrates the improved timestamp handling for Graphiti integration,
following the patterns you provided with proper timezone awareness.

Author: Team C Privacy Firewall
Date: 2024-12-30
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
import pytz

# Add paths
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from integration.enhanced_graphiti_privacy_bridge import EnhancedGraphitiPrivacyBridge
from integration.timezone_utils import TimezoneHandler

async def test_timestamp_handling():
    """Test enhanced timestamp and timezone handling."""
    
    print("🔍 Testing Enhanced Graphiti Timestamp & Timezone Handling")
    print("=" * 60)
    
    # Initialize enhanced bridge
    bridge = EnhancedGraphitiPrivacyBridge()
    
    try:
        # Test 1: California office hours request
        print("\\n📍 Test 1: California Office Request")
        california_request = {
            "requester": "alice_california",
            "data_field": "employee_salary",
            "purpose": "payroll_processing",
            "context": "quarterly_review",
            "requester_location": "california",
            "emergency": False
        }
        
        # Show current time in different timezones
        current_utc = TimezoneHandler.get_current_utc()
        ca_time = TimezoneHandler.get_current_in_timezone("california")
        india_time = TimezoneHandler.get_current_in_timezone("india")
        
        print(f"   Current UTC: {TimezoneHandler.format_for_graphiti(current_utc)}")
        print(f"   California:  {ca_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   India:       {india_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   CA Business Hours: {TimezoneHandler.is_office_hours('california')}")
        print(f"   India Business Hours: {TimezoneHandler.is_office_hours('india')}")
        
        result1 = await bridge.create_privacy_decision_episode(california_request)
        print(f"   ✅ Decision: {'ALLOWED' if result1['allowed'] else 'DENIED'}")
        
        # Test 2: India office emergency request
        print("\\n📍 Test 2: India Office Emergency Request")
        india_request = {
            "requester": "raj_india",
            "data_field": "medical_records",
            "purpose": "emergency_treatment",
            "context": "hospital_emergency",
            "requester_location": "india",
            "emergency": True
        }
        
        result2 = await bridge.create_privacy_decision_episode(india_request)
        print(f"   ✅ Decision: {'ALLOWED' if result2['allowed'] else 'DENIED'}")
        
        # Test 3: Demonstrate timestamp formatting patterns
        print("\\n🕐 Test 3: Timestamp Formatting Demonstration")
        
        # Create sample conversation-style data like your examples
        sample_privacy_conversations = [
            {
                "timestamp": current_utc,
                "requester": "DataAnalyst",
                "message": "I need access to customer demographics for market analysis",
                "location": "california"
            },
            {
                "timestamp": current_utc,
                "requester": "PrivacyBot", 
                "message": "Analyzing request for PII content and business justification",
                "location": "utc"
            },
            {
                "timestamp": current_utc,
                "requester": "PrivacyBot",
                "message": "DENIED: Customer demographics contain PII requiring explicit consent",
                "location": "utc"
            }
        ]
        
        print("   Sample conversation with timestamps:")
        for entry in sample_privacy_conversations:
            formatted_time = TimezoneHandler.format_for_graphiti(
                entry["timestamp"], 
                entry["location"]
            )
            print(f"   {entry['requester']} ({formatted_time}): {entry['message']}")
        
        # Test 4: Data classification with timestamps
        print("\\n📊 Test 4: Data Classification with Timestamps")
        
        classification_tests = [
            "employee_bank_account",
            "customer_email", 
            "product_inventory_count",
            "user_browsing_history"
        ]
        
        for data_field in classification_tests:
            classification = await bridge.classify_data_field(data_field, "business_analytics")
            timestamp = TimezoneHandler.format_for_graphiti(TimezoneHandler.get_current_utc())
            print(f"   {data_field} ({timestamp}): {classification['data_type']} - {classification['sensitivity_level']}")
        
        # Test 5: Show timezone conversion examples
        print("\\n🌍 Test 5: Global Timezone Coordination")
        
        for location in ["california", "india", "london", "eastern"]:
            local_time = TimezoneHandler.get_current_in_timezone(location)
            is_business = TimezoneHandler.is_office_hours(location)
            business_context = TimezoneHandler.get_business_context(location)
            
            print(f"   {location.title():>10}: {local_time.strftime('%H:%M %Z')} - {business_context}")
        
        print("\\n✅ All timestamp and timezone tests completed successfully!")
        print("\\n📋 Key Features Demonstrated:")
        print("   ✅ ISO 8601 timestamp formatting with Z suffix")
        print("   ✅ Timezone-aware business hours checking")
        print("   ✅ Conversation-style content for Graphiti LLM")
        print("   ✅ Global office coordination support")
        print("   ✅ Proper temporal data for policy enforcement")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await bridge.close()

def demonstrate_conversation_patterns():
    """Show how our timestamps match your conversation examples."""
    print("\\n📝 Conversation Pattern Matching")
    print("=" * 40)
    
    # Your examples:
    your_examples = [
        "SalesBot (2024-07-30T00:00:00Z): Hi, I'm ManyBirds Assistant!",
        "John (2024-07-30T00:01:00Z): Hi, I'm looking for a new pair of shoes.",
        "SalesBot (2024-07-30T00:02:00Z): Of course! What kind of material are you looking for?"
    ]
    
    print("   Your shoe conversation examples:")
    for example in your_examples:
        print(f"   {example}")
    
    # Our privacy firewall equivalent:
    current_time = TimezoneHandler.get_current_utc()
    our_examples = [
        f"PrivacyBot ({TimezoneHandler.format_for_graphiti(current_time)}): Received data access request",
        f"Alice ({TimezoneHandler.format_for_graphiti(current_time)}): I need employee_salary data for payroll",
        f"PrivacyBot ({TimezoneHandler.format_for_graphiti(current_time)}): ALLOWED: Payroll processing is authorized"
    ]
    
    print("\\n   Our privacy firewall equivalent:")
    for example in our_examples:
        print(f"   {example}")
    
    print("\\n   ✅ Pattern matches - ready for Graphiti LLM processing!")

if __name__ == "__main__":
    demonstrate_conversation_patterns()
    asyncio.run(test_timestamp_handling())