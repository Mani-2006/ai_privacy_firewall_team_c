#!/usr/bin/env python3
"""
Groq LLM Client for Team C Privacy Firewall
==========================================

This module provides Groq API integration with Llama models for the privacy firewall.
Replaces OpenAI dependency with Groq's faster inference for privacy decision making.

Supported Models:
- llama3-70b-8192 (full Llama 3 70B)
- llama3-8b-8192 (Llama 3 8B)
- llama2-70b-4096 (Llama 2 70B)

Author: Team C Privacy Firewall
Date: 2024-10-02
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GroqConfig:
    """Configuration for Groq API."""
    api_key: str
    model: str = "llama3-70b-8192"  # Full Llama 3 70B (not instant)
    base_url: str = "https://api.groq.com/openai/v1"
    timeout: int = 30
    max_tokens: int = 1000
    temperature: float = 0.1  # Lower for more consistent privacy decisions

class GroqLLMClient:
    """
    Groq API client for privacy decision making with Llama models.
    
    Provides OpenAI-compatible interface for easy integration with Graphiti.
    """
    
    def __init__(self, config: Optional[GroqConfig] = None):
        """Initialize Groq client with configuration."""
        if config is None:
            config = GroqConfig(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama3-70b-8192")
            )
        
        if not config.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=config.timeout
        )
        
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion using Groq API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response dict compatible with OpenAI format
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }
        
        try:
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Groq API error: {e}")
    
    async def classify_data_field(self, data_field: str, context: str = None) -> Dict[str, Any]:
        """
        Classify a data field for privacy sensitivity using Llama.
        
        Args:
            data_field: The data field to classify
            context: Additional context about the data
            
        Returns:
            Classification result with data type and sensitivity
        """
        context_info = f" in context: {context}" if context else ""
        
        messages = [
            {
                "role": "system",
                "content": """You are a privacy classification expert. Classify data fields into:

DATA TYPES:
- FinancialData: Bank accounts, credit cards, salaries, financial statements
- MedicalData: Health records, patient data, medical history
- PersonalData: Names, emails, addresses, demographics
- TechnicalData: System logs, configurations, performance metrics

SENSITIVITY LEVELS:
- RestrictedData: Medical records, financial details (highest protection)
- ConfidentialData: Employee salaries, personal details (high protection)  
- InternalData: Customer demographics, business metrics (medium protection)
- PublicData: Marketing materials, public announcements (low protection)

Respond with JSON only: {"data_type": "type", "sensitivity": "level", "reasoning": ["reason1", "reason2"]}"""
            },
            {
                "role": "user", 
                "content": f"Classify this data field: '{data_field}'{context_info}"
            }
        ]
        
        try:
            response = await self.chat_completion(messages, temperature=0.1)
            content = response["choices"][0]["message"]["content"]
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "field": data_field,
                    "data_type": result.get("data_type", "PersonalData"),
                    "sensitivity": result.get("sensitivity", "InternalData"),
                    "reasoning": result.get("reasoning", []),
                    "context_dependent": False,
                    "equivalents": []
                }
            else:
                # Fallback if JSON parsing fails
                return self._fallback_classification(data_field)
                
        except Exception as e:
            print(f"⚠️  Groq classification error: {e}")
            return self._fallback_classification(data_field)
    
    def _fallback_classification(self, data_field: str) -> Dict[str, Any]:
        """Fallback classification using keyword matching."""
        field_lower = data_field.lower()
        
        # Financial data keywords
        if any(word in field_lower for word in ['salary', 'bank', 'credit', 'financial', 'payment', 'account']):
            return {
                "field": data_field,
                "data_type": "FinancialData", 
                "sensitivity": "ConfidentialData",
                "reasoning": ["Contains financial terminology"],
                "context_dependent": False,
                "equivalents": []
            }
        
        # Medical data keywords
        elif any(word in field_lower for word in ['patient', 'medical', 'health', 'diagnosis', 'treatment']):
            return {
                "field": data_field,
                "data_type": "MedicalData",
                "sensitivity": "RestrictedData", 
                "reasoning": ["Contains medical terminology"],
                "context_dependent": False,
                "equivalents": []
            }
        
        # Default to personal data
        else:
            return {
                "field": data_field,
                "data_type": "PersonalData",
                "sensitivity": "InternalData",
                "reasoning": [],
                "context_dependent": False,
                "equivalents": []
            }
    
    async def make_privacy_decision(self, requester: str, data_field: str, purpose: str, 
                                   context: str, emergency: bool = False) -> Dict[str, Any]:
        """
        Make a privacy decision using Llama reasoning.
        
        Args:
            requester: Who is requesting access
            data_field: What data field they want
            purpose: Why they need it
            context: Context of the request
            emergency: Whether this is an emergency
            
        Returns:
            Privacy decision with reasoning
        """
        emergency_note = " THIS IS AN EMERGENCY REQUEST." if emergency else ""
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a privacy decision engine. Make access control decisions based on:

PRIVACY PRINCIPLES:
1. Data minimization - only necessary access
2. Purpose limitation - access must match stated purpose  
3. Emergency override - medical emergencies get priority
4. Role-based access - consider requester authority

DECISION RULES:
- FinancialData: Requires HR/Finance authorization (DENY unless emergency)
- MedicalData: Emergency override allowed for medical staff
- PersonalData: Standard access policy applies  
- TechnicalData: Technical staff access granted

Respond with JSON: {{"allowed": true/false, "reason": "explanation", "confidence": 0.0-1.0, "emergency_used": true/false}}

{emergency_note}"""
            },
            {
                "role": "user",
                "content": f"Access request: {requester} wants '{data_field}' for '{purpose}' in {context} context. Emergency: {emergency}"
            }
        ]
        
        try:
            response = await self.chat_completion(messages, temperature=0.1)
            content = response["choices"][0]["message"]["content"]
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "allowed": result.get("allowed", False),
                    "reason": result.get("reason", "Access denied by policy"),
                    "confidence": result.get("confidence", 0.8),
                    "emergency_used": result.get("emergency_used", emergency)
                }
            else:
                return self._fallback_decision(data_field, emergency)
                
        except Exception as e:
            print(f"⚠️  Groq decision error: {e}")
            return self._fallback_decision(data_field, emergency)
    
    def _fallback_decision(self, data_field: str, emergency: bool) -> Dict[str, Any]:
        """Fallback decision logic."""
        field_lower = data_field.lower()
        
        # Emergency medical override
        if emergency and any(word in field_lower for word in ['patient', 'medical', 'health']):
            return {
                "allowed": True,
                "reason": "Emergency override for medical data",
                "confidence": 0.95,
                "emergency_used": True
            }
        
        # Financial data restriction
        elif any(word in field_lower for word in ['salary', 'bank', 'credit', 'financial']):
            return {
                "allowed": False,
                "reason": "Financial data requires HR authorization", 
                "confidence": 0.9,
                "emergency_used": False
            }
        
        # Default allow for other data
        else:
            return {
                "allowed": True,
                "reason": "Standard access policy applied",
                "confidence": 0.8,
                "emergency_used": False
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

# OpenAI-compatible wrapper for Graphiti integration
class GroqOpenAIAdapter:
    """
    Adapter to make Groq client compatible with OpenAI interface.
    
    This allows Groq to work with Graphiti without modifying Graphiti code.
    """
    
    def __init__(self, groq_client: GroqLLMClient):
        self.groq_client = groq_client
    
    async def chat_completions_create(self, messages: List[Dict[str, str]], **kwargs):
        """OpenAI-compatible chat completions method."""
        response = await self.groq_client.chat_completion(messages, **kwargs)
        
        # Convert to OpenAI-like response object
        class MockChoice:
            def __init__(self, content):
                self.message = type('Message', (), {'content': content})()
        
        class MockResponse:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        return MockResponse(response["choices"][0]["message"]["content"])

# Factory function for easy instantiation
async def create_groq_client() -> GroqLLMClient:
    """Create a configured Groq LLM client."""
    return GroqLLMClient()

# Async context manager for proper cleanup
class GroqClientManager:
    """Context manager for Groq client lifecycle."""
    
    def __init__(self, config: Optional[GroqConfig] = None):
        self.config = config
        self.client = None
    
    async def __aenter__(self) -> GroqLLMClient:
        self.client = GroqLLMClient(self.config)
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()