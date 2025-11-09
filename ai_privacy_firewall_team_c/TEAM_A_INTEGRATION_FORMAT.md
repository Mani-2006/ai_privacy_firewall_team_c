# Team C → Team A Integration Format

**Date**: November 5, 2025  
**From**: Team C (Ken Mani Binu & Nibin Thomas)  
**To**: Team A (Mehara Ayisha T & Aflah Muhammed)

## Purpose

This document shows the **exact format** Team C is using to send data to Team A's temporal framework for privacy decision evaluation.

---

## Data Structure: 6-Tuple Format

Team C is sending data using your **`EnhancedContextualIntegrityTuple`** class with proper **`TemporalContext`** and **`TimeWindow`** objects.

### Python Object Structure

```python
from core.tuples import EnhancedContextualIntegrityTuple, TemporalContext, TimeWindow
from datetime import datetime, timezone, timedelta

# Example: Emergency Doctor Accessing Patient Data
enhanced_tuple = EnhancedContextualIntegrityTuple(
    data_subject="patient_heart_rate",
    data_sender="system",
    data_recipient="emergency_doctor",
    data_type="HealthData.VitalSigns",
    transmission_principle="secure",
    temporal_context=TemporalContext(
        situation="EMERGENCY",
        emergency_override=True,
        temporal_role="oncall_critical",
        access_window=TimeWindow(
            start=datetime(2025, 11, 5, 14, 0, 0, tzinfo=timezone.utc),
            end=datetime(2025, 11, 5, 15, 0, 0, tzinfo=timezone.utc),
            window_type="emergency"
        ),
        timestamp=datetime(2025, 11, 5, 14, 0, 0, tzinfo=timezone.utc),
        timezone="UTC"
    )
)

# Call your policy engine
temporal_result = temporal_policy_engine.evaluate_temporal_access(enhanced_tuple)
```

---

## Field Mapping: Team C → Team A

### User Request (What Team C Receives)

```json
{
  "data_field": "patient_heart_rate",
  "requester_role": "emergency_doctor",
  "context": "cardiac_arrest_emergency",
  "temporal_context": {
    "situation": "EMERGENCY",
    "urgency_level": "critical",
    "data_type": "HealthData.VitalSigns",
    "transmission_principle": "secure",
    "emergency_override_requested": true,
    "time_window_start": "2025-11-05T14:00:00Z",
    "time_window_end": "2025-11-05T15:00:00Z"
  }
}
```

### Team C's Transformation to Your Format

| Team C Field | Team A Field | Mapping Logic |
|--------------|--------------|---------------|
| `data_field` | `data_subject` | Direct mapping |
| - | `data_sender` | Fixed: `"system"` |
| `requester_role` | `data_recipient` | Direct mapping |
| `temporal_context.data_type` | `data_type` | Direct mapping (default: `"sensitive"`) |
| `temporal_context.transmission_principle` | `transmission_principle` | Direct mapping (default: `"secure"`) |
| `temporal_context.situation` | `temporal_context.situation` | Direct mapping |
| `temporal_context.urgency_level` | `temporal_context.emergency_override` | `high/critical → True`, `low/normal → False` |
| `temporal_context.urgency_level` | `temporal_context.temporal_role` | See mapping table below |
| `temporal_context.time_window_start` | `temporal_context.access_window.start` | TimeWindow object |
| `temporal_context.time_window_end` | `temporal_context.access_window.end` | TimeWindow object |
| `temporal_context.urgency_level` | `temporal_context.access_window.window_type` | `critical → "emergency"`, else `"access_window"` |

### Urgency Level → Temporal Role Mapping

```python
temporal_role_mapping = {
    "low": "user",
    "normal": "user", 
    "high": "oncall_high",
    "critical": "oncall_critical"
}
```

---

## Complete Examples

### Example 1: Critical Emergency (ER Doctor)

**Team C Receives:**
```json
{
  "data_field": "patient_heart_rate",
  "requester_role": "emergency_doctor",
  "context": "cardiac_arrest_emergency",
  "temporal_context": {
    "situation": "EMERGENCY",
    "urgency_level": "critical",
    "data_type": "HealthData.VitalSigns",
    "transmission_principle": "secure",
    "emergency_override_requested": true
  }
}
```

**Team C Sends to Team A:**
```python
EnhancedContextualIntegrityTuple(
    data_subject="patient_heart_rate",
    data_sender="system",
    data_recipient="emergency_doctor",
    data_type="HealthData.VitalSigns",
    transmission_principle="secure",
    temporal_context=TemporalContext(
        situation="EMERGENCY",                    # ✅ Valid: NORMAL, EMERGENCY, MAINTENANCE, INCIDENT, AUDIT
        emergency_override=True,                  # ✅ critical urgency → True
        temporal_role="oncall_critical",          # ✅ Valid: user, oncall_critical, oncall_high, etc.
        access_window=TimeWindow(
            start=datetime(...),
            end=datetime(...),
            window_type="emergency"               # ✅ Valid: business_hours, emergency, access_window
        ),
        timestamp=datetime(..., tzinfo=timezone.utc),
        timezone="UTC"
    )
)
```

**Team A Returns:**
```python
{
    "decision": "ALLOW",
    "confidence": 0.5,
    "reasoning": "Temporal evaluation",
    "emergency_override": True,
    "urgency_level": "critical",
    "time_window_valid": True
}
```

---

### Example 2: Normal Business Hours (HR Manager)

**Team C Receives:**
```json
{
  "data_field": "employee_directory",
  "requester_role": "hr_manager",
  "context": "routine_hr_review",
  "temporal_context": {
    "situation": "NORMAL",
    "urgency_level": "normal",
    "data_type": "PublicData.Directory",
    "transmission_principle": "secure"
  }
}
```

**Team C Sends to Team A:**
```python
EnhancedContextualIntegrityTuple(
    data_subject="employee_directory",
    data_sender="system",
    data_recipient="hr_manager",
    data_type="PublicData.Directory",
    transmission_principle="secure",
    temporal_context=TemporalContext(
        situation="NORMAL",                       # ✅ Normal situation
        emergency_override=False,                 # ✅ normal urgency → False
        temporal_role="user",                     # ✅ normal urgency → user
        access_window=TimeWindow(
            start=datetime(...),
            end=datetime(...),
            window_type="access_window"           # ✅ Normal access window
        ),
        timestamp=datetime(..., tzinfo=timezone.utc),
        timezone="UTC"
    )
)
```

**Team A Returns:**
```python
{
    "decision": "DENY",
    "confidence": 0.5,
    "reasoning": "Temporal evaluation",
    "emergency_override": False,
    "urgency_level": "normal",
    "time_window_valid": True
}
```

---

### Example 3: Low Priority Contractor Access

**Team C Receives:**
```json
{
  "data_field": "quarterly_revenue_reports",
  "requester_role": "temporary_contractor",
  "context": "unauthorized_access_attempt",
  "temporal_context": {
    "situation": "NORMAL",
    "urgency_level": "low",
    "data_type": "FinancialData.Revenue",
    "transmission_principle": "insecure"
  }
}
```

**Team C Sends to Team A:**
```python
EnhancedContextualIntegrityTuple(
    data_subject="quarterly_revenue_reports",
    data_sender="system",
    data_recipient="temporary_contractor",
    data_type="FinancialData.Revenue",
    transmission_principle="insecure",            # ⚠️ Insecure transmission
    temporal_context=TemporalContext(
        situation="NORMAL",
        emergency_override=False,                 # ✅ low urgency → False
        temporal_role="user",                     # ✅ low urgency → user
        access_window=TimeWindow(
            start=datetime(...),
            end=datetime(...),
            window_type="access_window"
        ),
        timestamp=datetime(..., tzinfo=timezone.utc),
        timezone="UTC"
    )
)
```

**Team A Returns:**
```python
{
    "decision": "DENY",
    "confidence": 0.5,
    "reasoning": "Temporal evaluation",
    "emergency_override": False,
    "urgency_level": "low",
    "time_window_valid": True
}
```

---

## Validation Requirements

### Required Fields (All Must Be Present)

✅ **`data_subject`** (string) - The data field being accessed  
✅ **`data_sender`** (string) - Source of data (we use "system")  
✅ **`data_recipient`** (string) - Who wants to access (requester role)  
✅ **`data_type`** (string) - Classification of data  
✅ **`transmission_principle`** (string) - How data is transmitted  
✅ **`temporal_context`** (TemporalContext object)

### TemporalContext Required Fields

✅ **`situation`** - Must be one of: `NORMAL`, `EMERGENCY`, `MAINTENANCE`, `INCIDENT`, `AUDIT`  
✅ **`emergency_override`** (boolean) - Emergency flag  
✅ **`temporal_role`** - Must be one of: `user`, `admin`, `system`, `emergency_responder`, `auditor`, `oncall_low`, `oncall_medium`, `oncall_high`, `oncall_critical`  
✅ **`access_window`** (TimeWindow object) - Time boundaries  
✅ **`timestamp`** (datetime with timezone) - When request was made  
✅ **`timezone`** (string) - Timezone identifier

### TimeWindow Required Fields

✅ **`start`** (datetime) - Window start time  
✅ **`end`** (datetime) - Window end time  
✅ **`window_type`** - Must be one of: `business_hours`, `emergency`, `access_window`, `maintenance`, `holiday`

---

## Questions for Team A

### 1. **Is this format correct?**
   - Are we using the right field names?
   - Are we mapping urgency levels correctly?
   - Should we handle any additional fields?

### 2. **What should we expect back?**
   - Current format we're receiving:
     ```python
     {
         "decision": "ALLOW" or "DENY",
         "confidence": 0.5,
         "reasoning": "Temporal evaluation",
         "emergency_override": boolean,
         "urgency_level": string,
         "time_window_valid": boolean
     }
     ```
   - Is this the correct response format from your `evaluate_temporal_access()` method?

### 3. **Are there any validation errors we should handle?**
   - What happens if we send invalid `situation` values?
   - What if time windows are malformed?
   - Should we catch specific exceptions?

### 4. **Performance considerations?**
   - How long should we expect evaluation to take?
   - Should we implement timeout handling?
   - Can we batch multiple requests?

---

## Current Integration Status

✅ **Working**: Sending 6-tuple format with proper field mapping  
✅ **Working**: Urgency level → emergency_override + temporal_role mapping  
✅ **Working**: TimeWindow creation with proper timezone handling  
✅ **Working**: Receiving and processing your responses  

⏳ **Pending**: Validation from Team A that format is 100% correct  
⏳ **Pending**: Clarification on response format expectations  

---

## Contact

**Team C**: Ken Mani Binu & Nibin Thomas  
**Implementation File**: `enhanced_privacy_api_service.py` (lines 465-510)  
**Testing**: Demo script available showing 3 real-world scenarios

Please review and confirm if this integration format matches your expectations! 🙏
