 Graphiti Policy Examples

This directory contains comprehensive policy examples demonstrating how to use Graphiti for implementing complex access control policies.

## Policy Examples

### 1. Time-of-Day Policy (`time_of_day_policy/`)
A simple time-based access control system that restricts non-critical queries to working hours while allowing critical queries anytime.

**Key Features:**
- Time-based access restrictions
- Query type classification (critical, non-critical, admin, research, weekend)
- Configurable working hours
- JSON-based configuration
- Clean Graphiti representation

**Files:**
- `time_of_day_policy_example.py` - Main implementation
- `policy_config.json` - Configuration file
- `time_of_day_policy_README.md` - Detailed documentation
- `README.md` - Quick start guide

### 2. Mission Phase Policy (`mission_phase_policy/`)
A complex role-based access control system with phase-specific restrictions for mission operations.

**Key Features:**
- Role-based access control (RBAC)
- Mission phase-specific policies
- Commander override permissions
- Emergency protocol access
- JSON-based configuration
- Clean Graphiti representation

**Files:**
- `mission_phase_policy_example.py` - Main implementation
- `mission_phase_policy_config.json` - Configuration file
- `mission_phase_policy_README.md` - Detailed documentation
- `README.md` - Quick start guide

### 3. Prompt Sensitivity / Output Policy (`prompt_sensitivity_policy/`)
A content filtering and redaction system that automatically detects and redacts sensitive information based on mission phases.

**Key Features:**
- Firewall detection of sensitive terms
- Phase-specific content redaction
- Dynamic redaction based on mission phase
- Code name, location, and personnel protection
- Emergency override for safety
- JSON-based configuration
- Clean Graphiti representation

**Files:**
- `prompt_sensitivity_example.py` - Main implementation
- `prompt_sensitivity_config.json` - Configuration file
- `README.md` - Quick start guide

### 4. Before/After Event Policy (`before_after_event_policy/`)
A time-based blocking system that restricts access to content based on scheduled events and timestamps.

**Key Features:**
- Scheduled event blocking based on timestamps
- Before/after event access control
- Dynamic blocking based on current time
- Multiple concurrent scheduled events
- Timezone-aware timestamp comparisons
- JSON-based configuration
- Clean Graphiti representation

**Files:**
- `before_after_event_example.py` - Main implementation
- `before_after_event_config.json` - Configuration file
- `README.md` - Quick start guide

### 5. Multi-Factor Contradiction Policy (`multi_factor_contradiction_policy/`)
A complex access control system that handles multiple conflicting rules and contradiction resolution.

**Key Features:**
- Multiple conflicting rules and restrictions
- Contradiction resolution based on priority hierarchy
- User clearance levels and permissions
- Time-based and mission phase restrictions
- Commander override and emergency access
- Detailed rationale for all decisions
- JSON-based configuration
- Clean Graphiti representation

**Files:**
- `multi_factor_contradiction_example.py` - Main implementation
- `multi_factor_contradiction_config.json` - Configuration file
- `README.md` - Quick start guide

### 6. Timezone-Aware Policy (`timezone_aware_policy/`)
A sophisticated timezone-aware access control system that handles user identification, location-based timezone detection, and dynamic policy application.

**Key Features:**
- User identification with email and name
- Location-based timezone detection and fallback
- Dynamic policy application based on user's local time
- Working hours defined per user's timezone
- Emergency phase overrides all timezone restrictions
- Comprehensive location-to-timezone mapping
- JSON-based configuration
- Clean Graphiti representation

**Files:**
- `timezone_aware_example.py` - Main implementation
- `timezone_aware_config.json` - Configuration file
- `README.md` - Quick start guide

## Prerequisites

1. **Environment Setup:**
   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="password"
   ```

2. **Neo4j Database:**
   - Install and start Neo4j
   - Or use Docker: `docker-compose up -d neo4j`

3. **Python Dependencies:**
   - Install Graphiti: `pip install -e .`
   - Required packages are in `pyproject.toml`

## Quick Start

### Time-of-Day Policy
```bash
cd time_of_day_policy
python time_of_day_policy_example.py
```

### Mission Phase Policy
```bash
cd mission_phase_policy
python mission_phase_policy_example.py
```

### Prompt Sensitivity Policy
```bash
cd prompt_sensitivity_policy
python prompt_sensitivity_example.py
```

### Before/After Event Policy
```bash
cd before_after_event_policy
python before_after_event_example.py
```

### Multi-Factor Contradiction Policy
```bash
cd multi_factor_contradiction_policy
python multi_factor_contradiction_example.py
```

### Timezone-Aware Policy
```bash
cd timezone_aware_policy
python timezone_aware_example.py
```

## Configuration

Both examples use JSON configuration files that allow you to easily modify:
- Policy rules and logic
- Test scenarios
- Access permissions
- Working hours and phases
- User roles and query types

## Common Features

All policy examples demonstrate:
- **Graphiti Integration**: Using knowledge graphs for policy storage and retrieval
- **JSON Configuration**: Easy-to-modify policy definitions
- **Clean Output**: Graphiti representation showing relevant relationships
- **Comprehensive Testing**: Multiple test scenarios with detailed results
- **Logging**: Detailed execution logs for debugging
- **Error Handling**: Robust error handling and validation

## Extending the Examples

These examples serve as templates for building your own policy systems:
1. Copy the folder structure
2. Modify the JSON configuration
3. Update the policy logic in the Python files
4. Add your own test scenarios
5. Customize the Graphiti representation

## Troubleshooting

- **Neo4j Connection Issues**: Check if Neo4j is running and credentials are correct
- **API Key Issues**: Ensure GROQ_API_KEY is set correctly
- **Configuration Errors**: Validate JSON syntax in config files
- **Log Files**: Check the generated log files for detailed error information 