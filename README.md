# Graphiti

> **Build Real-Time Knowledge Graphs for AI Agents**

[![Lint](https://github.com/getzep/Graphiti/actions/workflows/lint.yml/badge.svg?style=flat)](https://github.com/getzep/Graphiti/actions/workflows/lint.yml)
[![Unit Tests](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/getzep/Graphiti/actions/workflows/unit_tests.yml)
[![GitHub Repo stars](https://img.shields.io/github/stars/getzep/graphiti)](https://github.com/getzep/graphiti)

Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents operating in dynamic environments. Unlike traditional RAG methods, Graphiti continuously integrates user interactions, structured and unstructured enterprise data, and external information into a coherent, queryable graph.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Neo4j Database** (running locally or cloud)
3. **API Keys** (OpenAI, Groq, or Anthropic)

### Installation

```bash
# Clone the repository
git clone https://github.com/getzep/graphiti.git
cd graphiti

# Install dependencies
pip install -e .

# Or with poetry
poetry install
```

### Environment Setup

```bash
# Required environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Choose one LLM provider
export OPENAI_API_KEY="your_openai_key"
# OR
export GROQ_API_KEY="your_groq_key"
# OR
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Start Neo4j

**Option 1: Docker (Recommended)**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Option 2: Neo4j Desktop**
- Download and install [Neo4j Desktop](https://neo4j.com/download/)
- Create a new database with password `password`
- Start the database

## 📚 Policy Examples

Graphiti includes comprehensive policy examples demonstrating complex access control systems:

### 1. Time-of-Day Policy
**Simple time-based access control** - Restricts non-critical queries to working hours while allowing critical queries anytime.

```bash
cd time_of_day_policy
python time_of_day_policy_example.py
```

**Features:**
- Time-based access restrictions
- Query type classification (critical, non-critical, admin, research, weekend)
- Configurable working hours
- JSON-based configuration

### 2. Mission Phase Policy
**Complex role-based access control** with phase-specific restrictions for mission operations.

```bash
cd mission_phase_policy
python mission_phase_policy_example.py
```

**Features:**
- Role-based access control (RBAC)
- Mission phase-specific policies
- Commander override permissions
- Emergency protocol access

### 3. Prompt Sensitivity Policy
**Content filtering and redaction system** that automatically detects and redacts sensitive information.

```bash
cd prompt_sensitivity_policy
python prompt_sensitivity_example.py
```

**Features:**
- Firewall detection of sensitive terms
- Phase-specific content redaction
- Dynamic redaction based on mission phase
- Emergency override for safety

### 4. Before/After Event Policy
**Time-based blocking system** that restricts access to content based on scheduled events.

```bash
cd before_after_event_policy
python before_after_event_example.py
```

**Features:**
- Scheduled event blocking based on timestamps
- Before/after event access control
- Multiple concurrent scheduled events
- Timezone-aware timestamp comparisons

### 5. Multi-Factor Contradiction Policy
**Complex access control system** that handles multiple conflicting rules and contradiction resolution.

```bash
cd multi_factor_contradiction_policy
python multi_factor_contradiction_example.py
```

**Features:**
- Multiple conflicting rules and restrictions
- Contradiction resolution based on priority hierarchy
- User clearance levels and permissions
- Commander override and emergency access

### 6. Timezone-Aware Policy
**Sophisticated timezone-aware access control** that handles user identification and location-based timezone detection.

```bash
cd timezone_aware_policy
python timezone_aware_example.py
```

**Features:**
- User identification with email and name
- Location-based timezone detection
- Dynamic policy application based on user's local time
- Comprehensive location-to-timezone mapping

## 🛠️ Basic Usage

### Simple Example

```python
import asyncio
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.nodes import EpisodeType

async def main():
    # Initialize LLM client
    llm_config = LLMConfig(model="gpt-4o-mini")
    llm_client = OpenAIClient(llm_config)
    
    # Initialize Graphiti
    graphiti = Graphiti(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        llm_client=llm_client
    )
    
    # Build indices
    await graphiti.build_indices_and_constraints()
    
    # Add an episode
    await graphiti.add_episode(
        name="User Query",
        episode_body="What is the weather like today?",
        source=EpisodeType.text,
        source_description="User input",
        reference_time=datetime.now(timezone.utc)
    )
    
    # Query the graph
    results = await graphiti.query("weather information")
    print(results)
    
    # Close connection
    await graphiti.close()

# Run the example
asyncio.run(main())
```

## 🔧 Configuration

All policy examples use JSON configuration files that allow you to easily modify:
- Policy rules and logic
- Test scenarios
- Access permissions
- Working hours and phases
- User roles and query types

Example configuration structure:
```json
{
  "query_types": {
    "critical": {"allowed_hours": "always"},
    "non_critical": {"allowed_hours": "09:00-17:00"},
    "admin": {"allowed_hours": "08:00-20:00"}
  },
  "test_scenarios": [
    {
      "name": "Emergency system status check",
      "query": "Check system status immediately",
      "query_type": "critical",
      "expected_result": "ALLOWED"
    }
  ]
}
```

## 🏗️ Architecture

Graphiti uses a knowledge graph approach with:

- **Nodes**: Represent entities (users, resources, policies)
- **Edges**: Represent relationships and temporal connections
- **Episodes**: Capture events and interactions over time
- **Embeddings**: Enable semantic search and similarity matching

## 📖 Documentation

- **API Reference**: See `graphiti_core/` for detailed API documentation
- **Examples**: Check the `examples/` directory for more usage examples
- **Tests**: Run `pytest` to see test examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/getzep/graphiti/issues)
- **Discord**: [Join our Discord](https://discord.com/invite/W8Kw6bsgXQ)
- **Documentation**: [Full Documentation](https://docs.getzep.com)

---

