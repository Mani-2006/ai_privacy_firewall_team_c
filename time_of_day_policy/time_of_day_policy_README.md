# Time-of-Day Policy Example

This example demonstrates how to implement a simple time-of-day policy using Graphiti's knowledge graph capabilities.

## 🎯 What This Example Shows

- **Policy Storage**: How to store time-based policies in a knowledge graph
- **Runtime Enforcement**: How to query the graph at runtime to check if requests should be allowed
- **Flexible Rules**: How to handle different query types (critical vs non-critical)

## 📋 Policy Rules

The example implements these rules:
- **Critical queries**: Always allowed (emergency override)
- **Non-critical queries**: Only allowed during working hours (9AM–6PM)
- **Working hours**: 09:00–18:00

## 🚀 How to Run

1. **Make sure Neo4j is running** and accessible
2. **Set your environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="your-password"
   ```

3. **Run the example**:
   ```bash
   python time_of_day_policy_example.py
   ```

## 🔧 Troubleshooting

### Encoding Issues (Smart Quotes)

If you encounter this error:
```
'ascii' codec can't encode character '\u201d' in position 7: ordinal not in range(128)
```

**Solution**: Set the locale environment variables before running your script:
```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

**Complete working setup**:
```bash
# Activate your conda environment
conda activate graphiti

# Set environment variables
export NEO4J_PASSWORD="your-actual-password"
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Run your scripts
python time_of_day_policy_example.py
```

### Neo4j Authentication Issues

If you get authentication errors:
1. **Verify your Neo4j password** is correct
2. **Wait 15-30 minutes** if you hit authentication rate limits
3. **Reset your password** in Neo4j Desktop if needed

### Common Issues

- **Wrong conda environment**: Make sure you're in the `graphiti` environment, not `base`
- **Missing environment variables**: Ensure `OPENAI_API_KEY` and `NEO4J_PASSWORD` are set
- **Neo4j not running**: Start Neo4j Desktop or Docker container
- **Port conflicts**: Ensure Neo4j is accessible on `bolt://localhost:7687`

## 🔍 What the Example Does

1. **Initializes Graphiti** with OpenAI for LLM and embeddings
2. **Adds policies to the knowledge graph**:
   - Structured policy with entities and relationships
   - Text-based policies for context
3. **Tests various scenarios**:
   - Critical queries at different times
   - Non-critical queries during/outside working hours
   - Current time checks

## 📊 Expected Output

The example will test scenarios like:
```
📋 Test: Emergency system status check
   ✅ ALLOWED
   Reason: Critical queries are always allowed
   Time: 10:00

📋 Test: Summarize today's news
   ✅ ALLOWED
   Reason: Non-critical queries allowed during 09:00-18:00
   Time: 10:00
   Working Hours: 09:00-18:00

📋 Test: Summarize today's news
   ❌ DENIED
   Reason: Non-critical queries allowed during 09:00-18:00
   Time: 08:00
   Working Hours: 09:00-18:00
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Request  │───▶│  Policy Checker │───▶│  Graphiti Graph │
│                 │    │                 │    │                 │
│ Query Type      │    │ Time Validation │    │ Policy Storage  │
│ Current Time    │    │ Decision Logic  │    │ Working Hours   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Policy Decision │
                       │                 │
                       │ ✅ ALLOWED      │
                       │ ❌ DENIED       │
                       └─────────────────┘
```

## 🔧 Customization

You can easily modify the example to:

- **Change working hours**: Modify the `work_hours` in the policy
- **Add more query types**: Extend the `check_policy` method
- **Add more complex rules**: Include day-of-week restrictions, holidays, etc.
- **Integrate with your system**: Use the `TimeOfDayPolicy` class in your application

## 🧠 Knowledge Graph Structure

The policies are stored as:
- **Entities**: `NonCriticalQuery`, `WorkHours`, `QueryType`
- **Relationships**: `ALLOWED_DURING`, `HAS_TYPE`
- **Attributes**: Time windows, criticality levels, enforcement rules

## 💡 Use Cases

This pattern can be applied to:
- **API rate limiting** based on time
- **Content access controls** (e.g., adult content after hours)
- **System maintenance windows**
- **Compliance requirements** (e.g., trading hours)
- **Resource optimization** (e.g., heavy processing only during off-peak)

## 🔗 Related Examples

- Check out the `examples/` directory for more Graphiti use cases
- Look at the quickstart example for basic Graphiti setup
- Explore the MCP server for AI assistant integration 