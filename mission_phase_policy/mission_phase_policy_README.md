# Mission Phase Policy Example

This example demonstrates how to implement complex mission phase policies using Graphiti's knowledge graph capabilities, featuring role-based access control, phase-specific restrictions, and nested conditions.

## 🎯 What This Example Shows

- **Complex Policy Storage**: How to store multi-layered policies in a knowledge graph
- **Role-Based Access Control**: Different permissions for different user roles
- **Phase-Specific Restrictions**: Access rules that change based on mission phase
- **Override Mechanisms**: Emergency and role-based overrides
- **Nested Conditions**: Complex decision logic with multiple factors

## 📋 Policy Rules

The example implements these complex rules:

### 🎖️ User Roles
- **Commander**: Top clearance, can override most restrictions
- **Analyst**: High clearance, specialized access to detailed data
- **Operator**: Medium clearance, operational access
- **Observer**: Low clearance, limited access

### 🚀 Mission Phases
- **Pre-Deployment**: High security, restricted access
- **Active Mission**: Operational access for efficiency
- **Post Mission**: Analysis-focused access
- **Emergency**: Override mode for safety

### 🔐 Access Rules

#### Pre-Deployment Phase
- **MissionObjectives**: Only Commander can access
- **MissionData**: Commander and Analyst only
- **OperationalStatus**: All roles can access

#### Active Mission Phase
- **MissionObjectives**: Commander, Analyst, Operator can access
- **MissionData**: All roles can access
- **OperationalStatus**: All roles can access

#### Post Mission Phase
- **MissionData**: Only Analyst can access detailed data
- **MissionObjectives**: All roles can access
- **OperationalStatus**: All roles can access

#### Emergency Phase
- **All Information**: Accessible to all roles (safety override)

### 🛡️ Override Mechanisms
- **Commander Override**: Commanders can access MissionObjectives in any phase
- **Emergency Override**: EmergencyProtocols always accessible
- **Phase Override**: Emergency phase overrides normal restrictions

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
   python mission_phase_policy_example.py
   ```

## 🔧 Troubleshooting

### Encoding Issues
If you encounter encoding errors, set locale variables:
```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Neo4j Connection Issues
- Verify Neo4j is running on port 7687
- Check your password is correct
- Ensure you're in the correct conda environment

## 🔍 What the Example Does

1. **Initializes Graphiti** with OpenAI for LLM and embeddings
2. **Adds complex policies to the knowledge graph**:
   - Structured policies with entities and relationships
   - Text-based policies for natural language queries
   - Role definitions and permissions
   - Phase-specific access rules
3. **Tests various scenarios**:
   - Different user roles in different phases
   - Override mechanisms
   - Access restrictions and allowances

## 📊 Expected Output

The example will test scenarios like:
```
📋 Test: Commander accessing strategic objectives
   ✅ ALLOWED
   Reason: Commander can access MissionObjectives during Pre-Deployment
   Policy Applied: pre_deployment_commander_access

📋 Test: Analyst trying to access objectives
   ❌ DENIED
   Reason: Only Commander role can access MissionObjectives during Pre-Deployment
   Policy Applied: pre_deployment_restriction
   Restrictions: Commander role required

📋 Test: Observer accessing emergency protocols
   ✅ ALLOWED
   Reason: Emergency protocols are always accessible for safety
   Policy Applied: emergency_override
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Request  │───▶│  Policy Engine  │───▶│  Graphiti Graph │
│                 │    │                 │    │                 │
│ Role            │    │ Phase Check     │    │ Role Check      │
│ Phase           │    │ Override Logic  │    │ Query Type      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Policy Decision │
                       │                 │
                       │ ✅ ALLOWED      │
                       │ ❌ DENIED       │
                       │ Override Used   │
                       └─────────────────┘
```

## 🧠 Knowledge Graph Structure

The policies are stored as:
- **Entities**: `Commander`, `Analyst`, `Pre-Deployment`, `MissionObjectives`
- **Relationships**: `HAS_OVERRIDE_FOR`, `RESTRICTED_IN`, `CAN_ACCESS_DURING`
- **Attributes**: Clearance levels, security levels, access permissions

## 💡 Key Features

### 🔄 Dynamic Policy Resolution
The system queries the knowledge graph at runtime to determine access permissions based on:
- Current mission phase
- User role
- Information type being requested
- Override conditions

### 🛡️ Layered Security
- **Phase-based restrictions**: Different rules for different mission phases
- **Role-based permissions**: Access levels based on user role
- **Override mechanisms**: Emergency and role-based overrides
- **Nested conditions**: Complex decision logic

### 🧠 Intelligent Queries
- Natural language policy queries
- Semantic understanding of roles and phases
- Context-aware decision making

## 🔧 Customization

You can easily modify the example to:

- **Add new roles**: Extend the `UserRole` enum and policy logic
- **Add new phases**: Create new mission phases with specific rules
- **Add new information types**: Define new query types with access rules
- **Modify override logic**: Change when and how overrides apply
- **Add more complex conditions**: Include additional factors like location, time, etc.

## 🎭 Use Cases

This pattern can be applied to:
- **Military command systems** with role-based access
- **Corporate security systems** with department-based restrictions
- **Healthcare systems** with role and phase-based access
- **Financial trading systems** with time and role restrictions
- **Government systems** with clearance-based access control

## 🔗 Related Examples

- Check out the `time_of_day_policy_example.py` for simpler time-based policies
- Look at the `examples/` directory for more Graphiti use cases
- Explore the MCP server for AI assistant integration

## 🎯 Original Scenario Implementation

The example specifically implements the scenario from the prompt:
- **User123** with **Commander** role
- Accessing **MissionObjectives** during **Pre-Deployment** phase
- System finds **Commander HAS_OVERRIDE_FOR MissionObjectives** relationship
- **Result**: ✅ ALLOWED with override applied

This demonstrates how the knowledge graph can store complex relationships and the system can resolve nested conditions at runtime! 