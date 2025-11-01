# API Setup Guide for Graphiti

This guide will help you set up all the necessary API keys and environment variables for the Graphiti project.

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Run the setup script:**
   ```bash
   ./setup_zshrc.sh
   ```

2. **Follow the prompts** to enter your API keys

3. **Reload your shell:**
   ```bash
   source ~/.zshrc
   ```

4. **Test your setup:**
   ```bash
   check-keys
   ```

### Option 2: Manual Setup

1. **Copy the example file:**
   ```bash
   cp .zshrc_example ~/.zshrc
   ```

2. **Edit the file** and replace placeholder values with your actual API keys

3. **Reload your shell:**
   ```bash
   source ~/.zshrc
   ```

## 🔑 Required API Keys

### 1. Groq API Key (Required)
- **Purpose:** LLM client for the time-of-day policy example
- **Get it:** [https://console.groq.com/](https://console.groq.com/)
- **Environment variable:** `GROQ_API_KEY`

### 2. Neo4j Password (Required)
- **Purpose:** Database connection for storing knowledge graph
- **Get it:** Set when installing Neo4j
- **Environment variable:** `NEO4J_PASSWORD`

## 🔑 Optional API Keys

### 3. OpenAI API Key
- **Purpose:** Alternative LLM client
- **Get it:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Environment variable:** `OPENAI_API_KEY`

### 4. Anthropic API Key (Claude)
- **Purpose:** Claude LLM client
- **Get it:** [https://console.anthropic.com/](https://console.anthropic.com/)
- **Environment variable:** `ANTHROPIC_API_KEY`

### 5. Voyage AI API Key
- **Purpose:** Embeddings for semantic search
- **Get it:** [https://www.voyageai.com/](https://www.voyageai.com/)
- **Environment variable:** `VOYAGE_API_KEY`

### 6. Google API Key (Gemini)
- **Purpose:** Google Gemini LLM client
- **Get it:** [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- **Environment variable:** `GOOGLE_API_KEY`

## 🗄️ Database Configuration

### Neo4j Setup

#### Option A: Docker (Recommended)
```bash
# Start Neo4j
start-db

# Stop Neo4j
stop-db
```

#### Option B: Local Installation
1. Download from [https://neo4j.com/download/](https://neo4j.com/download/)
2. Install and set password
3. Update environment variables:
   ```bash
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="your_password"
   ```

## 🛠️ Useful Commands

After setup, you'll have these helpful commands:

```bash
# Check if all API keys are set
check-keys

# Run the time-of-day policy example
graphiti-policy

# Start Neo4j database
start-db

# Stop Neo4j database
stop-db
```

## 🔒 Security Best Practices

1. **File Permissions:**
   ```bash
   chmod 600 ~/.zshrc
   ```

2. **Don't commit to version control:**
   - Add `~/.zshrc` to your `.gitignore`
   - Never share your API keys

3. **Use environment variables:**
   - Don't hardcode keys in your scripts
   - Use the provided setup methods

4. **Regular rotation:**
   - Rotate API keys periodically
   - Monitor API usage

## 🧪 Testing Your Setup

1. **Check API keys:**
   ```bash
   check-keys
   ```

2. **Run the policy example:**
   ```bash
   graphiti-policy
   ```

3. **Test database connection:**
   ```bash
   # Start Neo4j if not running
   start-db
   
   # Wait a moment, then run the example
   graphiti-policy
   ```

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY environment variable is required"**
   - Make sure you've set the environment variable
   - Run `source ~/.zshrc` to reload

2. **"Connection refused" for Neo4j**
   - Start Neo4j: `start-db`
   - Check if port 7687 is available

3. **Permission denied for .zshrc**
   - Fix permissions: `chmod 600 ~/.zshrc`

4. **API key not working**
   - Verify the key is correct
   - Check if you have sufficient credits/quota

### Debug Commands

```bash
# Check environment variables
env | grep -E "(GROQ|NEO4J|OPENAI|ANTHROPIC|VOYAGE|GOOGLE)"

# Check if Neo4j is running
docker ps | grep neo4j

# Test Neo4j connection
nc -zv localhost 7687
```

## 📚 Additional Resources

- [Graphiti Documentation](https://github.com/zep-ai/graphiti)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Groq API Documentation](https://console.groq.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🤝 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your API keys are correct
3. Ensure Neo4j is running
4. Check the log files for detailed error messages
5. Open an issue on the Graphiti GitHub repository 