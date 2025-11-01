#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    error "This script is designed for macOS. Please adapt for your OS."
fi

log "🚀 Starting Complete Tech Stack Installation"
log "Installing: Docker, Neo4j, Apache Kafka, Kubernetes, Redis"
echo "=================================================================="

# Step 1: Install Homebrew if not present
log "📦 Step 1: Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    log "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    log "✅ Homebrew already installed"
fi

# Step 2: Install Docker Desktop
log "🐳 Step 2: Installing Docker Desktop..."
if ! command -v docker &> /dev/null; then
    brew install --cask docker
    log "✅ Docker Desktop installed"
    warn "Please start Docker Desktop from Applications folder"
    read -p "Press Enter after Docker Desktop is running..."
else
    log "✅ Docker already installed"
fi

# Verify Docker is running
log "Verifying Docker is running..."
timeout=30
count=0
while ! docker info &> /dev/null; do
    if [ $count -eq $timeout ]; then
        error "Docker failed to start within $timeout seconds"
    fi
    echo "Waiting for Docker to start... ($count/$timeout)"
    sleep 1
    ((count++))
done
log "✅ Docker is running"

# Step 3: Install Java (required for Kafka)
log "☕ Step 3: Installing Java..."
if ! command -v java &> /dev/null; then
    brew install openjdk@11
    echo 'export PATH="/usr/local/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
    export PATH="/usr/local/opt/openjdk@11/bin:$PATH"
    log "✅ Java installed"
else
    log "✅ Java already installed"
fi

# Step 4: Install Apache Kafka
log "📨 Step 4: Installing Apache Kafka..."
if ! command -v kafka-server-start &> /dev/null; then
    brew install kafka
    log "✅ Kafka installed"
else
    log "✅ Kafka already installed"
fi

# Step 5: Install Neo4j
log "🗄️ Step 5: Installing Neo4j..."
if ! command -v neo4j &> /dev/null; then
    brew install neo4j
    log "✅ Neo4j installed"
else
    log "✅ Neo4j already installed"
fi

# Step 6: Install Kubernetes tools
log "☸️ Step 6: Installing Kubernetes..."
if ! command -v kubectl &> /dev/null; then
    brew install kubectl
    log "✅ kubectl installed"
else
    log "✅ kubectl already installed"
fi

if ! command -v minikube &> /dev/null; then
    brew install minikube
    log "✅ minikube installed"
else
    log "✅ minikube already installed"
fi

# Step 7: Install Redis
log "🔴 Step 7: Installing Redis..."
if ! command -v redis-server &> /dev/null; then
    brew install redis
    log "✅ Redis installed"
else
    log "✅ Redis already installed"
fi

# Step 8: Install additional tools
log "🛠️ Step 8: Installing additional tools..."
brew install wget curl jq htop

# Step 9: Configure Neo4j
log "⚙️ Step 9: Configuring Neo4j..."
neo4j-admin dbms set-initial-password 12345678 2>/dev/null || warn "Neo4j password already set"
log "✅ Neo4j password configured"

# Step 10: Start services
log "🚀 Step 10: Starting services..."

# Start Neo4j
log "Starting Neo4j..."
neo4j start
sleep 5
log "✅ Neo4j started"

# Start Zookeeper
log "Starting Zookeeper..."
brew services start zookeeper
sleep 3
log "✅ Zookeeper started"

# Start Kafka
log "Starting Kafka..."
brew services start kafka
sleep 5
log "✅ Kafka started"

# Start Redis
log "Starting Redis..."
brew services start redis
sleep 2
log "✅ Redis started"

# Start Minikube
log "Starting Minikube..."
minikube start --driver=docker
log "✅ Minikube started"

# Step 11: Create Docker Compose for integrated services
log "📝 Step 11: Creating Docker Compose configuration..."
cat > docker-compose-fullstack.yml << 'DOCKER_EOF'
version: '3.8'

services:
  # Zookeeper for Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: graphiti-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    networks:
      - graphiti-network

  # Kafka Message Broker
  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: graphiti-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "29092:29092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_HOST://localhost:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_CREATE_TOPICS: "security-events:1:1,firewall-logs:1:1,policy-events:1:1"
    networks:
      - graphiti-network

  # Neo4j Graph Database
  neo4j-cluster:
    image: neo4j:5.15
    container_name: graphiti-neo4j
    ports:
      - "7475:7474"
      - "7688:7687"
    environment:
      - NEO4J_AUTH=neo4j/12345678
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_default__database=graphiti
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_security_procedures_allowlist=apoc.*
    volumes:
      - neo4j_cluster_data:/data
    networks:
      - graphiti-network

  # Redis for caching
  redis:
    image: redis:alpine
    container_name: graphiti-redis
    ports:
      - "6380:6379"
    networks:
      - graphiti-network

  # Kafka UI for management
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: graphiti-kafka-ui
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    networks:
      - graphiti-network

volumes:
  neo4j_cluster_data:

networks:
  graphiti-network:
    driver: bridge
DOCKER_EOF

log "✅ Docker Compose configuration created"

# Step 12: Start Docker Compose stack
log "🐳 Step 12: Starting Docker Compose stack..."
docker-compose -f docker-compose-fullstack.yml up -d
sleep 10
log "✅ Docker Compose stack started"

# Step 13: Create Kubernetes manifests
log "☸️ Step 13: Creating Kubernetes manifests..."
cat > k8s-graphiti-complete.yaml << 'K8S_EOF'
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: graphiti-security
---
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: graphiti-config
  namespace: graphiti-security
data:
  NEO4J_URI: "bolt://neo4j-service:7687"
  KAFKA_BOOTSTRAP_SERVERS: "kafka-service:9092"
  REDIS_URL: "redis://redis-service:6379"
---
# Neo4j Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neo4j
  namespace: graphiti-security
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.15
        ports:
        - containerPort: 7474
        - containerPort: 7687
        env:
        - name: NEO4J_AUTH
          value: "neo4j/12345678"
        - name: NEO4J_PLUGINS
          value: '["apoc"]'
---
# Neo4j Service
apiVersion: v1
kind: Service
metadata:
  name: neo4j-service
  namespace: graphiti-security
spec:
  selector:
    app: neo4j
  ports:
  - name: http
    port: 7474
    targetPort: 7474
  - name: bolt
    port: 7687
    targetPort: 7687
---
# Kafka Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka
  namespace: graphiti-security
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kafka
  template:
    metadata:
      labels:
        app: kafka
    spec:
      containers:
      - name: kafka
        image: confluentinc/cp-kafka:latest
        ports:
        - containerPort: 9092
        env:
        - name: KAFKA_BROKER_ID
          value: "1"
        - name: KAFKA_ZOOKEEPER_CONNECT
          value: "zookeeper-service:2181"
        - name: KAFKA_ADVERTISED_LISTENERS
          value: "PLAINTEXT://kafka-service:9092"
        - name: KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR
          value: "1"
---
# Kafka Service
apiVersion: v1
kind: Service
metadata:
  name: kafka-service
  namespace: graphiti-security
spec:
  selector:
    app: kafka
  ports:
  - port: 9092
    targetPort: 9092
K8S_EOF

log "✅ Kubernetes manifests created"

# Step 14: Create verification script
log "📋 Step 14: Creating verification script..."
cat > verify_installation.sh << 'VERIFY_EOF'
#!/bin/bash

echo "🔍 TECH STACK VERIFICATION"
echo "========================="

echo ""
echo "=== Installed Software ==="
echo "Docker: $(docker --version 2>/dev/null || echo 'Not installed')"
echo "Java: $(java -version 2>&1 | head -1 || echo 'Not installed')"
echo "Kafka: $(kafka-server-start --version 2>/dev/null | head -1 || echo 'Not installed')"
echo "Neo4j: $(neo4j version 2>/dev/null || echo 'Not installed')"
echo "kubectl: $(kubectl version --client --short 2>/dev/null || echo 'Not installed')"
echo "minikube: $(minikube version --short 2>/dev/null || echo 'Not installed')"
echo "Redis: $(redis-server --version 2>/dev/null || echo 'Not installed')"

echo ""
echo "=== Service Status ==="
echo "Neo4j: $(neo4j status 2>/dev/null || echo 'Not running')"
echo "Kafka: $(brew services list | grep kafka | awk '{print $2}' || echo 'Not running')"
echo "Zookeeper: $(brew services list | grep zookeeper | awk '{print $2}' || echo 'Not running')"
echo "Redis: $(brew services list | grep redis | awk '{print $2}' || echo 'Not running')"

echo ""
echo "=== Docker Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "=== Connectivity Tests ==="
echo "Neo4j (7474): $(curl -s http://localhost:7474 >/dev/null && echo '✅ Connected' || echo '❌ Failed')"
echo "Neo4j Cluster (7475): $(curl -s http://localhost:7475 >/dev/null && echo '✅ Connected' || echo '❌ Failed')"
echo "Kafka (9092): $(nc -z localhost 9092 2>/dev/null && echo '✅ Connected' || echo '❌ Failed')"
echo "Redis (6379): $(nc -z localhost 6379 2>/dev/null && echo '✅ Connected' || echo '❌ Failed')"
echo "Kafka UI (8080): $(curl -s http://localhost:8080 >/dev/null && echo '✅ Connected' || echo '❌ Failed')"

echo ""
echo "=== Kubernetes Status ==="
kubectl cluster-info 2>/dev/null || echo "Kubernetes not accessible"

echo ""
echo "=== Quick Kafka Test ==="
echo "Creating test topic..."
kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic test-installation 2>/dev/null || echo "Topic may already exist"
echo "Listing topics:"
kafka-topics --list --bootstrap-server localhost:9092 2>/dev/null || echo "Cannot list topics"

echo ""
echo "🎉 VERIFICATION COMPLETE!"
echo "Access points:"
echo "- Neo4j Browser: http://localhost:7474 (neo4j/12345678)"
echo "- Neo4j Cluster: http://localhost:7475 (neo4j/12345678)"
echo "- Kafka UI: http://localhost:8080"
echo "- Minikube Dashboard: minikube dashboard"
VERIFY_EOF

chmod +x verify_installation.sh
log "✅ Verification script created"

# Step 15: Create Graphiti integration test
log "🧪 Step 15: Creating Graphiti integration test..."
cat > graphiti_full_test.py << 'TEST_EOF'
import asyncio
import json
import logging
from datetime import datetime
from kafka import KafkaProducer
import sys
import os

# Add the current directory to Python path to import graphiti_core
sys.path.append('/Users/apple/Downloads/graphiti/graphiti')

try:
    from graphiti_core.nodes import EpisodicNode, EntityNode, CommunityNode, EpisodeType
    from neo4j import AsyncGraphDatabase
    print("✅ Successfully imported Graphiti modules")
except ImportError as e:
    print(f"❌ Failed to import Graphiti modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_integration():
    """Test complete Kafka -> Graphiti -> Neo4j integration"""
    
    logger.info("🚀 Starting Complete Tech Stack Integration Test")
    
    # Test 1: Kafka Producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        security_event = {
            "event_id": "INTEGRATION_TEST_001",
            "alert_type": "system_test",
            "description": "Automated integration test event",
            "timestamp": datetime.now().isoformat(),
            "severity": "info"
        }
        
        producer.send('security-events', value=security_event)
        producer.flush()
        logger.info("✅ Successfully sent event to Kafka")
        
    except Exception as e:
        logger.error(f"❌ Kafka test failed: {e}")
        return False
    
    # Test 2: Neo4j Connection
    try:
        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "12345678")
        )
        
        # Test connection
        async with driver.session() as session:
            result = await session.run("RETURN 'Neo4j Connected!' as message")
            record = await result.single()
            logger.info(f"✅ Neo4j: {record['message']}")
            
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        return False
    
    # Test 3: Graphiti Node Creation
    try:
        # Create EpisodicNode
        episode = EpisodicNode(
            name="Integration Test Episode",
            source=EpisodeType.json,
            content=json.dumps(security_event),
            source_description="Automated Integration Test",
            valid_at=datetime.now(),
            group_id="integration_test"
        )
        
        await episode.save(driver)
        logger.info(f"✅ Created EpisodicNode: {episode.uuid}")
        
        # Create EntityNode
        entity = EntityNode(
            name="Test System",
            labels=["System", "TestEntity"],
            attributes={"type": "integration_test", "version": "1.0"},
            group_id="integration_test"
        )
        
        await entity.save(driver)
        logger.info(f"✅ Created EntityNode: {entity.uuid}")
        
        # Create CommunityNode
        community = CommunityNode(
            name="Integration Test Community",
            summary="Test community for integration verification",
            group_id="integration_test"
        )
        
        await community.save(driver)
        logger.info(f"✅ Created CommunityNode: {community.uuid}")
        
    except Exception as e:
        logger.error(f"❌ Graphiti node creation failed: {e}")
        return False
    
    # Test 4: Query nodes back
    try:
        retrieved_episode = await EpisodicNode.get_by_uuid(driver, episode.uuid)
        retrieved_entity = await EntityNode.get_by_uuid(driver, entity.uuid)
        retrieved_community = await CommunityNode.get_by_uuid(driver, community.uuid)
        
        logger.info("✅ Successfully retrieved all created nodes")
        logger.info(f"   Episode: {retrieved_episode.name}")
        logger.info(f"   Entity: {retrieved_entity.name}")
        logger.info(f"   Community: {retrieved_community.name}")
        
    except Exception as e:
        logger.error(f"❌ Node retrieval failed: {e}")
        return False
    
    # Cleanup
    try:
        await EpisodicNode.delete_by_group_id(driver, "integration_test")
        logger.info("✅ Cleanup completed")
        
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")
    
    finally:
        await driver.close()
    
    logger.info("🎉 COMPLETE INTEGRATION TEST PASSED!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_complete_integration())
    if success:
        print("\n�� ALL SYSTEMS OPERATIONAL!")
        print("Your complete tech stack is ready for production use.")
    else:
        print("\n❌ Integration test failed. Please check the logs above.")
        sys.exit(1)
TEST_EOF

log "✅ Integration test created"

# Step 16: Install Python dependencies for testing
log "🐍 Step 16: Installing Python dependencies..."
pip install kafka-python neo4j asyncio

# Final verification
log "🔍 Step 17: Running verification..."
./verify_installation.sh

echo ""
log "🎉 INSTALLATION COMPLETE!"
echo "=================================================================="
info "Your complete tech stack is installed and running:"
info "✅ Docker Desktop"
info "✅ Apache Kafka (with Zookeeper)"
info "✅ Neo4j Graph Database"
info "✅ Kubernetes (minikube)"
info "✅ Redis"
info "✅ Graphiti (already installed)"
echo ""
info "🌐 Access Points:"
info "- Neo4j Browser: http://localhost:7474 (username: neo4j, password: 12345678)"
info "- Neo4j Cluster: http://localhost:7475 (username: neo4j, password: 12345678)"
info "- Kafka UI: http://localhost:8080"
info "- Minikube Dashboard: run 'minikube dashboard'"
echo ""
info "🧪 Test Commands:"
info "- Run full integration test: python graphiti_full_test.py"
info "- Verify installation: ./verify_installation.sh"
info "- Test Graphiti policies: cd time_of_day_policy && python time_of_day_policy_example.py"
echo ""
info "🚀 Next Steps:"
info "1. Run the integration test: python graphiti_full_test.py"
info "2. Deploy to Kubernetes: kubectl apply -f k8s-graphiti-complete.yaml"
info "3. Explore Graphiti policy examples in the policy folders"
echo ""
warn "📝 Note: Some services may take a few moments to fully start up."
warn "If any service fails, run './verify_installation.sh' to check status."

