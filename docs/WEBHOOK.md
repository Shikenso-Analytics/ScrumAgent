# ScrumAgent Webhook Server

This module provides a LangChain-powered webhook server for reacting to Taiga events with Discord integration.

## Features

- **LangChain Agent**: Uses a GPT-4o powered agent to intelligently respond to Taiga events
- **Ticket Assignment Detection**: When the bot is assigned to a ticket, the agent posts a personalized "Hello World" comment
- **Discord Notifications**: Sends notifications to Discord channels for:
  - Bot assignments
  - New ticket creation (user stories, tasks, issues, epics)
  - Status changes
- **Extensible Event Handler System**: Easily add new event handlers for different Taiga events

## Architecture

```
Taiga Webhook → FastAPI Server → Event Handlers
                                      ↓
                              ┌───────┴───────┐
                              ↓               ↓
                       LangChain Agent   Discord API
                              ↓               ↓
                        Taiga Comment    Channel Message
```

## Setup

### Environment Variables

Add these to your `.env` file:

```bash
# Required: Taiga credentials
TAIGA_API_URL="https://api.taiga.io"
TAIGA_URL="https://tree.taiga.io"
TAIGA_USERNAME="your-bot-username"
TAIGA_PASSWORD="your-bot-password"

# Optional: Use token instead of username/password
TAIGA_TOKEN=""

# Optional: Bot username for assignment detection (defaults to TAIGA_USERNAME)
TAIGA_BOT_USERNAME=""

# Optional: Secret for webhook verification
TAIGA_WEBHOOK_SECRET=""

# Required: Discord bot token for notifications
DISCORD_TOKEN="your-discord-bot-token"
DISCORD_GUILD_ID="your-guild-id"

# Required: OpenAI API key for LangChain agent
OPENAI_API_KEY="your-openai-api-key"

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT="ScrumAgent-Webhook"
```

### Configure Taiga-Discord Mappings

Edit `config/taiga_discord_maps.yaml` to map your Taiga projects to Discord channels:

```yaml
taiga_slag_to_discord_channel_map:
  my-project-slug: "1234567890123456789"  # Discord channel ID
  another-project: "9876543210987654321"

taiga_discord_user_map:
  taiga-user-id: "discord-username"
```

### Configure Taiga Webhook

1. Go to your Taiga project settings
2. Navigate to **Integrations** → **Webhooks**
3. Add a new webhook with:
   - **Name**: ScrumAgent Webhook
   - **URL**: `https://your-domain.com/webhook`
   - **Secret Key**: (optional, but recommended)

## Running Locally

### Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run the webhook server
python -m scrumagent.webhook_server
```

### Docker Compose

```bash
# Start the webhook server
docker-compose up -d

# View logs
docker-compose logs -f scrumagent-webhook
```

### Test the webhook

```bash
# Health check
curl http://localhost:8000/health

# Readiness check (verifies Taiga connection)
curl http://localhost:8000/ready

# Test webhook endpoint
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"action": "test", "type": "test", "by": {}, "date": "2024-01-01", "data": {}}'
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with nginx ingress controller
- kubectl configured to access your cluster

### Deployment Steps

1. **Create namespace**:
   ```bash
   kubectl create namespace scrumagent
   ```

2. **Create secrets** (copy and modify the example):
   ```bash
   cp k8s/secrets.yaml.example k8s/secrets.yaml
   # Edit k8s/secrets.yaml with your actual values
   kubectl apply -f k8s/secrets.yaml -n scrumagent
   ```

3. **Update ingress hostname**:
   Edit `k8s/ingress.yaml` and replace `scrumagent-webhook.example.com` with your actual domain.

4. **Deploy using kustomize**:
   ```bash
   # Update image registry in k8s/kustomization.yaml
   kubectl apply -k k8s/ -n scrumagent
   ```

   Or deploy individually:
   ```bash
   kubectl apply -f k8s/configmap.yaml -n scrumagent
   kubectl apply -f k8s/deployment.yaml -n scrumagent
   kubectl apply -f k8s/service.yaml -n scrumagent
   kubectl apply -f k8s/ingress.yaml -n scrumagent
   ```

5. **Build and push Docker image**:
   ```bash
   docker build -t your-registry/scrumagent-webhook:latest .
   docker push your-registry/scrumagent-webhook:latest
   ```

## Event Handlers

### Built-in Handlers

| Handler | Trigger | Actions |
|---------|---------|---------|
| `AssignmentHandler` | Bot assigned to ticket | LangChain agent adds comment + Discord notification |
| `TicketCreatedHandler` | New ticket created | Discord notification |
| `StatusChangeHandler` | Ticket status changed | Discord notification |

### Adding Custom Event Handlers

To add new event handlers, create a class that extends `EventHandler`:

```python
from scrumagent.webhook_server import EventHandler, EVENT_HANDLERS

class MyCustomHandler(EventHandler):
    def can_handle(self, payload, bot_info):
        # Return True if this handler should process the event
        return payload.action == "change" and payload.type == "task"
    
    def handle(self, payload, bot_info):
        # Process the event using LangChain agent or direct tools
        # Return a result dict
        return {"action": "custom_action", "result": "success"}

# Register the handler
EVENT_HANDLERS.append(MyCustomHandler())
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info with enabled features |
| `/health` | GET | Health check (for liveness probes) |
| `/ready` | GET | Readiness check (verifies Taiga/Discord connection) |
| `/webhook` | POST | Taiga webhook receiver |

## Monitoring

The `/ready` endpoint returns detailed status:

```json
{
  "status": "ready",
  "bot_user": "scrumagent-bot",
  "discord_enabled": true,
  "project_mappings": 3
}
```
