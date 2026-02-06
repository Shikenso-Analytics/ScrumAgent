"""
Taiga Webhook Server for ScrumAgent.

This server listens for Taiga webhook events and triggers LangChain agent actions.
Currently supports:
- Responding with "Hello World" when assigned to a ticket
- Sending Discord notifications for Taiga events
- Extensible event handlers for future automation
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from langchain_taiga.tools.taiga_tools import (
    add_comment_by_ref_tool,
    get_entity_by_ref_tool,
    get_taiga_api,
)

load_dotenv()

mod_path = Path(__file__).parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_USERNAME = os.getenv("TAIGA_BOT_USERNAME", os.getenv("TAIGA_USERNAME", ""))
WEBHOOK_SECRET = os.getenv("TAIGA_WEBHOOK_SECRET", "")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Load Discord mappings if available
try:
    with open(mod_path / "../config/taiga_discord_maps.yaml", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
        TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP = yaml_config.get("taiga_slag_to_discord_channel_map", {})
        TAIGA_USER_TO_DISCORD_USER_MAP = yaml_config.get("taiga_discord_user_map", {})
except FileNotFoundError:
    logger.warning("taiga_discord_maps.yaml not found, Discord integration disabled")
    TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP = {}
    TAIGA_USER_TO_DISCORD_USER_MAP = {}

# Initialize LLM for agent (optional - only if OpenAI key is available)
webhook_agent = None
if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(model_name="gpt-4o")
        webhook_agent = create_react_agent(
            llm,
            tools=[
                add_comment_by_ref_tool,
                get_entity_by_ref_tool,
            ],
            state_modifier=(
                "You are the ScrumAgent Webhook Handler. You respond to Taiga events automatically.\n\n"
                "## Your Capabilities\n"
                "1. add_comment_by_ref_tool - Add comments to Taiga entities\n"
                "2. get_entity_by_ref_tool - Get details about Taiga entities\n\n"
                "## Guidelines\n"
                "- When assigned to a ticket, introduce yourself with a friendly Hello World message\n"
                "- Keep comments concise and helpful\n"
                "- Include relevant context about the ticket\n"
            )
        )
        logger.info("LangChain agent initialized with OpenAI")
    except Exception as e:
        logger.warning("Failed to initialize LangChain agent: %s", e)
        webhook_agent = None
else:
    logger.info("OpenAI API key not set, using direct tool calls")

app = FastAPI(
    title="ScrumAgent Webhook Server",
    description="LangChain-powered webhook server that reacts to Taiga events",
    version="1.0.0",
)


class TaigaWebhookPayload(BaseModel):
    """Model for Taiga webhook payload."""

    action: str
    type: str
    by: Dict[str, Any]
    date: str
    data: Dict[str, Any]
    change: Optional[Dict[str, Any]] = None


# Discord Tools for the webhook agent
def send_discord_message(channel_id: str, message: str) -> str:
    """Send a message to a Discord channel."""
    if not DISCORD_TOKEN:
        return json.dumps({"error": "DISCORD_TOKEN not configured"})

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {DISCORD_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"content": message}

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        sent = response.json()
        return json.dumps({
            "success": True,
            "message_id": sent.get("id"),
            "content": sent.get("content", "")[:100]
        })
    except httpx.HTTPError as e:
        logger.error("Discord API error: %s", e)
        return json.dumps({"error": str(e)})


def get_discord_channel_for_project(project_slug: str) -> Optional[str]:
    """Get the Discord channel ID for a Taiga project."""
    return TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP.get(project_slug)


class EventHandler:
    """Base class for event handlers."""

    def can_handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> bool:
        """Check if this handler can process the event."""
        raise NotImplementedError

    def handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle the event and return a result."""
        raise NotImplementedError


class AssignmentHandler(EventHandler):
    """Handler for bot assignment events - uses LangChain agent."""

    def can_handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> bool:
        """Check if the bot was just assigned to this entity."""
        if payload.action != "change":
            return False

        change = payload.change
        if not change:
            return False

        diff = change.get("diff", {})
        bot_id = bot_info.get("id")
        bot_username = bot_info.get("username", "")
        bot_full_name = bot_info.get("full_name", "")

        # Check for assigned_to changes
        assigned_to_diff = diff.get("assigned_to")
        if assigned_to_diff:
            new_assigned = (
                assigned_to_diff.get("to")
                if isinstance(assigned_to_diff, dict)
                else None
            )
            if new_assigned in (bot_id, bot_username, bot_full_name):
                return True

        # Check for assigned_users changes
        assigned_users_diff = diff.get("assigned_users")
        if assigned_users_diff:
            if isinstance(assigned_users_diff, dict):
                new_assigned = assigned_users_diff.get("to", "")
                old_assigned = assigned_users_diff.get("from", "")
                if bot_full_name and bot_full_name in str(new_assigned):
                    if not old_assigned or bot_full_name not in str(old_assigned):
                        return True
                if bot_username and bot_username in str(new_assigned):
                    if not old_assigned or bot_username not in str(old_assigned):
                        return True
            elif isinstance(assigned_users_diff, list) and len(assigned_users_diff) >= 2:
                old_users = set(assigned_users_diff[0] or [])
                new_users = set(assigned_users_diff[1] or [])
                if bot_id in new_users and bot_id not in old_users:
                    return True

        return False

    def handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle bot assignment using LangChain agent or direct tool calls."""
        entity_info = extract_entity_info(payload)

        project_slug = entity_info.get("project_slug")
        entity_ref = entity_info.get("entity_ref")
        entity_type = entity_info.get("entity_type")
        subject = entity_info.get("subject", "")
        assigned_by = payload.by.get("full_name", payload.by.get("username", "someone"))

        if not all([project_slug, entity_ref, entity_type]):
            logger.error("Missing entity information: %s", entity_info)
            return {"error": "Missing entity information"}

        logger.info(
            "Bot assigned to %s #%s in project '%s' by %s",
            entity_type,
            entity_ref,
            project_slug,
            assigned_by,
        )

        # Try LangChain agent if available, otherwise use direct tool calls
        if webhook_agent:
            agent_prompt = (
                f"You have been assigned to a {entity_type} in Taiga.\n"
                f"Project: {project_slug}\n"
                f"Entity Reference: #{entity_ref}\n"
                f"Subject: {subject}\n"
                f"Assigned by: {assigned_by}\n\n"
                f"Please add a friendly 'Hello World' comment to this {entity_type} introducing yourself. "
                f"Mention that you're the ScrumAgent bot and you're ready to help. "
                f"Keep the comment concise and professional."
            )

            try:
                result = webhook_agent.invoke({
                    "messages": [HumanMessage(content=agent_prompt)]
                })
                agent_response = result["messages"][-1].content
                logger.info("Agent response: %s", agent_response[:200])

                # Also send Discord notification if channel mapping exists
                discord_result = self._notify_discord(
                    project_slug, entity_type, entity_ref, subject, assigned_by
                )

                return {
                    "action": "agent_handled",
                    "agent_response": agent_response,
                    "discord_notification": discord_result,
                }
            except Exception as e:
                logger.error("Agent failed: %s, falling back to direct comment", e)

        # Direct tool call (no agent / fallback)
        return self._fallback_comment(entity_info, assigned_by)

    def _notify_discord(
        self, project_slug: str, entity_type: str, entity_ref: int, 
        subject: str, assigned_by: str
    ) -> Optional[Dict[str, Any]]:
        """Send Discord notification about the assignment."""
        channel_id = get_discord_channel_for_project(project_slug)
        if not channel_id:
            return None

        message = (
            f"🤖 **ScrumAgent Assigned**\n"
            f"I've been assigned to {entity_type} **#{entity_ref}**: {subject}\n"
            f"Assigned by: {assigned_by}\n"
            f"I'll add a comment to the ticket shortly!"
        )

        result = send_discord_message(channel_id, message)
        return json.loads(result)

    def _fallback_comment(
        self, entity_info: Dict[str, Any], assigned_by: str
    ) -> Dict[str, Any]:
        """Fallback to direct comment if agent fails."""
        project_slug = entity_info.get("project_slug")
        entity_ref = entity_info.get("entity_ref")
        entity_type = entity_info.get("entity_type")
        subject = entity_info.get("subject", "")

        comment = (
            f"👋 Hello World! I'm the ScrumAgent Bot and I've been assigned to this {entity_type}.\n\n"
            f"**Subject:** {subject}\n"
            f"**Assigned by:** {assigned_by}\n\n"
            "I'm ready to help! Feel free to mention me for assistance."
        )

        try:
            result = add_comment_by_ref_tool.invoke({
                "project_slug": project_slug,
                "entity_ref": entity_ref,
                "entity_type": entity_type,
                "comment": comment,
            })
            return {
                "action": "fallback_commented",
                "result": json.loads(result) if isinstance(result, str) else result,
            }
        except Exception as e:
            logger.error("Fallback comment failed: %s", e)
            return {"error": str(e)}


class TicketCreatedHandler(EventHandler):
    """Handler for new ticket creation events."""

    def can_handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> bool:
        """Check if a new entity was created."""
        return payload.action == "create" and payload.type in (
            "userstory", "task", "issue", "epic"
        )

    def handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send Discord notification for new ticket."""
        entity_info = extract_entity_info(payload)
        project_slug = entity_info.get("project_slug")
        entity_type = entity_info.get("entity_type")
        entity_ref = entity_info.get("entity_ref")
        subject = entity_info.get("subject", "")
        created_by = payload.by.get("full_name", payload.by.get("username", "someone"))

        channel_id = get_discord_channel_for_project(project_slug)
        if not channel_id:
            return {"action": "no_discord_channel", "project": project_slug}

        message = (
            f"📝 **New {entity_type.title()} Created**\n"
            f"**#{entity_ref}**: {subject}\n"
            f"Created by: {created_by}"
        )

        result = send_discord_message(channel_id, message)
        return {
            "action": "discord_notified",
            "result": json.loads(result),
        }


class StatusChangeHandler(EventHandler):
    """Handler for status change events."""

    def can_handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> bool:
        """Check if status was changed."""
        if payload.action != "change":
            return False
        change = payload.change
        if not change:
            return False
        diff = change.get("diff", {})
        return "status" in diff

    def handle(
        self, payload: TaigaWebhookPayload, bot_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send Discord notification for status change."""
        entity_info = extract_entity_info(payload)
        project_slug = entity_info.get("project_slug")
        entity_type = entity_info.get("entity_type")
        entity_ref = entity_info.get("entity_ref")
        subject = entity_info.get("subject", "")
        changed_by = payload.by.get("full_name", payload.by.get("username", "someone"))

        diff = payload.change.get("diff", {})
        status_diff = diff.get("status", {})
        old_status = status_diff.get("from", "Unknown")
        new_status = status_diff.get("to", "Unknown")

        channel_id = get_discord_channel_for_project(project_slug)
        if not channel_id:
            return {"action": "no_discord_channel", "project": project_slug}

        message = (
            f"🔄 **Status Changed**\n"
            f"**{entity_type.title()} #{entity_ref}**: {subject}\n"
            f"`{old_status}` → `{new_status}`\n"
            f"Changed by: {changed_by}"
        )

        result = send_discord_message(channel_id, message)
        return {
            "action": "discord_notified",
            "result": json.loads(result),
        }


# Registry of event handlers
EVENT_HANDLERS: List[EventHandler] = [
    AssignmentHandler(),
    TicketCreatedHandler(),
    StatusChangeHandler(),
]


def get_bot_user_info() -> Optional[Dict[str, Any]]:
    """Get the bot's user info from Taiga API."""
    try:
        api = get_taiga_api()
        me = api.me()
        return {
            "id": me.id,
            "username": me.username,
            "full_name": me.full_name,
        }
    except (ValueError, ConnectionError, RuntimeError) as e:
        logger.error("Failed to get bot user info: %s", e)
        return None


def extract_entity_info(payload: TaigaWebhookPayload) -> Dict[str, Any]:
    """Extract entity information from webhook payload."""
    data = payload.data

    type_mapping = {
        "userstory": "userstory",
        "task": "task",
        "issue": "issue",
        "epic": "epic",
    }

    entity_type = type_mapping.get(payload.type)
    if not entity_type:
        return {}

    # Try multiple ways to get project slug
    project = data.get("project", {})
    if isinstance(project, dict):
        project_slug = project.get("slug", "")
    else:
        project_slug = ""
    
    # Fallback: try to get from project_extra_info
    if not project_slug:
        project_extra = data.get("project_extra_info", {})
        if isinstance(project_extra, dict):
            project_slug = project_extra.get("slug", "")
    
    # Fallback: try to extract slug from permalink
    if not project_slug and isinstance(project, dict):
        permalink = project.get("permalink", "")
        if "/project/" in permalink:
            project_slug = permalink.split("/project/")[-1].rstrip("/")
    
    # Fallback: construct from project name
    if not project_slug and isinstance(project, dict):
        project_name = project.get("name", "")
        if project_name:
            project_slug = project_name.lower().replace(" ", "-")
    
    entity_ref = data.get("ref")

    return {
        "entity_type": entity_type,
        "project_slug": project_slug,
        "entity_ref": entity_ref,
        "subject": data.get("subject", ""),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "status": "ok",
        "service": "ScrumAgent Webhook Server",
        "version": "1.0.0",
        "features": ["langchain-agent", "discord-integration", "taiga-webhooks"],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    try:
        bot_info = get_bot_user_info()
        discord_configured = bool(DISCORD_TOKEN)
        if bot_info:
            return {
                "status": "ready",
                "bot_user": bot_info.get("username"),
                "discord_enabled": discord_configured,
                "project_mappings": len(TAIGA_SLAG_TO_DISCORD_CHANNEL_MAP),
            }
        return {"status": "degraded", "message": "Could not fetch bot info"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle incoming Taiga webhook events.

    Taiga sends webhooks for various events:
    - create: New entity created
    - change: Entity modified
    - delete: Entity deleted
    - test: Webhook test event
    """
    try:
        body = await request.json()
        logger.info(
            "Received webhook: action=%s, type=%s",
            body.get("action"),
            body.get("type"),
        )

        # Handle test webhooks
        if body.get("action") == "test":
            logger.info("Received test webhook")
            return {"status": "ok", "message": "Test webhook received"}

        # Parse payload
        try:
            payload = TaigaWebhookPayload(**body)
        except (TypeError, ValueError) as e:
            logger.error("Failed to parse webhook payload: %s", e)
            raise HTTPException(
                status_code=400, detail=f"Invalid payload: {e}"
            ) from e

        # Get bot user info
        bot_info = get_bot_user_info()
        if not bot_info:
            logger.warning("Could not determine bot user info")
            return {"status": "ok", "message": "Bot user info not configured"}

        # Process through event handlers
        results = []
        for handler in EVENT_HANDLERS:
            if handler.can_handle(payload, bot_info):
                result = handler.handle(payload, bot_info)
                logger.info("Event handled by %s: %s", handler.__class__.__name__, result)
                results.append({
                    "handler": handler.__class__.__name__,
                    "result": result,
                })

        if results:
            return {"status": "ok", "handlers_triggered": results}

        return {"status": "ok", "message": "No action taken"}

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Webhook error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the webhook server."""
    import uvicorn

    logger.info("Starting ScrumAgent Webhook Server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
