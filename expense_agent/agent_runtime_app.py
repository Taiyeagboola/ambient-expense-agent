# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from vertexai.agent_engines.templates.adk import AdkApp

from expense_agent.agent import app as adk_app
from expense_agent.app_utils.telemetry import setup_telemetry
from expense_agent.app_utils.typing import Feedback

# Load environment variables from .env file at runtime
load_dotenv()


class AgentEngineApp(AdkApp):
    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        # Use standard Python logging for console logs
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        def mock_log_struct(info_dict, severity="INFO"):
            self.logger.info(f"[{severity}] Struct log: {info_dict}")

        self.logger.log_struct = mock_log_struct

        if not os.environ.get("GEMINI_API_KEY"):
            vertexai.init()
            setup_telemetry()
            super().set_up()
            if gemini_location:
                os.environ["GOOGLE_CLOUD_LOCATION"] = gemini_location
        else:
            setup_telemetry()
            super().set_up()
            # AdkApp.set_up overrides this to "1", set it back to "False"
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = [*operations.get("", []), "register_feedback"]
        return operations

    def clone(self) -> "AgentEngineApp":
        """Returns a clone of the Agent Runtime application."""
        return self


gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
agent_runtime = AgentEngineApp(
    app=adk_app,
    artifact_service_builder=lambda: (
        GcsArtifactService(bucket_name=logs_bucket_name)
        if logs_bucket_name
        else InMemoryArtifactService()
    ),
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Force local telemetry off (otel_to_cloud=False behavior)
    os.environ["GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"] = "false"
    os.environ["OTEL_TO_CLOUD"] = "False"
    
    self_logger = logging.getLogger("ambient_expense_agent")
    self_logger.info("Initializing Agent Runtime...")
    agent_runtime.set_up()
    self_logger.info("Agent Runtime initialized.")
    yield

# FastAPI application for ambient trigger endpoint
app = FastAPI(title="Ambient Expense Agent Web Service", lifespan=lifespan)

@app.post("/")
@app.post("/pubsub")
@app.post("/apps/{app_name}/trigger/pubsub")
async def handle_pubsub(request: Request, app_name: str | None = None):
    self_logger = logging.getLogger("ambient_expense_agent")
    try:
        payload = await request.json()
    except Exception as e:
        self_logger.error(f"Failed to parse request JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = payload.get("message")
    subscription = payload.get("subscription")

    if not message or not subscription:
        self_logger.error("Missing 'message' or 'subscription' in Pub/Sub payload")
        raise HTTPException(status_code=400, detail="Missing 'message' or 'subscription'")

    # Normalize fully-qualified subscription path to short name
    # e.g., "projects/my-project/subscriptions/my-subscription" -> "my-subscription"
    short_name = subscription.split("/")[-1] if subscription else "default-subscription"
    self_logger.info(f"Received Pub/Sub message. Subscription normalized to: {short_name}")

    # Extract message ID and base64 encoded data
    message_id = message.get("messageId", "session-id")
    data = message.get("data", "")

    # Construct the payload message text for the agent workflow
    message_text = json.dumps({"data": data})

    self_logger.info(f"Feeding message into workflow for session {message_id} (user: {short_name})...")
    
    # Ensure session exists
    try:
        await agent_runtime.async_get_session(user_id=short_name, session_id=message_id)
    except Exception:
        self_logger.info(f"Creating new session {message_id} for user {short_name}...")
        await agent_runtime.async_create_session(user_id=short_name, session_id=message_id)

    events = []
    try:
        async for event in agent_runtime.async_stream_query(
            message=message_text,
            user_id=short_name,
            session_id=message_id,
        ):
            events.append(event)
            self_logger.info(f"Workflow Event: {event}")
    except Exception as e:
        self_logger.error(f"Error processing workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", "session_id": message_id, "events_count": len(events)}
