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
import os
from collections.abc import AsyncGenerator
from typing import Any

import google.auth
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode, Workflow
from pydantic import BaseModel, Field, model_validator

from expense_agent.config import APPROVAL_THRESHOLD, MODEL_NAME

# Load environment variables from .env file
load_dotenv()

# Determine authentication mode and set configurations
if os.environ.get("GEMINI_API_KEY"):
    # Google AI Studio API Key authentication
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
    # Prevent Vertex AI SDK from crashing on default credentials checks
    try:
        google.auth.default()
    except Exception:
        from google.auth.credentials import AnonymousCredentials

        def mock_default(*args, **kwargs):
            return AnonymousCredentials(), "dummy-project"

        google.auth.default = mock_default
else:
    # Google Cloud / Vertex AI authentication
    try:
        _, project_id = google.auth.default()
        if project_id:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except Exception:
        from google.auth.credentials import AnonymousCredentials

        def mock_default(*args, **kwargs):
            return AnonymousCredentials(), "dummy-project"

        google.auth.default = mock_default
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy-project")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


class PubSubMessage(BaseModel):
    data: Any = Field(description="The expense details or Pub/Sub payload data key")

    @model_validator(mode="before")
    @classmethod
    def pre_validate(cls, data: Any) -> Any:
        # If it's a Pydantic model / Content object with parts
        if hasattr(data, "parts"):
            parts = data.parts
            if parts:
                first_part = parts[0]
                text_content = getattr(first_part, "text", "") or ""
                if text_content:
                    try:
                        parsed = json.loads(text_content)
                        if isinstance(parsed, dict) and "data" in parsed:
                            return {"data": parsed["data"]}
                        return {"data": parsed}
                    except Exception:
                        return {"data": text_content}

        if isinstance(data, dict):
            # Check if it is a dictionary representing a Content object
            if "parts" in data:
                parts = data["parts"]
                if parts and isinstance(parts, list):
                    first_part = parts[0]
                    text_content = ""
                    if isinstance(first_part, dict):
                        text_content = first_part.get("text", "")
                    elif hasattr(first_part, "text"):
                        text_content = first_part.text or ""

                    if text_content:
                        try:
                            parsed = json.loads(text_content)
                            if isinstance(parsed, dict) and "data" in parsed:
                                return {"data": parsed["data"]}
                            return {"data": parsed}
                        except Exception:
                            return {"data": text_content}
            # Standard dict representation
            if "data" in data:
                return data
            return {"data": data}

        return {"data": data}


class RiskAssessment(BaseModel):
    risk_score: int = Field(description="A risk rating between 1 (low) and 5 (high)")
    risk_factors: list[str] = Field(description="List of identified risk factors")
    explanation: str = Field(description="Detailed explanation of the risk rating")


def parse_expense(ctx: Context, node_input: PubSubMessage) -> Event:
    data_content = node_input.data
    expense_data = None
    if isinstance(data_content, str):
        # Try base64 decoding (Pub/Sub standard)
        try:
            decoded_bytes = base64.b64decode(data_content)
            decoded_str = decoded_bytes.decode("utf-8")
            expense_data = json.loads(decoded_str)
        except Exception:
            # Fallback to plain JSON string parsing
            try:
                expense_data = json.loads(data_content)
            except Exception as inner_err:
                raise ValueError(
                    f"Could not parse data string: {data_content}"
                ) from inner_err
    elif isinstance(data_content, dict):
        expense_data = data_content
    else:
        raise ValueError(f"Unsupported data type: {type(data_content)}")

    amount = float(expense_data.get("amount", 0))
    submitter = expense_data.get("submitter", "Unknown")
    category = expense_data.get("category", "General")
    description = expense_data.get("description", "")
    date = expense_data.get("date", "")

    expense_details = {
        "amount": amount,
        "submitter": submitter,
        "category": category,
        "description": description,
        "date": date,
    }

    if amount < APPROVAL_THRESHOLD:
        outcome = {
            "status": "approved",
            "auto_approved": True,
            "amount": amount,
            "submitter": submitter,
            "category": category,
            "description": description,
            "date": date,
        }
        return Event(output=outcome, actions=EventActions(route="auto_approve"))
    else:
        return Event(
            output=expense_details,
            actions=EventActions(
                route="needs_review",
                state_delta={"expense": expense_details},
            ),
        )


def security_checkpoint(ctx: Context, node_input: dict) -> Event:
    """Security Checkpoint Node.

    Scrubs SSNs and CCs from the description, and detects prompt injection.
    If prompt injection is detected, routes directly to human approval.
    Otherwise, routes to the LLM risk assessor with the scrubbed description.
    """
    from expense_agent.security import detect_prompt_injection, scrub_description

    description = node_input.get("description", "")
    already_redacted = ctx.state.get("redacted_categories") if ctx.state else None

    # 1. Scrub personal data
    scrubbed_desc, redacted_categories = scrub_description(
        description, already_redacted
    )

    # Update the expense details in node_input
    scrubbed_details = dict(node_input)
    scrubbed_details["description"] = scrubbed_desc

    # 2. Defend against prompt injection (on original description)
    injection_detected = detect_prompt_injection(description)

    # Prepare delta of what changed
    delta = {
        "expense": scrubbed_details,
    }
    if redacted_categories:
        delta["redacted_categories"] = redacted_categories

    if injection_detected:
        delta["security_event"] = True
        # Bypass LLM and route straight to human approval
        security_risk = RiskAssessment(
            risk_score=5,
            risk_factors=["Prompt Injection"],
            explanation="SUSPECTED PROMPT INJECTION: The description contains instructions that appear to attempt to bypass approval rules.",
        )
        return Event(
            output=security_risk,
            actions=EventActions(
                route="injection",
                state_delta=delta,
            ),
        )
    else:
        return Event(
            output=scrubbed_details,
            actions=EventActions(
                route="clean",
                state_delta=delta,
            ),
        )


# LLM Risk Assessor Agent
risk_assessor = LlmAgent(
    name="risk_assessor",
    model=MODEL_NAME,
    instruction=(
        "You are a risk assessment AI. Review the provided expense report details for suspicious "
        "activity, excessive pricing, policy violations, or other risk factors. "
        "Rate the risk from 1 (lowest risk) to 5 (highest risk), list the specific risk factors, "
        "and provide a detailed explanation of your judgment."
    ),
    output_schema=RiskAssessment,
)


# Human Approval Node
async def human_approval(
    ctx: Context, node_input: RiskAssessment
) -> AsyncGenerator[Event | RequestInput, None]:
    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        expense = ctx.state.get("expense", {})
        msg = (
            f"Review required: Submitter={expense.get('submitter')}, Amount=${expense.get('amount')}, "
            f"Category={expense.get('category')}, Risk Score={node_input.risk_score}. "
            f"Risk Explanation: {node_input.explanation}. Please reply with 'approve' or 'reject'."
        )
        yield RequestInput(interrupt_id="decision", message=msg)
        return

    decision_val = ctx.resume_inputs["decision"]
    if isinstance(decision_val, dict):
        decision_str = str(
            decision_val.get("decision")
            or decision_val.get("response")
            or next(iter(decision_val.values()))
        )
    else:
        decision_str = str(decision_val)
    status = "approved" if "approve" in decision_str.lower() else "rejected"

    expense = ctx.state.get("expense", {})
    outcome = {
        "status": status,
        "auto_approved": False,
        "amount": expense.get("amount"),
        "submitter": expense.get("submitter"),
        "category": expense.get("category"),
        "description": expense.get("description"),
        "date": expense.get("date"),
        "risk_score": node_input.risk_score,
        "risk_factors": node_input.risk_factors,
        "risk_explanation": node_input.explanation,
        "decision": decision_val,
        "redacted_categories": ctx.state.get("redacted_categories"),
        "security_event": ctx.state.get("security_event", False),
    }
    yield Event(output=outcome)


human_approval_node = FunctionNode(
    func=human_approval,
    name="human_approval",
    rerun_on_resume=True,
)


# Terminal Outcome Node
def record_outcome(node_input: dict) -> dict:
    return node_input


# Workflow wrapping the graph
root_agent = Workflow(
    name="root_agent",
    input_schema=PubSubMessage,
    edges=[
        ("START", parse_expense),
        (
            parse_expense,
            {
                "auto_approve": record_outcome,
                "needs_review": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "clean": risk_assessor,
                "injection": human_approval_node,
            },
        ),
        (risk_assessor, human_approval_node),
        (human_approval_node, record_outcome),
    ],
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
