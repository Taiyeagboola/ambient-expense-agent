import json
import os
import sys
import base64
from pathlib import Path
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events.request_input import RequestInput
from google.adk.events.event import Event
from google.genai import types

# Add project root to path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from expense_agent.agent import app

def json_serializable_default(obj):
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return base64.b64encode(obj).decode("utf-8")
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_unset=True)
    if hasattr(obj, "dict"):
        return obj.dict()
    try:
        return str(obj)
    except Exception:
        return repr(obj)

def serialize_event(event):
    event_dict = {}
    if isinstance(event, RequestInput):
        event_dict = {
            "type": "RequestInput",
            "interrupt_id": event.interrupt_id,
            "message": event.message
        }
        author = "system"
        role = "model"
    elif isinstance(event, Event):
        output = event.output
        if output is not None:
            if hasattr(output, "model_dump"):
                output = output.model_dump()
            elif hasattr(output, "dict"):
                output = output.dict()
                
        # Also clean up actions dict
        actions = event.model_dump(exclude_unset=True).get("actions", {})
        if "state_delta" in actions and actions["state_delta"]:
            state_delta = {}
            for k, v in actions["state_delta"].items():
                if hasattr(v, "model_dump"):
                    state_delta[k] = v.model_dump()
                elif hasattr(v, "dict"):
                    state_delta[k] = v.dict()
                else:
                    state_delta[k] = v
            actions["state_delta"] = state_delta

        event_dict = {
            "type": "Event",
            "output": output,
            "actions": actions,
            "node_info": event.model_dump(exclude_unset=True).get("node_info", {})
        }
        if event.content:
            if hasattr(event.content, "model_dump"):
                event_dict["content"] = event.content.model_dump(exclude_unset=True)
            elif hasattr(event.content, "dict"):
                event_dict["content"] = event.content.dict()
            else:
                parts_list = []
                if hasattr(event.content, "parts") and event.content.parts:
                    for part in event.content.parts:
                        p_dict = {}
                        if getattr(part, "text", None):
                            p_dict["text"] = part.text
                        if getattr(part, "function_call", None):
                            fc = part.function_call
                            p_dict["function_call"] = {
                                "name": fc.name,
                                "args": fc.args,
                                "id": fc.id
                            }
                        if getattr(part, "function_response", None):
                            fr = part.function_response
                            p_dict["function_response"] = {
                                "name": fr.name,
                                "response": fr.response,
                                "id": fr.id
                            }
                        parts_list.append(p_dict)
                event_dict["content"] = {
                    "role": getattr(event.content, "role", "model"),
                    "parts": parts_list
                }
        author = event.author or "root_agent"
        role = "model"
    elif isinstance(event, dict):
        if "content" in event:
            return {
                "author": event.get("author", "user"),
                "content": event["content"]
            }
        event_dict = event
        author = event.get("author", "user")
        role = "user"
    else:
        event_dict = {"type": "Unknown", "str": str(event)}
        author = "unknown"
        role = "model"

    return {
        "author": author,
        "content": {
            "role": role,
            "parts": [{"text": json.dumps(event_dict, default=json_serializable_default)}]
        }
    }

async def run_scenario(case_id, payload):
    session_service = InMemorySessionService()
    session = await session_service.create_session(user_id="eval_user", app_name="expense_agent")
    runner = Runner(app=app, session_service=session_service)

    # Wrap payload in the expected message Content
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(payload))]
    )

    all_serialized_events = []
    
    # 1. Initial run
    events = []
    async for event in runner.run_async(
        new_message=message,
        user_id="eval_user",
        session_id=session.id,
    ):
        events.append(event)
        all_serialized_events.append(serialize_event(event))

    # 2. Intercept human approval if needed
    request_input = None
    for e in events:
        if isinstance(e, RequestInput):
            request_input = e
            break
        if isinstance(e, Event) and hasattr(e, "get_function_calls"):
            for fc in e.get_function_calls():
                if fc.name == "adk_request_input":
                    request_input = RequestInput(
                        interrupt_id=fc.args.get("interruptId") or fc.id,
                        message=fc.args.get("message"),
                    )
                    break
            if request_input:
                break
            
    if request_input and request_input.interrupt_id == "decision":
        # Automate decision
        decision = "reject" if case_id == "prompt_injection" else "approve"
        
        # Format resume message
        resume_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="decision",
                        id="decision",
                        response={"decision": decision},
                    )
                )
            ],
        )
        
        all_serialized_events.append({
            "author": "user",
            "content": {"role": "user", "parts": [{"text": f"Decision: {decision}"}]}
        })
        
        # Resume run
        resume_events = []
        async for event in runner.run_async(
            new_message=resume_message,
            user_id="eval_user",
            session_id=session.id,
        ):
            resume_events.append(event)
            all_serialized_events.append(serialize_event(event))
            
        events.extend(resume_events)

    # 3. Find final output
    final_output = None
    for e in reversed(events):
        if isinstance(e, Event) and e.output:
            final_output = e.output
            break

    return all_serialized_events, final_output

async def main():
    dataset_path = Path("tests/eval/datasets/basic-dataset.json")
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found.")
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    eval_cases = dataset.get("eval_cases", [])
    output_cases = []

    for case in eval_cases:
        case_id = case["eval_case_id"]
        prompt_text = case["prompt"]["parts"][0]["text"]
        payload = json.loads(prompt_text)

        print(f"Running evaluation case: {case_id}...")
        serialized_events, final_output = await run_scenario(case_id, payload)

        if final_output is not None:
            if hasattr(final_output, "model_dump"):
                final_response_text = json.dumps(final_output.model_dump())
            elif isinstance(final_output, dict):
                final_response_text = json.dumps(final_output)
            else:
                final_response_text = str(final_output)
        else:
            final_response_text = "No response"

        output_case = {
            "eval_case_id": case_id,
            "prompt": case["prompt"],
            "agent_data": {
                "turns": [
                    {
                        "turn_index": 0,
                        "events": serialized_events
                    }
                ]
            },
            "responses": [
                {
                    "response": {
                        "role": "model",
                        "parts": [{"text": final_response_text}]
                    }
                }
            ]
        }
        output_cases.append(output_case)

    output_path = Path("artifacts/traces/generated_traces.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_dataset = {"eval_cases": output_cases}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dataset, f, indent=2)

    print(f"Successfully generated {len(output_cases)} traces to {output_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
