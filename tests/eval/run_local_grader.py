import os
import json
import sys
import time
import traceback

# 1. Load environment variables from .env
with open(".env", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

# Set mandatory CLI environment variables
os.environ["GOOGLE_CLOUD_PROJECT"] = "dummy-project"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# 2. Mock google.auth.default and google.cloud.storage.Client
import google.auth
from google.auth.credentials import Credentials

class DummyCredentials(Credentials):
    def __init__(self):
        super().__init__()
        self.token = "dummy-token"
    def refresh(self, request):
        self.token = "dummy-token"

def mock_default(*args, **kwargs):
    return DummyCredentials(), "dummy-project"

google.auth.default = mock_default

import google.cloud.storage as storage
class DummyStorageClient:
    def __init__(self, *args, **kwargs):
        pass
storage.Client = DummyStorageClient

# 3. Patch vertexai Client's evaluate method
import google.genai
from google.genai import types
import vertexai
from vertexai._genai.types.common import (
    EvaluationResult, EvalCaseResult, ResponseCandidateResult, 
    EvalCaseMetricResult, AggregatedMetricResult
)

def local_evaluate(self, dataset, metrics, **kwargs):
    try:
        print("\n--- Running Monkeypatched Local LLM-as-Judge Evaluation ---", flush=True)
        api_key = os.environ.get("GEMINI_API_KEY")
        genai_client = google.genai.Client(api_key=api_key)
        
        eval_case_results = []
        metric_scores = {metric.name: [] for metric in metrics}
        metric_errors = {metric.name: 0 for metric in metrics}
        
        for case_idx, case in enumerate(dataset.eval_cases):
            case_id = case.eval_case_id or f"case_{case_idx}"
            print(f"Evaluating case '{case_id}'...", flush=True)
            
            prompt_text = case.prompt.parts[0].text if case.prompt and case.prompt.parts else ""
            response_text = case.responses[0].response.parts[0].text if case.responses and case.responses[0].response and case.responses[0].response.parts else ""
            
            # Format agent trace events
            agent_data_str = ""
            if case.agent_data and case.agent_data.turns:
                turns_list = []
                for turn in case.agent_data.turns:
                    turn_dict = {"turn_index": turn.turn_index, "events": []}
                    if turn.events:
                        for event in turn.events:
                            event_text = event.content.parts[0].text if event.content and event.content.parts else ""
                            try:
                                event_data = json.loads(event_text)
                                turn_dict["events"].append({
                                    "author": event.author,
                                    "details": event_data
                                })
                            except Exception:
                                turn_dict["events"].append({
                                    "author": event.author,
                                    "details": event_text
                                })
                    turns_list.append(turn_dict)
                agent_data_str = json.dumps(turns_list, indent=2)
            else:
                agent_data_str = "No agent trace data available."

            metric_results = {}
            for metric in metrics:
                rendered_prompt = (metric.prompt_template
                    .replace("{prompt}", prompt_text)
                    .replace("{response}", response_text)
                    .replace("{agent_data}", agent_data_str))
                
                # We will attempt fallback models if the first model fails with quota issues
                models_to_try = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-2.0-flash", "gemini-2.5-pro"]
                success = False
                err_msg = ""
                
                for model_name in models_to_try:
                    max_retries = 5
                    backoff_factor = 2
                    initial_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            # Baseline sleep to ease API calling rate
                            time.sleep(1.5)
                            
                            # Run the model judge call using AI Studio
                            response = genai_client.models.generate_content(
                                model=model_name,
                                contents=rendered_prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json"
                                )
                            )
                            
                            res_json = json.loads(response.text)
                            score = float(res_json.get("score", 1.0))
                            explanation = str(res_json.get("explanation", "No explanation provided."))
                            
                            print(f"  Metric {metric.name} evaluated successfully using model {model_name}: Score = {score}", flush=True)
                            metric_results[metric.name] = EvalCaseMetricResult(
                                metric_name=metric.name,
                                score=score,
                                explanation=explanation
                            )
                            metric_scores[metric.name].append(score)
                            success = True
                            break  # exit retry loop for current model
                        except Exception as e:
                            err_msg = str(e)
                            
                            # Check if the error indicates a daily quota exhaustion
                            is_daily_quota = "GenerateRequestsPerDay" in err_msg or "daily" in err_msg.lower() or "limit: 20" in err_msg or "limit: 15" in err_msg
                            if is_daily_quota:
                                print(f"  Model {model_name} hit daily request quota limit. Falling back to next model...", flush=True)
                                break  # exit retry loop for current model, will try next model
                            
                            # Check if error is transient / retryable (429, 503, connection timeout, etc.)
                            is_retryable = "429" in err_msg or "503" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "UNAVAILABLE" in err_msg or "10060" in err_msg or "disconnected" in err_msg.lower() or "timeout" in err_msg.lower() or "connection" in err_msg.lower()
                            if is_retryable and attempt < max_retries - 1:
                                delay = initial_delay * (backoff_factor ** attempt)
                                print(f"  Attempt {attempt + 1} with model {model_name} failed: {e}. Retrying in {delay}s...", flush=True)
                                time.sleep(delay)
                            else:
                                print(f"  Failed with model {model_name}: {e}", flush=True)
                                break  # exit retry loop for current model
                                
                    if success:
                        break  # exit model fallback loop
                        
                if not success:
                    print(f"  All models failed for metric {metric.name} on case {case_id}. Final error: {err_msg}", flush=True)
                    metric_results[metric.name] = EvalCaseMetricResult(
                        metric_name=metric.name,
                        error_message=err_msg
                    )
                    metric_errors[metric.name] += 1
                    
            eval_case_results.append(EvalCaseResult(
                eval_case_index=case_idx,
                response_candidate_results=[
                    ResponseCandidateResult(
                        response_index=0,
                        metric_results=metric_results
                    )
                ]
            ))
            
        summary_metrics = []
        for metric in metrics:
            scores = metric_scores[metric.name]
            num_error = metric_errors[metric.name]
            num_valid = len(scores)
            num_total = num_valid + num_error
            
            mean_score = sum(scores) / num_valid if num_valid > 0 else 0.0
            
            stdev_score = 0.0
            if num_valid > 1:
                variance = sum((s - mean_score) ** 2 for s in scores) / (num_valid - 1)
                stdev_score = variance ** 0.5
                
            pass_rate = sum(1 for s in scores if s >= 3) / num_valid if num_valid > 0 else 0.0
            
            summary_metrics.append(AggregatedMetricResult(
                metric_name=metric.name,
                num_cases_total=num_total,
                num_cases_valid=num_valid,
                num_cases_error=num_error,
                mean_score=mean_score,
                stdev_score=stdev_score,
                pass_rate=pass_rate
            ))
            
        print("--- Monkeypatched Local Evaluation Complete ---\n", flush=True)
        return EvaluationResult(
            eval_case_results=eval_case_results,
            summary_metrics=summary_metrics,
            evaluation_dataset=[dataset]
        )
    except Exception as outer_e:
        print("CRITICAL EXCEPTION IN LOCAL_EVALUATE:", flush=True)
        traceback.print_exc()
        raise outer_e

# Apply patch to Evals class
from vertexai._genai import evals as vertexai_evals
vertexai_evals.Evals.evaluate = local_evaluate

# 4. Import and execute the agents-cli click command
from google.agents.cli.eval.cmd_grade import cmd_grade

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [
            "--traces", "artifacts/traces/generated_traces.json",
            "--config", "tests/eval/eval_config.yaml"
        ]
    cmd_grade.main(args=args)
