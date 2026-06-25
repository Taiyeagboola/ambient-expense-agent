# Makefile for Ambient Expense Agent

.PHONY: install playground test lint run generate-traces grade

install:
	uv sync --dev --extra lint

playground:
	uv run agents-cli playground

test:
	uv run pytest

lint:
	uv run agents-cli lint

run:
	uv run uvicorn expense_agent.agent_runtime_app:app --host 127.0.0.1 --port 8080 --reload

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	C:\Users\Admiun\AppData\Roaming\uv\tools\google-agents-cli\Scripts\python.exe tests/eval/run_local_grader.py


