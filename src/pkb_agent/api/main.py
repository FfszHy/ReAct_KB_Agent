"""ASGI entry point: ``uvicorn pkb_agent.api.main:app --reload``."""

from pkb_agent.api.app import create_app

app = create_app()
