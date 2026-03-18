from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# LOAD ENV FIRST — BEFORE ANYTHING ELSE
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# Now safe to import the rest of the app
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from app.agents.langgraph.suggestion_graph import build_suggestion_graph
from app.agents.langgraph.orchestrator_graph import build_orchestrator_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in your .env file.")
if not os.getenv("WEATHER_API_KEY"):
    raise RuntimeError("WEATHER_API_KEY is missing. Put it in your .env file.")
if not os.getenv("HOTELS_API_KEY"):
    raise RuntimeError("HOTELS_API_KEY is missing. Put it in your .env file.")

app = FastAPI(title="BookIT", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message.")


suggestion_executor = build_suggestion_graph()
orchestrator_executor = build_orchestrator_graph(suggestion_executor=suggestion_executor)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat_agentic")
def chat_agentic(req: ChatRequest):
    try:
        result = orchestrator_executor.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={"recursion_limit": 20},
        )

        final_obj = result.get("final") or {}
        return {
            "message": final_obj.get("message", ""),
            "blocks": final_obj.get("blocks", []) or [],
            "meta": {"trace_types": [m.__class__.__name__ for m in result["messages"]]},
        }

    except Exception as e:
        logger.error("chat_agentic failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))