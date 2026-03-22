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
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
        blocks = final_obj.get("blocks", []) or []

        # Automatically analyze prices if hotel_price blocks exist
        price_analysis = None
        price_blocks = [b for b in blocks if b.get("type") == "hotel_price"]
        if price_blocks:
            lines = []
            for block in price_blocks:
                data = block.get("data", {})
                hotel = data.get("hotel", "Unknown")
                snippets = [
                    r.get("snippet", "")
                    for r in (data.get("price_results") or [])
                    if r.get("snippet")
                ]
                snippet_text = " | ".join(snippets) if snippets else "No price data found."
                lines.append(f"- {hotel}: {snippet_text}")

            price_summary = "\n".join(lines)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            msgs = [
                SystemMessage(content=(
                    "You are a hotel price analyst. "
                    "Given a list of hotels and their price information scraped from the web, "
                    "identify which is the cheapest and which offers the best value for money. "
                    "Be concise. Do not output JSON or markdown tables."
                )),
                HumanMessage(content=f"Hotel prices:\n{price_summary}"),
            ]
            resp = llm.invoke(msgs)
            price_analysis = getattr(resp, "content", "")

        return {
            "message": final_obj.get("message", ""),
            "blocks": blocks,
            "price_analysis": price_analysis,
            "meta": {"trace_types": [m.__class__.__name__ for m in result["messages"]]},
        }

    except Exception as e:
        logger.error("chat_agentic failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))