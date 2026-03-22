from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.prompts.load_prompt import load_prompt
from app.agents.tool_registry.tools_registry import (
    get_weather_forecast,
    search_hotels,
    search_hotel_price,
)


class OrchestratorState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Dict[str, Any]
    suggestions: Optional[Dict[str, Any]]
    blocks: Optional[List[Dict[str, Any]]]
    final: Optional[Dict[str, Any]]


RouteNext = Literal["suggestion_only", "tools_only", "tools_then_suggestion"]


class Plan(BaseModel):
    route: RouteNext
    city: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    need_weather: bool = False
    need_hotels: bool = False
    min_rating: int = 4
    hotel_limit: int = 10
    suggestion_region: str = "Europe"


PLANNER_PROMPT = load_prompt("router_prompt.txt")
FINAL_SYSTEM_PROMPT = load_prompt("final_prompt.txt")


def _last_user_message(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _try_parse_str(s: str) -> Optional[Any]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _tool_name(m: ToolMessage) -> str:
    name = (getattr(m, "name", None) or "").strip()
    if name:
        return name
    ak = getattr(m, "additional_kwargs", None) or {}
    return (ak.get("name") or "").strip()


def _collect_tool_payloads(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tool = _tool_name(m)
        content = getattr(m, "content", None)

        if isinstance(content, dict):
            d = dict(content)
            d["_tool"] = tool
            payloads.append(d)
            continue

        if isinstance(content, str):
            parsed = _try_parse_str(content)
            if isinstance(parsed, dict):
                parsed["_tool"] = tool
                payloads.append(parsed)
            else:
                payloads.append({"_tool": tool, "_raw": content})
            continue

        payloads.append({"_tool": tool, "_raw": content})

    return payloads


def _build_blocks(
    tool_payloads: List[Dict[str, Any]],
    suggestions: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    for p in tool_payloads:
        tool = (p.get("_tool") or "").strip()

        if tool == "get_weather_forecast" or ("forecast" in p and "city" in p):
            data = {k: v for k, v in p.items() if k != "_tool"}
            blocks.append({"type": "weather_forecast", "data": data})
            continue

        if tool == "search_hotels" or ("hotels" in p and "query" in p):
            blocks.append(
                {
                    "type": "hotel_list",
                    "data": {
                        "query": p.get("query"),
                        "items": p.get("hotels", []),
                        "error": p.get("error"),
                    },
                }
            )
            continue

        if tool == "search_hotel_price":
            blocks.append(
                {
                    "type": "hotel_price",
                    "data": {
                        "hotel": p.get("hotel"),
                        "city": p.get("city"),
                        "price_results": p.get("price_results", []),
                    },
                }
            )
            continue

    if suggestions and isinstance(suggestions, dict) and "suggestions" in suggestions:
        blocks.append({"type": "destination_suggestions", "data": suggestions})

    return blocks


def _context_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    for b in blocks:
        t = b.get("type")
        data = b.get("data") or {}

        if t == "weather_forecast":
            city = data.get("city", "")
            forecast = data.get("forecast", []) or []
            lines.append(f"Weather forecast for {city}: {len(forecast)} points.")

        elif t == "hotel_list":
            items = data.get("items") or []
            err = data.get("error")
            lines.append(f"Hotels returned: {len(items)}.")
            if err:
                lines.append(f"Hotels error: {err}")

        elif t == "hotel_price":
            hotel = data.get("hotel", "")
            snippets = [r.get("snippet", "") for r in (data.get("price_results") or [])]
            price_text = " | ".join(s for s in snippets if s)
            lines.append(f"Price info for {hotel}: {price_text}")

        elif t == "destination_suggestions":
            suggs = data.get("suggestions") or []
            lines.append(f"Destination suggestions: {len(suggs)}.")

    return "\n".join(lines).strip()


def build_orchestrator_graph(*, suggestion_executor):
    planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Plan)
    final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    tools = [get_weather_forecast, search_hotels, search_hotel_price]
    tool_node = ToolNode(tools)

    def plan_node(state: OrchestratorState):
        last_user = _last_user_message(state["messages"])
        prompt = PLANNER_PROMPT.replace("{user_message}", last_user)
        plan = planner_llm.invoke(prompt)
        return {"plan": plan.model_dump()}

    def tool_calls_node(state: OrchestratorState):
        """Produce weather and hotel tool calls based on plan."""
        plan = state.get("plan") or {}
        city = (plan.get("city") or "").strip()

        calls: List[Dict[str, Any]] = []

        if plan.get("need_weather") and city:
            calls.append(
                {"name": "get_weather_forecast", "args": {"city": city}, "id": "call_weather_1"}
            )

        if plan.get("need_hotels") and city:
            calls.append(
                {
                    "name": "search_hotels",
                    "args": {
                        "city": city,
                        "min_rating": int(plan.get("min_rating", 4)),
                        "limit": int(plan.get("hotel_limit", 10)),
                        "amenities": plan.get("amenities") or None,
                    },
                    "id": "call_hotels_1",
                }
            )

        if not calls:
            return {}

        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    def price_calls_node(state: OrchestratorState):
        """After hotels are fetched, fire price searches for the top 3."""
        plan = state.get("plan") or {}
        if not plan.get("need_hotels"):
            return {}

        tool_payloads = _collect_tool_payloads(state["messages"])
        hotels: List[Dict[str, Any]] = []
        for p in tool_payloads:
            if p.get("_tool") == "search_hotels" or ("hotels" in p and "query" in p):
                hotels = p.get("hotels", [])
                break

        if not hotels:
            return {}

        city = (plan.get("city") or "").strip()
        calls = [
            {
                "name": "search_hotel_price",
                "args": {"hotel_name": h["name"], "city": city},
                "id": f"call_price_{i}",
            }
            for i, h in enumerate(hotels[:3])
        ]

        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    def suggestion_node(state: OrchestratorState):
        """Invoke SuggestionAgent when the plan requests it."""
        last_user = _last_user_message(state["messages"])

        result = suggestion_executor.invoke(
            {"messages": [HumanMessage(content=last_user)]},
            config={"recursion_limit": 10},
        )

        final_msg = result["messages"][-1]
        content = getattr(final_msg, "content", "")

        if isinstance(content, str):
            parsed_any = _try_parse_str(content)
            parsed = (
                parsed_any
                if isinstance(parsed_any, dict)
                else {"error": "bad_suggestion_json", "raw": content}
            )
        else:
            parsed = {"error": "bad_suggestion_content_type", "raw": content}

        return {"messages": [final_msg], "suggestions": parsed}

    def final_node(state: OrchestratorState):
        tool_payloads = _collect_tool_payloads(state["messages"])
        blocks = _build_blocks(tool_payloads, state.get("suggestions"))

        last_user = _last_user_message(state["messages"])
        context_text = _context_from_blocks(blocks)

        # Collect all hotel price blocks into one summary for the LLM
        price_blocks = [b for b in blocks if b["type"] == "hotel_price"]
        price_summary = ""
        if price_blocks:
            lines = []
            for b in price_blocks:
                hotel = b["data"].get("hotel", "")
                snippets = [r.get("snippet", "") for r in (b["data"].get("price_results") or []) if r.get("snippet")]
                snippet_text = " | ".join(snippets) if snippets else "No price data found."
                lines.append(f"- {hotel}: {snippet_text}")
            price_summary = "Hotel price information:\n" + "\n".join(lines)

        user_content = f"User request:\n{last_user}\n\nContext:\n{context_text}"
        if price_summary:
            user_content += f"\n\n{price_summary}\n\nBased on the price information above, tell the user which hotel offers the best value for money and which is the cheapest option."

        msgs = [
            SystemMessage(content=FINAL_SYSTEM_PROMPT),
            HumanMessage(content=user_content.strip()),
        ]
        resp = final_llm.invoke(msgs)

        final_obj = {"message": getattr(resp, "content", ""), "blocks": blocks}
        return {"messages": [resp], "final": final_obj}

    # Routing functions

    def route_from_plan(state: OrchestratorState):
        r = (state.get("plan") or {}).get("route")
        if r == "suggestion_only":
            return "suggestion"
        return "tool_calls"  # tools_only or tools_then_suggestion both start with tools

    def after_price_tools(state: OrchestratorState):
        r = (state.get("plan") or {}).get("route")
        if r == "tools_then_suggestion":
            return "suggestion"
        return "final"

    # Graph

    graph = StateGraph(OrchestratorState)

    graph.add_node("planner", plan_node)
    graph.add_node("tool_calls", tool_calls_node)
    graph.add_node("tools", tool_node)
    graph.add_node("price_calls", price_calls_node)
    graph.add_node("price_tools", tool_node)
    graph.add_node("suggestion", suggestion_node)
    graph.add_node("finalizer", final_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_from_plan,
        {"suggestion": "suggestion", "tool_calls": "tool_calls"},
    )

    graph.add_edge("tool_calls", "tools")
    graph.add_edge("tools", "price_calls")
    graph.add_edge("price_calls", "price_tools")

    graph.add_conditional_edges(
        "price_tools",
        after_price_tools,
        {"suggestion": "suggestion", "final": "finalizer"},
    )

    graph.add_edge("suggestion", "finalizer")
    graph.add_edge("finalizer", END)

    return graph.compile()