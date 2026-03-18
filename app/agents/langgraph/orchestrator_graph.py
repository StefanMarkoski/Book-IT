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
from app.agents.tool_registry.tools_registry import get_weather_forecast, search_hotels


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


def _build_blocks(tool_payloads: List[Dict[str, Any]], suggestions: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    if suggestions and isinstance(suggestions, dict) and "suggestions" in suggestions:
        blocks.append({"type": "destination_suggestions", "data": suggestions})

    return blocks


def _context_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    """
    Keep this minimal. It's only for the final LLM to render a response.
    """
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

        elif t == "destination_suggestions":
            suggs = data.get("suggestions") or []
            lines.append(f"Destination suggestions: {len(suggs)}.")

    return "\n".join(lines).strip()


def build_orchestrator_graph(*, suggestion_executor):
    planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Plan)
    final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    tools = [get_weather_forecast, search_hotels]
    tool_node = ToolNode(tools)

    def plan_node(state: OrchestratorState):
        last_user = _last_user_message(state["messages"])
        prompt = PLANNER_PROMPT.replace("{user_message}", last_user)
        plan = planner_llm.invoke(prompt)
        return {"plan": plan.model_dump()}

    def tool_calls_node(state: OrchestratorState):
        """
        Produce tool calls based on plan. ToolNode will execute them.
        """
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

    def suggestion_node(state: OrchestratorState):
        """
        Invoke SuggestionAgent when the plan requests it.
        Suggestions should be generic destination ideas (not hotels).
        """
        last_user = _last_user_message(state["messages"])

        result = suggestion_executor.invoke(
            {"messages": [HumanMessage(content=last_user)]},
            config={"recursion_limit": 10},
        )

        final_msg = result["messages"][-1]
        content = getattr(final_msg, "content", "")

        if isinstance(content, str):
            parsed_any = _try_parse_str(content)
            parsed = parsed_any if isinstance(parsed_any, dict) else {"error": "bad_suggestion_json", "raw": content}
        else:
            parsed = {"error": "bad_suggestion_content_type", "raw": content}

        return {"messages": [final_msg], "suggestions": parsed}

    def final_node(state: OrchestratorState):
        tool_payloads = _collect_tool_payloads(state["messages"])
        blocks = _build_blocks(tool_payloads, state.get("suggestions"))

        last_user = _last_user_message(state["messages"])
        context_text = _context_from_blocks(blocks)

        msgs = [
            SystemMessage(content=FINAL_SYSTEM_PROMPT),
            HumanMessage(content=f"User request:\n{last_user}\n\nContext:\n{context_text}".strip()),
        ]
        resp = final_llm.invoke(msgs)

        final_obj = {"message": getattr(resp, "content", ""), "blocks": blocks}
        return {"messages": [resp], "final": final_obj}

    # ------- Routing functions (based ONLY on plan) -------

    def route_from_plan(state: OrchestratorState):
        r = (state.get("plan") or {}).get("route")
        if r == "suggestion_only":
            return "suggestion"
        return "tool_calls"  # tools_only or tools_then_suggestion both start with tools

    def after_tools(state: OrchestratorState):
        r = (state.get("plan") or {}).get("route")
        if r == "tools_then_suggestion":
            return "suggestion"
        return "final"

    # ------- Graph -------

    graph = StateGraph(OrchestratorState)

    graph.add_node("plan", plan_node)
    graph.add_node("tool_calls", tool_calls_node)
    graph.add_node("tools", tool_node)
    graph.add_node("suggestion", suggestion_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("plan")

    graph.add_conditional_edges(
        "plan",
        route_from_plan,
        {"suggestion": "suggestion", "tool_calls": "tool_calls"},
    )

    graph.add_edge("tool_calls", "tools")

    graph.add_conditional_edges(
        "tools",
        after_tools,
        {"suggestion": "suggestion", "final": "final"},
    )

    graph.add_edge("suggestion", "final")
    graph.add_edge("final", END)

    return graph.compile()