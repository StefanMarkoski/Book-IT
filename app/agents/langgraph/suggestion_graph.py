from __future__ import annotations

from typing import TypedDict, Annotated, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.tool_registry.tools_registry import web_search
from app.prompts.load_prompt import load_prompt


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


SUGGESTION_SYSTEM_PROMPT = load_prompt("suggestion_system.txt")


def build_suggestion_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    tools = [web_search]

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SUGGESTION_SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()