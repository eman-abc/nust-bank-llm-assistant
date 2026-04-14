"""
NUST Bank Assistant - hybrid LangGraph orchestrator.
"""

from __future__ import annotations
import textwrap
from langgraph.graph import END, StateGraph

from backend.nodes.context_builder import run_context_builder
from backend.nodes.grounding_checker import run_grounding_checker
from backend.nodes.guardrail import guardrail_node
from backend.nodes.hybrid_retriever import run_hybrid_retriever
from backend.nodes.privacy_sanitizer import run_privacy_sanitizer
from backend.nodes.query_normalizer import run_query_normalizer
from backend.nodes.reranker import run_reranker
from backend.nodes.synthesizer import run_synthesizer
from backend.nodes.intent_classifier import run_intent_classifier
from backend.state import AgentState

def route_if_safe(state: AgentState) -> str:
    if state.get("is_safe") is False:
        return "refuse"
    return "continue"

def route_after_context_builder(state: AgentState) -> str:
    if (state.get("selected_context") or "").strip():
        return "synthesizer"
    return "grounding_checker"

def route_after_query_analysis(state: AgentState) -> str:
    if state.get("query_intent") == "greeting":
        return "synthesizer"
    return "hybrid_retriever"

workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("privacy_sanitizer", run_privacy_sanitizer)
workflow.add_node("query_normalizer", run_query_normalizer)
workflow.add_node("intent_classifier", run_intent_classifier)
workflow.add_node("hybrid_retriever", run_hybrid_retriever)
workflow.add_node("reranker", run_reranker)
workflow.add_node("context_builder", run_context_builder)
workflow.add_node("synthesizer", run_synthesizer)
workflow.add_node("grounding_checker", run_grounding_checker)

workflow.set_entry_point("guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_if_safe,
    {
        "refuse": "synthesizer",
        "continue": "privacy_sanitizer",
    },
)
workflow.add_conditional_edges(
    "privacy_sanitizer",
    route_if_safe,
    {
        END: END,
        "continue": "query_normalizer",
    },
)

workflow.add_edge("query_normalizer", "intent_classifier")
workflow.add_conditional_edges(
    "intent_classifier",
    route_after_query_analysis,
    {
        "synthesizer": "synthesizer",
        "hybrid_retriever": "hybrid_retriever",
    },
)
workflow.add_edge("hybrid_retriever", "reranker")
workflow.add_edge("reranker", "context_builder")

workflow.add_conditional_edges(
    "context_builder",
    route_after_context_builder,
    {
        "synthesizer": "synthesizer",
        "grounding_checker": END,
    },
)
workflow.add_edge("synthesizer", END)

bank_bot = workflow.compile()

if __name__ == "__main__":
    print("NUST Bank Assistant - Ready.")
    while True:
        try:
            user_line = input("\nYou: ").strip()
            if user_line.lower() == "exit": break
            result = bank_bot.invoke({"user_query": user_line})
            print(f"\nResponse: {result.get('final_response')}")
        except KeyboardInterrupt:
            break
