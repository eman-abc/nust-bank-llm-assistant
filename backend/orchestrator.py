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
from backend.state import AgentState


def route_if_safe(state: AgentState) -> str:
    return END if state.get("is_safe") is False else "continue"


def route_after_context_builder(state: AgentState) -> str:
    if (state.get("selected_context") or "").strip():
        return "synthesizer"
    return "grounding_checker"


workflow = StateGraph(AgentState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("privacy_sanitizer", run_privacy_sanitizer)
workflow.add_node("query_normalizer", run_query_normalizer)
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
        END: END,
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

workflow.add_edge("query_normalizer", "hybrid_retriever")
workflow.add_edge("hybrid_retriever", "reranker")
workflow.add_edge("reranker", "context_builder")
workflow.add_conditional_edges(
    "context_builder",
    route_after_context_builder,
    {
        "synthesizer": "synthesizer",
        "grounding_checker": "grounding_checker",
    },
)
workflow.add_edge("synthesizer", "grounding_checker")
workflow.add_edge("grounding_checker", END)

bank_bot = workflow.compile()


if __name__ == "__main__":
    print(
        textwrap.dedent(
            """
            ============================================================
            NUST Bank Assistant - Hybrid LangGraph CLI
            Type your question, or 'exit' to quit.
            ============================================================
            """
        ).strip()
    )

    while True:
        try:
            user_line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_line.lower() == "exit":
            print("Goodbye.")
            break

        result: AgentState = bank_bot.invoke({"user_query": user_line})
        print("\n" + "-" * 60)
        print("STATE SUMMARY")
        print("-" * 60)
        print(f"  is_safe                 {result.get('is_safe')!r}")
        print(f"  query_intent            {result.get('query_intent', '')!r}")
        print(f"  retrieval_confidence    {result.get('retrieval_confidence', 0.0)!r}")
        print(f"  grounding_passed        {result.get('grounding_passed')!r}")
        print(f"  citations               {result.get('citations', [])!r}")
        print("-" * 60)
        print("FINAL RESPONSE")
        print("-" * 60)
        print((result.get("final_response") or "").strip() or "(empty)")
        print("-" * 60)
