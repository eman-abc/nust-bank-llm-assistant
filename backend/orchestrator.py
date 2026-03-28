"""
NUST Bank Assistant — LangGraph orchestrator (Phase 3).

This module wires the three pipeline nodes into a single **deterministic** graph:

    evaluator → (safe?) → retriever → synthesizer → END
                    ↘ (unsafe) → END

The graph shares one `AgentState` object across steps; each node returns a **partial**
update dict that LangGraph merges into the running state.

Run locally (from repository root, with venv + deps installed):

    python -m backend.orchestrator
"""

from __future__ import annotations

import textwrap

# ---------------------------------------------------------------------------
# 1. LangGraph core: StateGraph defines the workflow; END is the terminal sink.
# ---------------------------------------------------------------------------
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# 2. Shared typed state (single source of truth for all nodes).
# ---------------------------------------------------------------------------
from backend.state import AgentState

# ---------------------------------------------------------------------------
# 3. Node callables — each accepts full AgentState and returns a partial update.
# ---------------------------------------------------------------------------
from backend.evaluator import run_evaluator
from backend.retriever import run_retriever
from backend.synthesizer import run_synthesizer


# =============================================================================
# Routing: after the security evaluator, either stop or continue to retrieval.
# =============================================================================


def route_after_evaluation(state: AgentState) -> str:
    """
    Conditional router following the evaluator node.

    Contract (assignment spec):
      - If ``is_safe`` is False → terminate immediately (skip retriever + synthesizer).
      - Otherwise → continue to the vector retriever.

    LangGraph's ``END`` constant is the interned string ``__end__``; returning it
    matches the ``END`` key in ``path_map``. Returning ``"retriever"`` routes to
    that node by name.
    """
    if state.get("is_safe") is False:
        return END
    return "retriever"


# =============================================================================
# Graph construction
# =============================================================================

# Blank workflow over our TypedDict schema — LangGraph validates merges against these keys.
workflow = StateGraph(AgentState)

# Register named nodes (strings are how edges refer to them).
workflow.add_node("evaluator", run_evaluator)
workflow.add_node("retriever", run_retriever)
workflow.add_node("synthesizer", run_synthesizer)

# First hop: START → evaluator (same as set_entry_point("evaluator")).
workflow.set_entry_point("evaluator")

# Branch after evaluator: unsafe path ends the run; safe path loads FAISS context.
# path_map tells LangGraph how to interpret the return value of route_after_evaluation.
workflow.add_conditional_edges(
    "evaluator",
    route_after_evaluation,
    {
        END: END,
        "retriever": "retriever",
    },
)

# Linear tail: retrieved chunks → LLM answer → graph END.
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("synthesizer", END)

# Compiled runnable: invoke with a dict matching AgentState (at least user_query).
bank_bot = workflow.compile()


# =============================================================================
# Local CLI tester — run:  python -m backend.orchestrator
# =============================================================================

if __name__ == "__main__":
    print(
        textwrap.dedent(
            """
            ============================================================
            NUST Bank Assistant — LangGraph CLI (Phase 3)
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

        # Minimal seed state — downstream nodes fill is_safe, scrubbed_query, etc.
        initial_state: AgentState = {"user_query": user_line}

        try:
            result: AgentState = bank_bot.invoke(initial_state)
        except Exception as exc:
            # Should be rare if nodes fail gracefully; never crash the REPL silently.
            print(f"\n[Pipeline error] {exc.__class__.__name__}: {exc}")
            continue

        # --- Readable post-run report (good for demos / viva) ---
        safe = result.get("is_safe")
        flagged = safe is False

        print("\n" + "-" * 60)
        print("ROUTING / STATE SUMMARY")
        print("-" * 60)
        print(f"  Evaluator flagged (unsafe)?     {flagged}")
        print(f"  is_safe                         {safe!r}")
        print(f"  scrubbed_query                  {result.get('scrubbed_query', '')!r}")

        ctx = result.get("retrieved_context") or ""
        ctx_preview = (ctx[:400] + "…") if len(ctx) > 400 else ctx
        print(f"  retrieved_context (preview)     {ctx_preview!r}")

        print("-" * 60)
        print("FINAL RESPONSE (customer-facing)")
        print("-" * 60)
        print((result.get("final_response") or "").strip() or "(empty)")
        print("-" * 60)
