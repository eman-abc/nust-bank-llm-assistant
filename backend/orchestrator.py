"""
NUST Bank Assistant — LangGraph orchestrator (Phase 3).

This module wires the pipeline nodes into a single **deterministic** graph:

    guardrail → (safe?) → evaluator → (safe?) → retriever → synthesizer → END
        ↘ (unsafe) → END      ↘ (unsafe) → END

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
from backend.nodes.guardrail import guardrail_node
from backend.nodes.evaluator import run_evaluator
from backend.nodes.retriever import run_retriever
from backend.nodes.synthesizer import run_synthesizer
import textwrap
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# =============================================================================
# Routing: Semantic Firewall -> Evaluator
# =============================================================================

def route_after_guardrail(state: AgentState) -> str:
    """
    Conditional router following the Prompt Injection guardrail node.
    
    - If ``is_safe`` is False (injection detected) → terminate immediately (END).
    - Otherwise → continue to the standard PII/intent evaluator.
    """
    if state.get("is_safe") is False:
        return END
    return "evaluator"


# =============================================================================
# Routing: Evaluator -> Retriever
# =============================================================================

def route_after_evaluation(state: AgentState) -> str:
    """
    Conditional router following the evaluator node.

    - If ``is_safe`` is False → terminate immediately (skip retriever + synthesizer).
    - Otherwise → continue to the vector retriever.
    """
    if state.get("is_safe") is False:
        return END
    return "retriever"


# =============================================================================
# Graph construction
# =============================================================================

# Blank workflow over our TypedDict schema
workflow = StateGraph(AgentState)

# Register named nodes
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("evaluator", run_evaluator)
workflow.add_node("retriever", run_retriever)
workflow.add_node("synthesizer", run_synthesizer)

# 🚨 THE ENTRY POINT IS NOW THE FIREWALL 🚨
workflow.set_entry_point("guardrail")

# Branch after guardrail: unsafe ends the run; safe continues to evaluator.
workflow.add_conditional_edges(
    "guardrail",
    route_after_guardrail,
    {
        END: END,
        "evaluator": "evaluator",
    },
)

# Branch after evaluator: unsafe path ends the run; safe path loads FAISS context.
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

# Compiled runnable: invoke with a dict matching AgentState
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

        # Minimal seed state
        initial_state: AgentState = {"user_query": user_line}

        try:
            result: AgentState = bank_bot.invoke(initial_state)
        except Exception as exc:
            print(f"\n[Pipeline error] {exc.__class__.__name__}: {exc}")
            continue

        # --- Readable post-run report (good for demos / viva) ---
        safe = result.get("is_safe")
        flagged = safe is False

        print("\n" + "-" * 60)
        print("ROUTING / STATE SUMMARY")
        print("-" * 60)
        print(f"  System flagged (unsafe)?        {flagged}")
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