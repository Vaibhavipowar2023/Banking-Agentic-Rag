import os
import json
import math
import pandas as pd
import streamlit as st
from agent import get_agent, run_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Banking RAG Agent",
    page_icon="🏦",
    layout="centered",
)

st.markdown("""
<style>
    .step-box {
        background: #f0f4ff;
        border-left: 3px solid #4361ee;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.83rem;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-score {
        font-size: 2rem;
        font-weight: 600;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


def check_setup():
    if not os.path.exists("qdrant_storage"):
        st.error("""
### Database not found
Run ingestion first:
python ingest.py
Then refresh this page.
        """)
        st.stop()

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY missing. Add to .env file.")
        st.stop()

    if not os.getenv("COHERE_API_KEY"):
        st.error("COHERE_API_KEY missing. Add to .env file.")
        st.stop()


def display_steps(steps):
    if not steps:
        return
    with st.expander(
        f"Agent thinking steps ({len(steps)} tool used)"
    ):
        for step in steps:
            st.markdown(f"""
<div class="step-box">
    <strong>Step {step['step']} — {step['tool']}</strong><br>
    <small>Searched for: {step['tool_input']}</small><br><br>
    <small>{step['observation'][:200]}...</small>
</div>
""", unsafe_allow_html=True)


def load_eval_results():
    path = "eval_data/evaluation_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def main():
    check_setup()

    st.title("🏦 Banking RAG Agent")
    st.caption(
        "AI powered banking assistant using "
        "Qdrant + Cohere + Groq + LangChain"
    )
    st.divider()

    @st.cache_resource(show_spinner="Loading AI agent...")
    def load_agent():
        return get_agent()

    agent_executor = load_agent()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
**Banking RAG Agent** answers questions
from official RBI and SEBI documents.

**Tools available:**
- Search banking docs
- Calculate EMI
- Compare schemes
- Web search
- Get RBI circular

**Tech stack:**
- Groq Llama 3.1 8B Instant
- Cohere embed-english-v3.0
- Qdrant vector database
- LangChain ReAct agent
- RAGAS evaluation
        """)
        st.divider()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.caption("Data: Official RBI/SEBI documents")

    # 3 Tabs
    tab1, tab2, tab3 = st.tabs([
        "Banking Assistant",
        "EMI Calculator",
        "Evaluation",
    ])

    # ── Tab 1: Banking Assistant ──────────────────────────────
    with tab1:
        st.subheader("Ask any banking question")
        st.caption(
            "Questions answered from official RBI documents"
        )

        if "messages" not in st.session_state or \
                len(st.session_state.get("messages", [])) == 0:
            st.markdown("#### Try asking:")
            cols = st.columns(2)
            examples = [
                "What documents do I need for KYC?",
                "How do I file a Banking Ombudsman complaint?",
                "What are RBI rules for credit card charges?",
                "What is Video KYC as per RBI?",
                "What is the time limit for Ombudsman complaint?",
                "Can a credit card be issued without my consent?",
            ]
            for i, q in enumerate(examples):
                col = cols[i % 2]
                if col.button(q, use_container_width=True):
                    st.session_state.setdefault("messages", [])
                    st.session_state["pending"] = q
                    st.rerun()

        st.session_state.setdefault("messages", [])

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "steps" in msg:
                    display_steps(msg["steps"])

        query = None
        if "pending" in st.session_state:
            query = st.session_state.pop("pending")
        else:
            query = st.chat_input("Ask a banking question...")

        if query:
            with st.chat_message("user"):
                st.write(query)
            st.session_state.messages.append({
                "role":    "user",
                "content": query,
            })

            with st.chat_message("assistant"):
                with st.spinner("Searching RBI documents..."):
                    try:
                        result = run_agent(agent_executor, query)
                        answer = result["answer"]
                        steps  = result["steps"]
                    except Exception as e:
                        answer = f"Error: {str(e)}"
                        steps  = []

                st.write(answer)
                display_steps(steps)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "steps":   steps,
            })

    # ── Tab 2: EMI Calculator ─────────────────────────────────
    with tab2:
        st.subheader("Loan EMI Calculator")
        st.caption("Calculate your monthly EMI for any loan")

        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input(
                "Loan Amount (Rs)",
                min_value=10000,
                max_value=100000000,
                value=5000000,
                step=100000,
            )

        with col2:
            rate = st.number_input(
                "Interest Rate (% per year)",
                min_value=1.0,
                max_value=30.0,
                value=8.5,
                step=0.1,
            )

        with col3:
            tenure = st.number_input(
                "Tenure (years)",
                min_value=1,
                max_value=30,
                value=20,
                step=1,
            )

        if st.button(
            "Calculate EMI",
            type="primary",
            use_container_width=True,
        ):
            monthly_rate   = rate / (12 * 100)
            months         = int(tenure * 12)
            emi            = (
                amount * monthly_rate *
                math.pow(1 + monthly_rate, months)
            ) / (math.pow(1 + monthly_rate, months) - 1)
            total_payment  = emi * months
            total_interest = total_payment - amount

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Monthly EMI",    f"Rs {emi:,.0f}")
            c2.metric("Total Payment",  f"Rs {total_payment:,.0f}")
            c3.metric("Total Interest", f"Rs {total_interest:,.0f}")

            st.info(f"""
**Loan Summary:**
- Principal: Rs {amount:,.0f}
- Interest Rate: {rate}% per annum
- Tenure: {tenure} years ({months} months)
- Monthly EMI: Rs {emi:,.0f}
- Total Interest: Rs {total_interest:,.0f}
- Interest as % of principal: {(total_interest/amount*100):.1f}%
            """)

            with st.spinner("Fetching RBI guidelines..."):
                result = run_agent(
                    agent_executor,
                    "What are RBI guidelines for home loans?"
                )
                st.markdown("**RBI Guidelines:**")
                st.write(result["answer"])

    # ── Tab 3: Evaluation ─────────────────────────────────────
    with tab3:
        st.subheader("Evaluation Dashboard")
        st.caption(
            "Accuracy metrics measured using manual RAGAS scoring"
        )

        eval_results = load_eval_results()

        if not eval_results:
            st.warning("""
Evaluation has not been run yet.

Run this command first:
python evaluate.py
Then refresh this page.
            """)
        else:
            overall = eval_results.get("overall_scores", {})

            st.markdown("#### Overall Scores")
            cols = st.columns(3)
            metrics = [
                ("faithfulness",      "Faithfulness"),
                ("answer_relevancy",  "Answer Relevancy"),
                ("context_precision", "Context Precision"),
            ]

            for i, (key, label) in enumerate(metrics):
                score = overall.get(key, 0)
                color = (
                    "#28a745" if score >= 0.8
                    else "#ffc107" if score >= 0.6
                    else "#dc3545"
                )
                cols[i].markdown(f"""
<div class="metric-box">
    <div class="metric-score" style="color:{color}">
        {score:.2f}
    </div>
    <div class="metric-label">{label}</div>
</div>
""", unsafe_allow_html=True)

            st.divider()
            st.markdown("#### Per Question Breakdown")
            per_q = eval_results.get("per_question", [])

            if per_q:
                rows = []
                for i, item in enumerate(per_q):
                    row = {
                        "Q":        f"Q{i+1}",
                        "Question": item["question"][:60] + "...",
                    }
                    scores = item.get("scores", {})
                    for key, label in metrics:
                        row[label] = round(scores.get(key, 0), 3)
                    rows.append(row)

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # Weakest question
                faith_col = "Faithfulness"
                if faith_col in df.columns:
                    worst_idx = df[faith_col].idxmin()
                    st.warning(f"""
**Weakest question (Faithfulness):**
{per_q[worst_idx]['question']}
Score: {df[faith_col][worst_idx]:.3f}
                    """)


if __name__ == "__main__":
    main()