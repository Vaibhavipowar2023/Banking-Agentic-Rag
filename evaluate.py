"""
evaluate.py
-----------
Evaluates Banking RAG Agent using RAGAS framework.
Uses sequential processing to avoid rate limit timeouts.
"""

import os
import json
import time
import math
import pandas as pd
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from tools import get_vectorstore
from dotenv import load_dotenv

load_dotenv()

# Configuration
GROQ_MODEL        = "llama-3.1-8b-instant"
COHERE_MODEL      = "embed-english-v3.0"
TEST_QUESTIONS    = "eval_data/test_questions.json"
EVAL_RESULTS_FILE = "eval_data/evaluation_results.json"
NUM_QUESTIONS     = 5


def load_test_questions() -> list:
    with open(TEST_QUESTIONS, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} test questions.")
    return questions[:NUM_QUESTIONS]


def get_answer_and_context(question, vectorstore, llm) -> dict:
    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        docs     = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        context_text = "\n\n".join(contexts)
        prompt = f"""You are a helpful Indian banking assistant.
Answer the question based strictly on the context below.
If not found say: I could not find this in the documents.

Context:
{context_text}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        return {
            "question": question,
            "answer":   response.content.strip(),
            "contexts": contexts,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {
            "question": question,
            "answer":   "Error generating answer.",
            "contexts": [],
        }


def score_faithfulness(question, answer, contexts, llm) -> float:
    """
    Manual faithfulness scoring.
    Asks LLM: is each claim in the answer supported by context?
    """
    try:
        context_text = "\n\n".join(contexts[:3])
        prompt = f"""You are an evaluator checking if an answer is faithful to the context.

Context:
{context_text}

Answer to evaluate:
{answer}

Question:
{question}

Instructions:
- Check if every claim in the answer is supported by the context
- Score from 0.0 to 1.0
- 1.0 = completely faithful, all claims in context
- 0.5 = partially faithful, some claims not in context
- 0.0 = not faithful, answer contradicts or ignores context

Return ONLY a number between 0.0 and 1.0. Nothing else."""

        time.sleep(2)  # avoid rate limits
        response = llm.invoke(prompt)
        score    = float(response.content.strip())
        return min(max(score, 0.0), 1.0)

    except Exception as e:
        print(f"  Faithfulness scoring error: {e}")
        return 0.0


def score_relevancy(question, answer, llm) -> float:
    """
    Manual answer relevancy scoring.
    Asks LLM: does the answer address the question?
    """
    try:
        prompt = f"""You are an evaluator checking if an answer is relevant to the question.

Question: {question}

Answer: {answer}

Instructions:
- Check if the answer directly addresses what was asked
- Score from 0.0 to 1.0
- 1.0 = perfectly relevant
- 0.5 = partially relevant
- 0.0 = completely irrelevant

Return ONLY a number between 0.0 and 1.0. Nothing else."""

        time.sleep(2)
        response = llm.invoke(prompt)
        score    = float(response.content.strip())
        return min(max(score, 0.0), 1.0)

    except Exception as e:
        print(f"  Relevancy scoring error: {e}")
        return 0.0


def score_context_precision(question, contexts, ground_truth, llm) -> float:
    """
    Manual context precision scoring.
    What fraction of retrieved chunks are relevant?
    """
    try:
        relevant = 0
        for ctx in contexts:
            prompt = f"""Is this context chunk relevant to answering the question?

Question: {question}
Context chunk: {ctx[:300]}

Return ONLY 'yes' or 'no'."""

            time.sleep(1)
            response = llm.invoke(prompt)
            if "yes" in response.content.strip().lower():
                relevant += 1

        return relevant / len(contexts) if contexts else 0.0

    except Exception as e:
        print(f"  Precision scoring error: {e}")
        return 0.0


def run_evaluation():
    print("=" * 55)
    print("  Banking RAG Agent — RAGAS Evaluation")
    print("=" * 55)

    test_data = load_test_questions()

    print("\nLoading models...")
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    vectorstore = get_vectorstore()

    # Run RAG for each question
    print(f"\nRunning RAG for {len(test_data)} questions...")
    results = []

    for i, item in enumerate(test_data, 1):
        print(f"\n  Question {i}/{len(test_data)}: "
              f"{item['question'][:50]}...")

        # Get answer and context
        rag_result = get_answer_and_context(
            item["question"], vectorstore, llm
        )

        # Score each metric manually
        print("    Scoring faithfulness...")
        faith = score_faithfulness(
            item["question"],
            rag_result["answer"],
            rag_result["contexts"],
            llm,
        )

        print("    Scoring relevancy...")
        relev = score_relevancy(
            item["question"],
            rag_result["answer"],
            llm,
        )

        print("    Scoring context precision...")
        prec = score_context_precision(
            item["question"],
            rag_result["contexts"],
            item["ground_truth"],
            llm,
        )

        results.append({
            "question":          item["question"],
            "answer":            rag_result["answer"],
            "ground_truth":      item["ground_truth"],
            "faithfulness":      round(faith, 4),
            "answer_relevancy":  round(relev, 4),
            "context_precision": round(prec, 4),
        })

        print(f"    Faith: {faith:.2f} | "
              f"Relevancy: {relev:.2f} | "
              f"Precision: {prec:.2f}")

        time.sleep(2)  # pause between questions

    # Calculate overall scores
    print("\n" + "=" * 55)
    print("  Evaluation Results")
    print("=" * 55)

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    overall = {}

    print("\nOverall Scores:")
    for metric in metrics:
        scores = [r[metric] for r in results]
        avg    = sum(scores) / len(scores)
        overall[metric] = round(avg, 4)
        rating = (
            "Excellent" if avg >= 0.8
            else "Good" if avg >= 0.6
            else "Needs improvement"
        )
        print(f"  {metric:<25} {avg:.4f}  {rating}")

    print("\nPer Question Scores:")
    for i, r in enumerate(results, 1):
        print(f"\n  Q{i}: {r['question'][:60]}...")
        for metric in metrics:
            print(f"    {metric:<25} {r[metric]:.4f}")

    # Find weakest
    weakest = min(results, key=lambda x: x["faithfulness"])
    print(f"\nWeakest question (faithfulness):")
    print(f"  {weakest['question'][:70]}")
    print(f"  Score: {weakest['faithfulness']:.4f}")

    # Save results
    eval_output = {
        "overall_scores": overall,
        "per_question":   results,
    }

    os.makedirs("eval_data", exist_ok=True)
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\nResults saved to {EVAL_RESULTS_FILE}")
    print("=" * 55)
    return eval_output


if __name__ == "__main__":
    run_evaluation()