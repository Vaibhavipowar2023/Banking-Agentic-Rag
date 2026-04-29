"""
5 banking specific tools for the FinSight AI agent:

1. search_banking_docs   - searches Qdrant with Cohere embeddings
2. calculate_emi         - calculates EMI for any loan
3. compare_schemes       - compares two banking products
4. web_search            - searches web for current rates via Tavily
5. get_rbi_circular      - finds specific RBI circular by topic
"""
import os
import math
from functools import lru_cache
from langchain.tools import tool
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
QDRANT_PATH = "qdrant_storage"
COLLECTION_NAME = "banking_docs"
COHERE_MODEL =  "embed-english-v3.0"

@lru_cache(maxsize=1)
def get_vectorstore():
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not found.")

    print("Loading Qdrant and Cohere embeddings...")
    embeddings = CohereEmbeddings(
        model=COHERE_MODEL,
        cohere_api_key=cohere_key,
    )

    # Check if Qdrant storage exists
    if not os.path.exists(QDRANT_PATH):
        raise FileNotFoundError(
            f"Qdrant storage not found at '{QDRANT_PATH}'. "
            "Please run python ingest.py first."
        )

    client = QdrantClient(path=QDRANT_PATH)

    # Verify collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"  Available collections: {collection_names}")

    if COLLECTION_NAME not in collection_names:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found. "
            f"Available: {collection_names}. "
            "Please run python ingest.py again."
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    print(f"  Vectorstore loaded successfully.")
    return vectorstore

# Tool 1 : Search Banking Documents
@tool
def search_banking_docs(query : str) -> str :
    """
       Search RBI and SEBI banking documents for information
       about KYC guidelines, credit card rules, loan policies,
       Banking scheme, and other RBI regulations.
       Use this for any question about Indian banking guidelines.
       Returns relevant excerpts with document and page citations.
       """
    try :
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k" : 5},
        )
        docs = retriever.invoke(query)

        if not docs :
            return "No relevant information found in banking documents"

        results = []
        for i, doc in enumerate(docs, 1):
            filename = (
                    doc.metadata.get("filename") or
                    doc.metadata.get("source") or
                    "Unknown"
            )
            if filename != "Unknown":
                filename = os.path.basename(filename)
            page = doc.metadata.get("page", "?")
            results.append(
                f"[Source {i}: {filename} | Page {page}]\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Error searching documents: {str(e)}"

# Tool 2 : EMI Calculator
@tool
def calculate_emi(loan_details: str) -> str:
    """
    Calculate EMI for any loan given the loan amount,
    interest rate and tenure.
    Input format: 'amount=500000, rate=8.5, tenure=20'
    amount is in rupees, rate is annual percentage,
    tenure is in years.
    Example: 'amount=5000000, rate=8.5, tenure=20'
    for a 50 lakh home loan at 8.5% for 20 years.
    """
    try:
        # Parse input
        params = {}
        for part in loan_details.split(","):
            if "=" in part:
                key, value = part.strip().split("=")
                params[key.strip()] = float(value.strip())

        amount  = params.get("amount", 0)
        rate    = params.get("rate", 0)
        tenure  = params.get("tenure", 0)

        if not all([amount, rate, tenure]):
            return (
                "Please provide amount, rate and tenure. "
                "Example: amount=500000, rate=8.5, tenure=20"
            )

        # EMI formula
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        # P = principal amount
        # r = monthly interest rate
        # n = number of months
        monthly_rate = rate / (12 * 100)
        months       = int(tenure * 12)
        emi          = (
            amount * monthly_rate *
            math.pow(1 + monthly_rate, months)
        ) / (math.pow(1 + monthly_rate, months) - 1)

        total_payment  = emi * months
        total_interest = total_payment - amount

        return f"""EMI Calculation Results:
        Loan Amount:      Rs {amount:,.0f}
        Interest Rate:    {rate}% per annum
        Tenure:           {tenure} years ({months} months)
        Monthly EMI:      Rs {emi:,.0f}
        Total Payment:    Rs {total_payment:,.0f}
        Total Interest:   Rs {total_interest:,.0f}
        Interest Burden:  {(total_interest/amount*100):.1f}% of principal
        
        Note: This is an approximate calculation.
        Actual EMI may vary based on bank processing
        fees and exact interest calculation method."""

    except Exception as e:
        return f"EMI calculation error: {str(e)}"


# Tool 3 : Compare Banking Schemes
@tool
def compare_schemes(schemes : str) -> str :
    """
        Compare two banking products or schemes side by side.
        Use this when user wants to compare two things like
        FD vs RD, Home Loan vs Personal Loan, KYC types etc.
        Input: two scheme names separated by 'vs'
        Example: 'Fixed Deposit vs Recurring Deposit'
        or 'Home Loan vs Personal Loan'
        or 'Video KYC vs Physical KYC'
        """
    try :
        if "vs" not in schemes.lower() :
            return  (
                "Please provide two schemes separated  by 'vs'."
                "Example: 'Fixed Deposit vs Recurring Deposit'"
            )

        parts = schemes.lower().split("vs")
        scheme_a = parts[0].strip()
        scheme_b = parts[1].strip()

        vectorstore = get_vectorstore()

        # Search for both schemes
        docs_a = vectorstore.similarity_search(scheme_a, k = 3)
        docs_b = vectorstore.similarity_search(scheme_b, k = 3)

        result_a = "\n\n".join([
        f"[{doc.metadata.get('filename', '?')} "
        f"p.{doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content.strip()[:300]}"
        for doc in docs_a
        ])

        result_b = "\n\n".join([
        f"[{doc.metadata.get('filename', '?')} "
        f"p.{doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content.strip()[:300]}"
        for doc in docs_b
        ])

        return f"""
        
        SCHEME A — {scheme_a.upper()}:
        {result_a}
    
        {'=' * 50}
    
        SCHEME B — {scheme_b.upper()}:
        {result_b}
    """

    except Exception as e :
        return f"Error comparing schemes: {str(e)}"


# ── Tool 4: Web Search ────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the web for current banking information,
    latest RBI notifications, current interest rates,
    or recent changes in banking regulations.
    Use this when the user asks about current rates
    or recent updates not available in documents.
    Example queries:
    'current home loan interest rates India 2026'
    'latest RBI repo rate 2026'
    'recent RBI circular on credit cards'
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "TAVILY_API_KEY not found. Add it to .env file."

        client  = TavilyClient(api_key=api_key)
        results = client.search(
            query=f"India banking RBI {query}",
            max_results=3,
            search_depth="basic",
        )

        if not results or "results" not in results:
            return "No web results found."

        formatted = []
        for i, r in enumerate(results["results"], 1):
            formatted.append(
                f"[Web Source {i}: {r.get('title', 'Unknown')}]\n"
                f"URL: {r.get('url', '')}\n"
                f"{r.get('content', '')[:400]}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Web search error: {str(e)}"


# ── Tool 5: Get RBI Circular ──────────────────────────────────
@tool
def get_rbi_circular(topic: str) -> str:
    """
    Find specific RBI circulars, master directions,
    or guidelines by topic from the banking documents.
    Use this when user asks about a specific RBI rule,
    circular number, or master direction.
    Example topics:
    'KYC updation rules'
    'credit card billing statement'
    'banking ombudsman complaint process'
    'minimum balance charges'
    """
    try:
        vectorstore = get_vectorstore()

        # Search with banking specific query
        search_query = f"RBI circular guideline {topic}"
        docs = vectorstore.similarity_search(
            search_query,
            k=5,
        )

        if not docs:
            return f"No RBI circular found for topic: {topic}"

        results = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "Unknown")
            page     = doc.metadata.get("page", "?")
            results.append(
                f"[RBI Document {i}: {filename} | Page {page}]\n"
                f"{doc.page_content.strip()}"
            )

        return (
            f"RBI Guidelines on '{topic}':\n\n" +
            "\n\n---\n\n".join(results)
        )

    except Exception as e:
        return f"Error finding RBI circular: {str(e)}"


# List of all tools
all_tools = [
    search_banking_docs,
    calculate_emi,
    compare_schemes,
    web_search,
    get_rbi_circular,
]

if __name__ == "__main__" :
    print("Testing EMI calculator...")
    result = calculate_emi.invoke(
        "amount=5000000, rate=8.5, tenure=20"
    )
    print(result)
