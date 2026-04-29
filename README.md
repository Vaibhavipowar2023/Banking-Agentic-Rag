# 🏦 Banking RAG Agent

Agentic AI banking assistant that answers questions from
official RBI and SEBI documents using Qdrant + Cohere + Groq.

---

## What it does

- Answers banking questions from official RBI documents
- Calculates EMI for any loan
- Compares banking schemes
- Searches web for current rates via Tavily
- Evaluates accuracy using RAGAS framework

---

## Tech Stack

| Component      | Tool                          |
|----------------|-------------------------------|
| LLM            | Llama 3.1 8B via Groq (free)  |
| Embeddings     | Cohere embed-english-v3.0     |
| Vector DB      | Qdrant (local)                |
| Agent          | LangChain ReAct               |
| Evaluation     | Manual RAGAS scoring          |
| UI             | Streamlit                     |

---

## Agent Tools

The AI agent has 5 tools it can use to answer questions:

| Tool | What it does | When agent uses it |
|------|-------------|-------------------|
| `search_banking_docs` | Searches RBI/SEBI documents in Qdrant using Cohere embeddings | For any question about RBI guidelines, KYC, credit cards, loans |
| `calculate_emi` | Calculates monthly EMI given loan amount, interest rate and tenure | When user asks about loan EMI or repayment |
| `compare_schemes` | Compares two banking products side by side | When user asks to compare FD vs RD, Home Loan vs Personal Loan etc |
| `web_search` | Searches live web via Tavily for current rates and recent updates | For current repo rate, latest RBI notifications, recent news |
| `get_rbi_circular` | Finds specific RBI circulars and master directions by topic | When user asks about a specific RBI rule or circular |

The agent automatically decides which tool to use based on the question — you do not need to tell it.

---

## Evaluation Results

| Metric            | Score | Rating |
|-------------------|-------|--------|
| Faithfulness      | 0.76  | Good   |
| Answer Relevancy  | 0.60  | Good   |
| Context Precision | 0.73  | Good   |

Evaluated using manual RAGAS scoring on 5 test questions from official RBI documents.

---

## Project Structure
banking-rag-agent/

├── ingest.py        # Cohere embeddings + Qdrant storage

├── tools.py         # 5 banking agent tools

├── agent.py         # LangChain ReAct agent

├── evaluate.py      # RAGAS evaluation

├── app.py           # 3 tab Streamlit UI

├── data/            # RBI/SEBI PDFs

├── eval_data/       # Test questions + results

└── pyproject.toml

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/banking-rag-agent.git
cd banking-rag-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Add API keys

Create `.env` file:
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

Get free keys from:
- console.groq.com
- dashboard.cohere.com
- app.tavily.com

### 5. Run ingestion
```bash
python ingest.py
```

### 6. Start the app
```bash
streamlit run app.py
```

---

## Demo Questions

**Banking Assistant tab:**
- What is KYC and what documents do I need?
- What is the Banking Ombudsman complaint process?
- Can a bank issue a credit card without my permission?
- What is Video KYC as per RBI?
- What is the current RBI repo rate?
- Compare Fixed Deposit vs Recurring Deposit

**EMI Calculator tab:**
- Enter Rs 50,00,000 loan at 8.5% for 20 years
- Monthly EMI = Rs 43,391

**Evaluation tab:**
- View RAGAS accuracy scores per question

---

## Data Sources

All documents are official government sources:

| Document | Source |
|----------|--------|
| KYC Master Direction 2016 | rbi.org.in |
| Credit Card Guidelines 2022 | rbi.org.in |
| KYC FAQ Document | rbi.org.in |
| Credit Card FAQ | rbi.org.in |
| Banking Ombudsman Scheme 2006 | rbi.org.in |
| Integrated Ombudsman Scheme 2021 | rbi.org.in |
| Ombudsman Amendments | rbi.org.in |

---

## API Keys needed

| Key | Source | Cost |
|-----|--------|------|
| GROQ_API_KEY | console.groq.com | Free |
| COHERE_API_KEY | dashboard.cohere.com | Free |
| TAVILY_API_KEY | app.tavily.com | Free |
