import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from tools import all_tools
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"

BANKING_PROMPT = PromptTemplate.from_template("""You are FinSight AI — a friendly
knowledgeable Indian banking assistant.

TOOLS:
{tools}

Tool names: {tool_names}

STRICT FORMAT — follow exactly:

Question: the input question
Thought: I need to search for this information
Action: search_banking_docs
Action Input: specific search keywords only
Observation: the search result
Thought: I have the information needed to answer clearly
Final Answer: Write a SHORT answer in maximum 3-4 sentences only.
Use simple English like explaining to a common person.
Mention the RBI document in one line only.
End with one short practical tip in one sentence.
Do NOT write long paragraphs. Keep it brief and clear.

CRITICAL RULES:
- Use ONLY ONE tool call — never call any tool twice
- Action Input is plain keywords only — example: KYC documents required India
- Immediately after Observation write Final Answer — no exceptions
- Final Answer must be in your own words — do not paste raw document text
- Always sound like a helpful banking advisor explaining to a customer

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")


def get_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file."
        )

    print("Loading Groq LLM...")
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
    )

    print("Creating ReAct agent...")
    agent = create_react_agent(
        llm=llm,
        tools=all_tools,
        prompt=BANKING_PROMPT,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=2,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    print("Agent ready!")
    return agent_executor


def run_agent(agent_executor, question: str) -> dict:
    try:
        result = agent_executor.invoke({"input": question})
        answer = result.get("output", "")
        steps  = result.get("intermediate_steps", [])

        # If agent stopped without Final Answer
        # build answer from tool observations manually
        if (
            not answer or
            "Agent stopped" in answer or
            answer.strip() == "" or
            "Based on official RBI documents" in answer
        ):
            if steps:
                # Get all observations from all tool calls
                all_observations = []
                for action, observation in steps:
                    if str(observation).strip():
                        all_observations.append(
                            str(observation)[:600]
                        )

                # Use Groq directly to summarise the observations
                from langchain_groq import ChatGroq
                import os
                llm = ChatGroq(
                    model=GROQ_MODEL,
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY"),
                )
                summary_prompt = f"""You are a friendly Indian banking assistant.
                Answer this question in maximum 3-4 sentences only.
                Use very simple English. Be short and direct.
                Mention the RBI document name in one line.
                End with one short practical tip in one sentence only.

Question: {question}

Document excerpts:
{chr(10).join(all_observations)}

Your friendly answer:"""

                response = llm.invoke(summary_prompt)
                answer   = response.content.strip()
            else:
                answer = (
                    "I could not find relevant information. "
                    "Please visit rbi.org.in for official guidelines."
                )

        # Format steps for UI
        formatted_steps = []
        seen_tools = set()
        for i, (action, observation) in enumerate(steps, 1):
            key = f"{action.tool}:{action.tool_input[:50]}"
            if key not in seen_tools:
                seen_tools.add(key)
                formatted_steps.append({
                    "step":        i,
                    "tool":        action.tool,
                    "tool_input":  action.tool_input,
                    "observation": str(observation)[:400],
                })

        return {
            "answer": answer,
            "steps":  formatted_steps,
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "steps":  [],
        }

if __name__ == "__main__":
    print("Testing FinSight AI agent...")
    agent_executor = get_agent()

    result = run_agent(
        agent_executor,
        "I want a home loan of 50 lakhs for 20 years. What will my EMI be?"
    )

    print("\nAnswer:")
    print(result["answer"])
    print(f"\nTools used: {len(result['steps'])}")
    for step in result["steps"]:
        print(f"  Tool: {step['tool']}")
        print(f"  Input: {step['tool_input']}")