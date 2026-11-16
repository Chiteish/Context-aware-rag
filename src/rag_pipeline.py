
"""Simple RAG pipeline tailored for LangChain v1.x."""

import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq


def build_rag_chain(retriever):
    """Return an object implementing conversational RAG."""

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    memory = InMemoryChatMessageHistory()

    class SimpleRAG:
        def __init__(self, retriever, llm, memory):
            self.retriever = retriever
            self.llm = llm
            self.memory = memory

        def invoke(self, inputs, config=None):
            # Extract query
            if isinstance(inputs, dict):
                query = inputs.get("input") or inputs.get("question")
            else:
                query = inputs

            docs = self.retriever.invoke(query)
            context = "\n\n".join(d.page_content for d in docs)

            # Build conversation history
            history_lines = []

            for m in self.memory.messages:
                if isinstance(m, HumanMessage):
                    history_lines.append(f"User: {m.content}")
                elif isinstance(m, AIMessage):
                    history_lines.append(f"AI: {m.content}")

            history_text = "\n".join(history_lines)

            prompt = (
                (f"Conversation history:\n{history_text}\n\n"
                 if history_text else "")
                + f"Context:\n{context}\n\n"
                + f"Question: {query}\nAnswer:"
            )

            answer = self.llm.invoke(prompt)

            # Update memory
            self.memory.add_user_message(query)
            self.memory.add_ai_message(answer)

            return {
                "answer": answer,
                "source_documents": docs
            }

    return SimpleRAG(retriever, llm, memory)
