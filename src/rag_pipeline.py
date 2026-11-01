"""Simple RAG pipeline tailored for LangChain v1.x."""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


def build_rag_chain(retriever):
    """Return an object with an `invoke` method implementing conversational RAG.

    The old helper classes like :class:`ConversationalRetrievalChain` were
    removed in LangChain 1.0+.  This function uses only core primitives so it
    works with the current 1.2.8 installation.

    Args:
        retriever: a LangChain retriever (e.g. FAISS.as_retriever()).
    """

    llm = ChatOllama(model="mistral")
    memory = InMemoryChatMessageHistory()

    class SimpleRAG:
        def __init__(self, retriever, llm, memory):
            self.retriever = retriever
            self.llm = llm
            self.memory = memory

        def invoke(self, inputs, config=None):
            # extract query
            if isinstance(inputs, dict):
                query = inputs.get("input") or inputs.get("question")
            else:
                query = inputs

            docs = self.retriever.invoke(query)
            context = "\n\n".join(d.page_content for d in docs)

            # build conversation history text
            history_lines = []
            for m in self.memory.messages:
                if isinstance(m, HumanMessage):
                    history_lines.append(f"User: {m.content}")
                elif isinstance(m, AIMessage):
                    history_lines.append(f"AI: {m.content}")
            history_text = "\n".join(history_lines)

            prompt = (
                (f"Conversation history:\n{history_text}\n\n" if history_text else "") +
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )

            answer = self.llm.invoke(prompt)

            # update memory
            self.memory.add_user_message(query)
            self.memory.add_ai_message(answer)

            return {"answer": answer, "source_documents": docs}

    return SimpleRAG(retriever, llm, memory)