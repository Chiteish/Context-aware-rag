from langchain_core.prompts import PromptTemplate

def get_prompt_template():
    template = """
Answer the user's question using only the provided document context.

- Find the most relevant information in the context.
- Give a clear, accurate, and direct answer.
- Do not guess or invent information.
- If the answer is not found in the context, say:
  "I don't know based on the provided document."
- Keep the answer simple and easy to understand.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )