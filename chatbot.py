from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY


# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


def ask_llm(context, question):
    """
    Sends the retrieved context and user question to Gemini.
    """

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context provided below.

If the answer is not present in the context, say:
"I couldn't find that information in the provided document."

========================
Context:
{context}
========================

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content