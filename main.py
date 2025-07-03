import logging
import os
from typing import Optional
import uvicorn
from fastapi import FastAPI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe, langfuse_context
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from pydantic import BaseModel
from qdrant_client import QdrantClient

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)

REDIS_URL = os.environ["REDIS_CONN_STRING"]
app = FastAPI()

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    show_progress_bar=True,
)

# Initialize the callback handler for Langfuse
langfuse_handler = CallbackHandler(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
    trace_name="ai-response",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    check_compatibility=False
)

langfuse_client = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)

config = RailsConfig.from_path("./config")
guardrails = RunnableRails(config, input_key="user_input")

class QueryRequest(BaseModel):
    user_input: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
@observe
def embed_documents():
    try:
        collection_name = "smartphones"
        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if collection_exists:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store
        else:
            return []

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieve information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: A summary of the smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    try:
        product_db = embed_documents()
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        print(f"Error during smartphone information retrieval for model {model}: {e}")
        return f"Error during smartphone information retrieval: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
def generate_context(llm_tools):
    """
    Process tool calls from the language model and collect their responses.

    :param
        llm_with_tools: The language model instance with bound tools.

    :returns
        Toolresponse
    """

    # Process each tool call based on its name
    for tool_call in llm_tools.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_response = smartphone_info_tool.invoke(tool_call).content
            return tool_response
    return ""

# ---------------------------
# Main Conversation Loop
# ---------------------------

@app.post("/ask")
@observe(name="ai-response")
def main(request: QueryRequest):
    uid = request.user_id
    sid = request.session_id
    langfuse_context.update_current_trace(
        session_id=sid,
        user_id=uid
    )
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    def get_redis_history(session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=120)

    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=800,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    langfuse_context_prompt = langfuse_client.get_prompt("context-prompt", label="production")
    langchain_context_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_context_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            langfuse_context_prompt.get_langchain_prompt()[1]
        ]
    )

    langchain_context_prompt.metadata = {"langfuse_prompt": langfuse_context_prompt}

    context_chain = langchain_context_prompt | trimmer | llm_with_tools | generate_context
    context_chain_with_history = RunnableWithMessageHistory(
        context_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    context_chain_with_history_and_rails = guardrails | context_chain_with_history

    langfuse_review_prompt = langfuse_client.get_prompt("review-prompt")
    langchain_review_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_review_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            langfuse_review_prompt.get_langchain_prompt()[1]
        ]
    )

    langchain_review_prompt.metadata = {"langfuse_prompt": langfuse_review_prompt}

    review_chain = langchain_review_prompt | llm
    review_chain_with_history = RunnableWithMessageHistory(
        review_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    try:
        while True:
            context = context_chain_with_history_and_rails.invoke(
                {"user_input": request.user_input},
                config={
                    "configurable": {"session_id": uid},
                    "callbacks": [langfuse_handler], "run_name": "context"
                }
            )

            context_result = context.get("output") if isinstance(context, dict) else context
            if context_result and context_result.strip().lower() == "i'm sorry, i can't respond to that.":
                return context_result
            else:
                final_response = review_chain_with_history.invoke(
                    {"user_input": request.user_input, "user_id": uid, "context": context},
                    config={
                        "configurable": {"session_id": uid},
                        "callbacks": [langfuse_handler], "run_name": "final_response"
                    }
                )
                return final_response.get("output") if isinstance(final_response, dict) else final_response

    except Exception as e:
        if hasattr(e, "message") and "Budget has been exceeded" in e.message:
            return "Your budget has been exceeded. Please for 10 minutes before trying again."
        return f"An unexpected error occurred in the main loop: {e}"



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
