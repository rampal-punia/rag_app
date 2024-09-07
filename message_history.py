from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_redis import RedisChatMessageHistory
import settings

# # Initialize RedisChatMessageHistory
# history = RedisChatMessageHistory(
#     session_id="user_123", redis_url=settings.REDIS_URL)


def get_redis_history(session_id: str):
    # Function to get or create a RedisChatMessageHistory instance
    return RedisChatMessageHistory(
        session_id=session_id,
        redis_url=settings.REDIS_URL,
        key_prefix="custom_prefix:",
        ttl=3600,  # Set TTL to 1 hour
        index_name="custom_index",
    )


# Add messages to the history
history = get_redis_history('user_456')
history.add_user_message("Hello, AI assistant!")
history.add_ai_message("Hello! How can I assist you today?")

# Retrieve messages
print("Chat History:")

for message in history.messages:
    print(f"{type(message).__name__}: {message.content}")
