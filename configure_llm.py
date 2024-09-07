from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
import requests

import settings


class LLMConfig:
    HUGGINGFACE_MODELS = {
        "Mistral-7B": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        "Mixtral-8x7B": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mixtral-8x22B": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Mistral-Nemo": "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407",
    }

    @classmethod
    def configure_llm(cls, model_name, **kwargs):
        if model_name not in cls.HUGGINGFACE_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        return HuggingFaceEndpoint(
            endpoint_url=cls.HUGGINGFACE_MODELS[model_name],
            task='text-generation',
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
            **kwargs
        )

    @staticmethod
    def get_llm(model_name="Mixtral-8x7B", temperature=0.04, tokens=1024, top_k=20, top_p=0.85,
                typical_p=0.95, repetition_penalty=1.03, is_streaming=True):
        return LLMConfig.configure_llm(
            model_name=model_name,
            max_new_tokens=tokens,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            streaming=is_streaming,
        )


class CustomPromptTemplates:
    @staticmethod
    def get_chat_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ('human',
             "Conversation history:\n{history}\n\nNew User message: {input}"),
            ("human", "Now, respond to the new message.")
        ])

    @staticmethod
    def get_summarizer_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful summarizer."),
            ("human", "Now, summarize these given paragraphs: {input}.")
        ])

    @staticmethod
    def get_doc_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant helping to answer questions based on a given document. Use the following context to answer the user's question. If you cannot answer the question based on the context, say that you don't have enough information to answer accurately."),
            ('human',
             "Related Context:\n{context}\n\nNew User message: {input}"),
            ("human", "Now, respond to the new message.")
        ])

    @staticmethod
    def get_doc_prompt_with_history():
        return ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant helping to answer, in short (150 to 200 words only), the questions based on a given document. Use the following context to answer the user's question. If you cannot answer the question based on the context, say that you don't have enough information to answer accurately."),
            ('human',
             "Conversation history:\n{history}\n\nNew User message: {input}"),
            ('human', "Related Context:\n{context}"),
            ("human", "Now, respond to the new message.")
        ])


class ChainBuilder:
    @staticmethod
    def create_chat_chain(prompt, llm, output_parser, run_name):
        return prompt | llm.with_config({'run_name': 'model'}) | output_parser.with_config({'run_name': run_name})

    @staticmethod
    def create_qa_chain(retriever, prompt, llm, output_parser):
        return (
            {
                "context": retriever,
                # "context": retriever | DocumentUtils.format_docs,
                "input": RunnablePassthrough(),
            }
            | prompt
            | llm.with_config({'run_name': 'model'})
            | output_parser
        )

    @staticmethod
    def create_doc_chain(retrieved_docs, prompt, llm, output_parser, run_name):
        chain = (
            RunnablePassthrough.assign(context=retrieved_docs)
            | prompt
            | llm.with_config({'run_name': 'model'})
            | output_parser.with_config({'run_name': run_name})
        )
        return RunnableWithMessageHistory(
            chain,
            RedisChatMessageHistory,
            input_messages_key="question",
            history_messages_key="chat_history",
        )


class LLMInvoker:
    @staticmethod
    def invoke_llm(memory_chain, user_question: str = 'What is modern science', session_id='123456789'):
        return memory_chain.invoke(user_question)
        # return memory_chain.invoke(
        #     {"question": user_question},
        #     config={"configurable": {"session_id": session_id}},
        # )


class ExternalAPIs:
    @staticmethod
    def generate_summary(payload):
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    @staticmethod
    def generate_title(payload):
        API_URL = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
        headers = {"Authorization": f"Bearer {settings.HUGGINGFACEHUB_API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()


class DocumentUtils:
    @staticmethod
    def get_sources(docs):
        return [", ".join([doc.metadata["source"] for doc in docs])]

    @staticmethod
    def format_docs(docs):
        # DocumentUtils.get_sources(docs)
        return "\n\n".join([doc.page_content for doc in docs])


if __name__ == '__main__':
    # Example usage
    from loader import DocumentProcessor

    # Initialize components
    output_parser = StrOutputParser()
    llm = LLMConfig.get_llm(model_name="Mixtral-8x7B")
    prompt = CustomPromptTemplates.get_doc_prompt()
    vector_store = DocumentProcessor().vector_store.vector_store
    retriever = vector_store.as_retriever(search_type="mmr",
                                          search_kwargs={'k': 3, 'fetch_k': 4})

    # print(retriever.invoke(
    #     "How much money college received in academic year 2022-2023"))
    # Create QA chain
    qa_chain = ChainBuilder.create_qa_chain(
        retriever=retriever,
        prompt=prompt,
        llm=llm,
        output_parser=output_parser
    )
    # # Create memory chain
    # memory_chain = RunnableWithMessageHistory(
    #     qa_chain,
    #     RedisChatMessageHistory,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    # )

    # Example query
    # print(qa_chain)
    # result = qa_chain.invoke(
    #     "How much money college received in academic year 2022-2023")
    # print(result)
    user_question = "How much money college received in academic year 2022-2023"
    answer = LLMInvoker.invoke_llm(qa_chain, user_question=user_question)
    print(answer)

    # # Example query
    # result = memory_chain.invoke(
    #     "How much money college received in academic year 2022-2023")
    # print(result)
