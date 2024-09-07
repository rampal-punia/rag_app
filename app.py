import streamlit as st
from streamlit.logger import get_logger
from langchain_core.output_parsers import StrOutputParser

from loader import DocumentProcessor
from configure_llm import LLMConfig, CustomPromptTemplates, ChainBuilder, LLMInvoker, ExternalAPIs, DocumentUtils

logger = get_logger(__name__)


def setup_page():
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📚 Document Q&A")


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload a PDF, DOCX, TXT, or MD file📄\n"
            "2. View the document summary\n"
            "3. Ask questions about the document💬\n"
        )
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "This app allows you to upload a document, "
            "view its summary, and ask questions about its content."
        )


def process_document(uploaded_file):
    if not uploaded_file:
        return None

    with st.spinner("Processing document... This may take a while⏳"):
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process the document
        processor = DocumentProcessor()
        processor.process_document(uploaded_file.name)

        # Remove the temporary file
        import os
        os.remove(uploaded_file.name)

    st.success("Document processed successfully!")
    return processor


def generate_summary(processor):
    # Get the first few paragraphs of the document
    search_results = processor.vector_store.similarity_search("", k=2)

    # Extract documents from the search results
    docs = [result[0]
            for result in search_results if isinstance(result, tuple)]

    text = DocumentUtils.format_docs(docs)

    with st.spinner("Generating summary..."):
        summary = ExternalAPIs.generate_summary({"inputs": text})

    return summary[0]['summary_text'] if summary else "Unable to generate summary."


def setup_qa_interface(processor):
    llm = LLMConfig.get_llm(model_name="Mixtral-8x7B")
    prompt = CustomPromptTemplates.get_doc_prompt()
    output_parser = StrOutputParser()
    vector_store = processor.vector_store.vector_store
    retriever = vector_store.as_retriever(search_type="mmr",
                                          search_kwargs={'k': 3, 'fetch_k': 4})

    qa_chain = ChainBuilder.create_qa_chain(
        retriever=retriever,
        prompt=prompt,
        llm=llm,
        output_parser=output_parser
    )

    # qa_chain = ChainBuilder.create_doc_chain(
    #     DocumentUtils.format_docs,
    #     # processor.vector_store.similarity_search,
    #     prompt,
    #     llm,
    #     StrOutputParser(),
    #     "qa_chain"
    # )

    return qa_chain


def main():
    setup_page()
    sidebar()

    uploaded_file = st.file_uploader("Upload a document", type=[
                                     "pdf", "docx", "txt", "md"])

    if uploaded_file:
        processor = process_document(uploaded_file)

        if processor:
            summary = generate_summary(processor)

            with st.expander("Document Summary", expanded=False):
                st.write(summary)

            st.markdown("---")

            qa_chain = setup_qa_interface(processor)

            st.subheader("Ask a question about the document")
            user_question = st.text_input("Enter your question here:")

            if user_question:
                with st.spinner("Generating answer..."):
                    # answer = qa_chain.invoke(user_question)
                    answer = LLMInvoker.invoke_llm(qa_chain, user_question)
                    st.write("Answer:", answer)


if __name__ == "__main__":
    main()
