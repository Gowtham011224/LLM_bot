import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

text_gen_model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
base_url="http://localhost:1234/v1"

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection1")
db = Chroma(
    client=persistent_client,
    collection_name="collection1",
    embedding_function=embedding_function,
)

def pdf_path(pdf, pdf_storage):
    if not os.path.exists(pdf_storage):
        os.makedirs(pdf_storage)

    file_name = pdf.name
    destination_path = os.path.join(pdf_storage, file_name)
    
    with open(destination_path, 'wb') as f:
        f.write(pdf.read())
    
    return destination_path

# Get or create a vector database
def get_vectordb(documents):

    db = Chroma.from_documents(documents, embedding_function)
    return db

# Get model response based on user query and chat history
def get_model_response(user_query, chat_history, content):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}

    Relevant data: {content} 
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Using LM Studio Local Inference Server
    llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", temperature=0.4)
    chain = prompt | llm | StrOutputParser()

    response = chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "content": content
    })
    return response

# Retrieve relevant data from vector DB
def get_relevant_data(query, db):
    if not db:
        return "No vector database available. Please upload and process a PDF first."

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([query])
    docs = db.similarity_search_by_vector(embeddings)
    return docs[0].page_content if docs else "No relevant data found."

# Main function to handle Streamlit interactions
def main():
    st.set_page_config(page_title="LLM Chatbot", page_icon="🤖")
    st.title("LLM Chatbot")

    # Initialize vectorstore in session state if it does not exist
    if "db" not in st.session_state:
        st.session_state.db = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]

    pdf_storage = "uploaded_pdfs"
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf = st.file_uploader("Upload your PDFs here and click on 'Process'", type=['pdf'])

        if st.button("Process") and pdf:
            path = pdf_path(pdf, pdf_storage)
            st.write("Processing PDF...")
            with st.spinner("Processing"):
                try:
                    loader = PyPDFLoader(path)
                    pages = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(separators=["SNIPPET"], chunk_size=1024, chunk_overlap=64)
                    documents = text_splitter.split_documents(pages)
                    st.session_state.db = get_vectordb(documents)
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    st.session_state.db = None
    
    # Display chat messages
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    
    user_query = st.chat_input("Type your message here...")
    if user_query and user_query.strip():
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        content = get_relevant_data(user_query, st.session_state.db)

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_model_response(user_query, st.session_state.chat_history, content)
            st.write(response)
            st.session_state.chat_history.append(AIMessage(content=response))

# Execute main function
if __name__ == "__main__":
    main()
