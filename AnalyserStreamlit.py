from langchain_community.llms import Ollama 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import tempfile

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(
    model="llama2",
    base_url="http://localhost:11434",
    temperature=0.3
)

# Load ESG base documents
def load_esg_base():
    loader = PyPDFLoader("C:/Users/chakr/Downloads/esg_regulations.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)

vector_store = FAISS.from_documents(load_esg_base(), embeddings)

def analyze_content(query, file):
    try:
        if file:
            # Streamlit's file uploader returns an in-memory file-like object.
            # Write its contents to a temporary file so that PyPDFLoader can access it.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read()) 
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            user_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            user_chunks = splitter.split_documents(user_docs)
            
            user_store = FAISS.from_documents(user_chunks, embeddings)
            user_store.merge_from(vector_store)
            retriever = user_store.as_retriever()
        else:
            retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_template(
            """Analyze this ESG document for compliance risks:
{context}

Query: {input}

Format findings as:
- [RISK LEVEL] [SECTION]: [DESCRIPTION]"""
        )

        chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt
            | llm
        )

        return chain.invoke(query)

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("ESG Document Analyzer with Local LLM")

    file_input = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    query_input = st.text_input("Your Compliance Question")

    if st.button("Analyze"):
        if not query_input:
            st.error("Please enter a compliance question.")
        else:
            result = analyze_content(query_input, file_input)
            st.subheader("Analysis Results")
            st.text_area("", result, height=300)

if __name__ == "__main__":
    main()
