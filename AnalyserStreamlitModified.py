from regulatory_data_manager import RegulatoryDataManager
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import schedule
import threading
import time

# Initialize components
data_manager = RegulatoryDataManager()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Persistent Chroma DB
vector_store = Chroma(
    persist_directory="regulatory_db",
    embedding_function=embeddings
)

def update_vector_store():
    """Refresh vector store with latest regulations"""
    try:
        new_docs = data_manager.get_rag_documents()
        if new_docs:
            vector_store.add_documents(new_docs)
            vector_store.persist()
            print("Vector store updated successfully")
    except Exception as e:
        print(f"Update error: {str(e)}")

# Automatic updates (every 6 hours)
schedule.every(6).hours.do(update_vector_store)

# Start scheduler in background
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()

# Modified Streamlit UI
def main():
    st.title("Regulatory Compliance Analyzer")
    
    # Manual update trigger
    if st.button("⟳ Update Regulations Now"):
        with st.spinner("Fetching latest regulations..."):
            if data_manager.fetch_from_api():
                update_vector_store()
                st.success("Regulations updated!")
            else:
                st.error("Update failed")

    # Existing analysis UI
    file_input = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    query_input = st.text_input("Your Compliance Question")
    
    if st.button("Analyze"):
        if not query_input:
            st.error("Please enter a compliance question")
        else:
            # Modified retrieval
            retriever = vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': 6, 'fetch_k': 20}
            )
            
            # Rest of your existing chain
            prompt = ChatPromptTemplate.from_template("""
            Analyze compliance with latest regulations:
            {context}
            
            Query: {input}
            
            Format findings as:
            - [REGULATION] [COMPLIANCE STATUS]: [RATIONALE]""")
            
            chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
            )
            
            result = chain.invoke(query_input)
            st.subheader("Compliance Analysis")
            st.markdown(f"``````")

if __name__ == "__main__":
    main()
