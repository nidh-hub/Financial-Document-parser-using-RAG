# New data pipeline module (regulatory_data_manager.py)
import requests
import os
import json
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Changed to Chroma for better local management

class RegulatoryDataManager:
    def __init__(self, storage_path="regulatory_data"):
        self.storage_path = storage_path
        self.raw_data_dir = os.path.join(storage_path, "raw")
        self.processed_dir = os.path.join(storage_path, "processed")
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def fetch_from_api(self, api_endpoint="https://api.regulatory.gov/v2/reports"):
        """Fetch latest regulatory reports via API"""
        try:
            response = requests.get(api_endpoint)
            response.raise_for_status()
            
            # Save raw data with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(self.raw_data_dir, f"report_{timestamp}.json")
            with open(raw_path, 'w') as f:
                json.dump(response.json(), f)
                
            return self._process_data(raw_path)
            
        except Exception as e:
            print(f"API fetch error: {str(e)}")
            return False

    def _process_data(self, raw_file_path):
        """Convert raw reports to RAG-ready chunks"""
        try:
            with open(raw_file_path) as f:
                data = json.load(f)
                
            # Convert JSON to text documents
            documents = [f"{item['title']}\n{item['content']}" for item in data['reports']]
            
            # Save processed text
            processed_path = os.path.join(self.processed_dir, 
                                        os.path.basename(raw_file_path).replace('.json', '.txt'))
            with open(processed_path, 'w') as f:
                f.write("\n\n".join(documents))
                
            return True
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return False

    def get_rag_documents(self):
        """Load processed documents for RAG system"""
        loader = DirectoryLoader(self.processed_dir, glob="**/*.txt")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)
