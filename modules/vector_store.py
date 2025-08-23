import os
import shutil
#from streamlit import st
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHROMA_PATH = "chroma-db"

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )


# Function to split document into semantic chunks
# This function uses the SemanticChunker to create chunks based on semantic meaning
def semantic_text_splitter(document):
    """Split document into semantic chunks

    Args:   
         doc: The documetn to be split into chunks.

    Returns:        
         list: A list of semantic chunks."""
    

    if not document:
        return []
    
    # Create a SemanticChunker instance with the embeddings
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile" , min_chunk_size=300)

    docs = text_splitter.split_documents(document)

    return docs



def create_vector_store(documents):

    chunks = semantic_text_splitter(documents)
  
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

     # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    
    return db