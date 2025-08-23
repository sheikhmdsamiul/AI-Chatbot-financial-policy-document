from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration for embeddings and Chroma vector store
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHROMA_PATH = "chroma-db"

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )


# Function to split document into semantic chunks
# This function uses the SemanticChunker to create chunks based on semantic meaning
def semantic_text_splitter(document):
    """Split document into semantic chunks.
    Args:
        document (list): List of LangChain Document objects.
    Returns:
        list: List of chunked Document objects.
    """
    

    if not document:
        return []
    
    # Create a SemanticChunker instance with the embeddings
    # Using percentile with 85% threshold for balanced chunking
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
        min_chunk_size=300
    )

    docs = text_splitter.split_documents(document)

    # Debug: Print the chunks to verify
    print("________Semantic Chunks:__________/n")
    
    # Print each chunk
    for doc in docs:
        print(doc)

    return docs



def create_vector_store(documents):
    """Create a Chroma vector store from the documents.
    Args:
        documents (list): List of LangChain Document objects.
    Returns:
        Chroma: The Chroma vector store.
    """

    # Split documents into semantic chunks
    chunks = semantic_text_splitter(documents)

     # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    
    return db