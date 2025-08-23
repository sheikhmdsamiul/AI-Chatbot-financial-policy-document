import re
from langchain.schema import Document
import streamlit as st
from agentic_doc.parse import parse

from modules.vector_store import create_vector_store

# Path to the PDF file
pdf_path = "context_pdf/Policy file.pdf"

# Parse the PDF file using agentic-doc
results = parse(pdf_path)


# Function to convert parser output to LangChain Documents
# Each document corresponds to a page in the PDF
def parse_output_to_documents(parsed_output):
    """Convert parser output to LangChain Documents.
    Args:
        parsed_output (list): The output from the parser.
    Returns:
        list: A list of LangChain Document objects.
    """
    # Dictionary to store content by page
    page_content = {}
    
    # Regex to capture content + metadata
    # Example match: "Some text...<!-- paragraph, from page 2 (l=72.0,t=144.0,r=540.0,b=216.0), with ID 123e4567-e89b-12d3-a456-426614174000 -->"
    pattern = re.compile(
        r"(.*?)<!--\s*(?P<type>\w+), from page (?P<page>\d+) "
        r"\(l=(?P<l>[0-9.]+),t=(?P<t>[0-9.]+),r=(?P<r>[0-9.]+),b=(?P<b>[0-9.]+)\), "
        r"with ID (?P<id>[a-f0-9\-]+)\s*-->",
        re.DOTALL
    )
    
    # Extract content and metadata
    # Iterate through parsed output and populate page_content
    for result in parsed_output:
        for match in pattern.finditer(result.markdown):
            content = match.group(1).strip()
            page_num = int(match.group("page"))
            doc_type = match.group("type").lower()
            
            # Initialize page if not exists
            if page_num not in page_content:
                page_content[page_num] = []
            
            # Add content to the page
            if content:  # Only add non-empty content
                page_content[page_num].append(content)
    
    # Create documents
    documents = []
    for page_num, contents in page_content.items():
        if contents:  # Only create document if page has content
            # Combine all content from the page
            combined_content = "\n\n".join(contents)
            
            # Create metadata
            metadata = {
                "page": page_num,
                "source": "Policy file.pdf"
            }
            
            # Create Document object
            # Append to documents list
            documents.append(Document(page_content=combined_content, metadata=metadata))
    
    return documents

documents = parse_output_to_documents(results)

# Debug: Print the documents to verify
print("________Lanchain Documents:__________/n")
print(documents)

# Create the vector store from the documents through semantic chunking
vectorstore = create_vector_store(documents)