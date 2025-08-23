import re
from langchain.schema import Document
import streamlit as st
from agentic_doc.parse import parse

pdf_path = "Policy file.pdf"
results = parse(pdf_path)




  
def parse_output_to_documents(parsed_output):
    """
    Convert parser output (with HTML + metadata comments) into LangChain Documents.
    Tables remain stored as raw HTML strings.
    """
    documents = []

    # Regex to capture content + metadata
    pattern = re.compile(
        r"(.*?)<!--\s*(?P<type>\w+), from page (?P<page>\d+) "
        r"\(l=(?P<l>[0-9.]+),t=(?P<t>[0-9.]+),r=(?P<r>[0-9.]+),b=(?P<b>[0-9.]+)\), "
        r"with ID (?P<id>[a-f0-9\-]+)\s*-->",
        re.DOTALL
    )
    for result in parsed_output:
        for match in pattern.finditer(result.markdown):
            content = match.group(1).strip()
            doc_type = match.group("type").lower()
            metadata = {
                "page": int(match.group("page")),
                "bbox": (
                    float(match.group("l")),
                    float(match.group("t")),
                    float(match.group("r")),
                    float(match.group("b"))
                ),
                "id": match.group("id"),
                "type": doc_type
            }

            # Store raw HTML for tables, plain text otherwise
            documents.append(Document(page_content=content, metadata=metadata))

    return documents

st.write(parse_output_to_documents(results))


