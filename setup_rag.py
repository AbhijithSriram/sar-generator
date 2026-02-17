import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_sar_templates(filepath='data/sar_templates.json'):
    """Load SAR narrative templates"""
    with open(filepath, 'r') as f:
        templates = json.load(f)
    return templates

def create_vector_store(templates, persist_directory='./chroma_db'):
    """Create ChromaDB vector store with SAR templates"""
    print("Creating cloud-compatible vector store...")

    # Initialize HuggingFace embeddings (Replaced Ollama)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Prepare documents
    documents = []
    metadatas = []

    for i, template in enumerate(templates):
        documents.append(template['narrative'])
        metadatas.append({
            'typology': template['typology'],
            'template_id': str(i),
            'key_elements': ', '.join(template['key_elements'])
        })

    # Create ChromaDB collection
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )

    print(f"[OK] Vector store created with {len(documents)} templates")
    return vectorstore

def test_retrieval(vectorstore, query):
    """Test vector retrieval"""
    print(f"\nTesting retrieval with query: '{query}'")
    results = vectorstore.similarity_search(query, k=2)

    for i, doc in enumerate(results):
        print(f"  {i+1}. Typology: {doc.metadata['typology']}")
        preview = doc.page_content.strip()[:100].replace('\n', ' ')
        print(f"     Preview: {preview}...")
    return results

# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("RAG PIPELINE SETUP (CLOUD COMPATIBLE)")
    print("=" * 60)

    # Load templates
    templates = load_sar_templates()
    
    # Create fresh vector store
    vectorstore = create_vector_store(templates)

    # Test retrieval
    test_retrieval(vectorstore, "cash deposits structuring below threshold")
    test_retrieval(vectorstore, "elderly customer exploitation unusual withdrawals")

    print("\n" + "=" * 60)
    print("RAG SETUP COMPLETE!")
    print(f"Vector store saved to: {os.path.abspath('./chroma_db/')}")