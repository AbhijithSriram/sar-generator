import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

def load_sar_templates(filepath='data/sar_templates.json'):
    """Load SAR narrative templates"""
    with open(filepath, 'r') as f:
        templates = json.load(f)
    return templates

def create_vector_store(templates, persist_directory='./chroma_db'):
    """Create ChromaDB vector store with SAR templates"""
    print("Creating vector store...")

    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

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

    results = vectorstore.similarity_search(query, k=3)

    print(f"Top {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. Typology: {doc.metadata['typology']}")
        print(f"     Key elements: {doc.metadata['key_elements']}")
        preview = doc.page_content.strip()[:150].replace('\n', ' ')
        print(f"     Preview: {preview}...")

    return results

def test_llm(model_name="mistral:7b"):
    """Quick test of the LLM"""
    print(f"\nTesting LLM: {model_name}")
    llm = Ollama(
        model=model_name,
        temperature=0.1,
        num_predict=200
    )

    response = llm.invoke("What is structuring in anti-money laundering? Answer in 2 sentences.")
    print(f"LLM Response: {response[:300]}")
    return llm

# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("RAG PIPELINE SETUP")
    print("=" * 60)

    # Load templates
    templates = load_sar_templates()
    print(f"Loaded {len(templates)} SAR templates")

    # Create vector store
    vectorstore = create_vector_store(templates)

    # Test retrieval with different queries
    print("\n" + "-" * 40)
    test_retrieval(vectorstore, "rapid movement of funds wire transfers foreign")
    print()
    test_retrieval(vectorstore, "cash deposits multiple branches structuring below threshold")
    print()
    test_retrieval(vectorstore, "shell company no business operations offshore")
    print()
    test_retrieval(vectorstore, "elderly customer exploitation unusual withdrawals")

    # Test LLM
    print("\n" + "-" * 40)
    llm = test_llm("mistral:7b")

    print("\n" + "=" * 60)
    print("RAG SETUP COMPLETE!")
    print("=" * 60)
    print("\nVector store saved to ./chroma_db/")
    print("Ready for SAR generation in Phase 5")
