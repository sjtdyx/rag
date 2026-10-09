### COMPLETE THE CODE  ###

from policy_loader import load_policy_documents
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


# Create the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Create a Chroma vector store
vector_store = Chroma(
    collection_name="university_support",
    embedding_function=embeddings
)


# Add the provided policy documents one at a time
for i, document in enumerate(documents, start=1):
    print(f"Embedding document {i}/{len(documents)}")

    vector_store.add_documents([document])


# Retrieve relevant documents
def retrieve_documents(question, k=3):

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results