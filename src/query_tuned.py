from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

storage_context = StorageContext.from_defaults(
    persist_dir="storage_tuned"
)
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=5)

def search_policies(query, top_k=5):
    nodes = retriever.retrieve(query)
    docs = [n.node.text for n in nodes]
    scores = [n.score for n in nodes]
    return {
        "documents": [docs],
        "scores": scores
    }