import os
from dotenv import load_dotenv
import chromadb
from huggingface_hub import InferenceClient
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
import requests
import json
from chromadb.api.types import Documents, EmbeddingFunction

# Load environment variables from .env file
load_dotenv()

# setup api key 
hf_key = os.getenv("HUGGINGFACE_API_KEY")

# Ollama API endpoint - assuming it's running locally
OLLAMA_API_URL = "http://localhost:11434/api"

# Create a custom embedding function for Ollama
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        
    def __call__(self, input: Documents) -> list:
        embeddings = []
        for text in input:
            embeddings.append(self._get_ollama_embedding(text))
        return embeddings
        
    def _get_ollama_embedding(self, text):
        try:
            # Using Ollama's embedding API endpoint
            response = requests.post(
                f"{OLLAMA_API_URL}/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

# Create an instance of the custom Ollama embedding function
ollama_ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")

# 1. Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chromadb_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=ollama_ef
)

# 2. Initialize the InferenceClient/model you want to use
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_key)

# 3. Function to load documents from a directory, where i store my documents
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# 4. Function to from load documents to split text into chunks so i can embed them later , and its store in database
def split_text(text, chunk_size=1000, chunk_overlap=20): # more overlap for better contextual meaning
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# 5. Load documents from the directory after splitting them into chunks
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

#print(f"Loaded {len(documents)} documents")

### now i have a documents in text format i will split them into chunks and store them in database


# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

#print(f"Split documents into {len(chunked_documents)} chunks")

# NOW ITS TIME FOR EMBEDDING 

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = ollama_ef([doc["text"]])[0]

#print(doc["embedding"])

# now insert each embedding into chromadb

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting cchunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks --> its tell how many chunks are most relevant to the question
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")

# NOW FOR GENERATING RESPONSE FROM HUGGINGFACE OR ANY LLM

# Function to generate a response from huggingface and how its repond to the question , so basiaclly  setup format for the response
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat_completion(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about AI replacing TV writers strike."
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
