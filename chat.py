# simple chatbot using huggingface api and chromadb
import os
from dotenv import load_dotenv
import chromadb
from huggingface_hub import InferenceClient
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction

# Load environment variables from .env file
load_dotenv()

# setup api key 
hf_key = os.getenv("HUGGINGFACE_API_KEY")

hf_ef = HuggingFaceEmbeddingFunction(
    api_key=hf_key, model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 1. Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chromadb_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=hf_ef
)

# 2. Initialize the InferenceClient/model you want to use
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_key)

# 3. Query the model
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is human life expectancy in the United States?",
        },
    ]
)
# 4. Print the response
print(response["choices"][0]["message"]["content"])