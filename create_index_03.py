import os 
from dotenv import load_dotenv 
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pinecone = Pinecone(
    api_key = os.getenv("PINECONE_API_KEY")
)
index_name = os.getenv("PINECONE_INDEX_NAME")

# OpenAI text-embedding-3-small is typically 1536 dimensions (typically use cosine)
dims = 1536

# create the index once, if it already exists, reuse that 
existing_indexes = [i.name for i in pinecone.list_indexes()]

# if the index name does not already exist, create a new one 
if index_name not in existing_indexes:
    pinecone.create_index(
        name = index_name, 
        dimension = dims, 
        metric = "cosine",
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index: {index_name} already exists")