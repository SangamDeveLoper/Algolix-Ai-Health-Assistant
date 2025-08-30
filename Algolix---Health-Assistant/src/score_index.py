import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helper import load_pdf_file, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")

# Load and process documents
extracted_data = load_pdf_file(data='../Data/')
text_chunks = text_split(extracted_data)

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "algolix-health-assistant"

# Create index if it doesn't exist
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )
    print(f"Created new index: {index_name}")
else:
    print(f"Using existing index: {index_name}")

# Connect to the index
index = pc.Index(index_name)

# Create vector store from existing index
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# Add documents if we have them
if text_chunks:
    print(f"Adding {len(text_chunks)} document chunks to the index...")
    docsearch.add_documents(text_chunks)
    print("Documents added successfully!")
else:
    print("No document chunks to add.")

print("Vector store setup complete!")