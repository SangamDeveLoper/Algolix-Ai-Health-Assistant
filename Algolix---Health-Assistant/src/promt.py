import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

system_promt = (
    "You are a helpful AI assistant that helps people find medical information. "
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Please provide concise and accurate medical information based on the context provided. "
    "\n\n"
    "Context: {context}"
)

promt = ChatPromptTemplate.from_messages(
    [
        ("system", system_promt),
        ("human", "{input}"),
    ]
)

# Make sure to set your Google API key in environment variables
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)