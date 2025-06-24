from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# function to load vectore database
def load_vectordb(persist_directory='rag/Vector_dase'):
    loaded_db=Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return loaded_db


# Function to get relevant docs from the db
def get_relevant_docs(query,k=5):
    vector_store=load_vectordb()
    retriver=vector_store.as_retriever(search_kwargs={"k":k})
    docs=retriver.get_relevant_documents(query)
    return docs
