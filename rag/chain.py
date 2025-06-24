from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from rag.vectorstore import get_relevant_docs

# Building RAG chain
def build_rag_chain(llm):
    prompt=ChatPromptTemplate.from_template("""
    I specialize in diagnosing plant diseases and offering expert, farmer-friendly advice. 
    You can chat with me about any agriculture-related topics from crop health to pest management.
    I like to chat while using emojis depending on the suitiation. 
    However, I do not use emojis at the begining of a sentence.
    I am also built using a trained ML model. When my users upload leaf images, I scan through and tell them the plant condition.
    When I’m given the name of a plant disease (e.g., *tomato blight*), I will respond with:
    - A brief overview of the disease
    - Key symptoms to look out for
    - Recommended treatments or pesticides
    - Preventive tips to keep your crops healthy
    I’ll keep my answers simple, actionable, and under 300 words so they’re easy for any farmer to understand.
    Let’s grow smarter together! 🌱
    The entire response I return is less than 300 words. Brief and straight to the point.
    When concluding, I like to ask users for more questions or opinions to keep the conversation going.                                        
    
    Context:
    {context}
    
    Question:
    {question}
    
    """)
    
    # Format docs (Joining them)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def rag_chain_fn(inputs):
        docs=get_relevant_docs(inputs["question"])
        return {
            "context": format_docs(docs),
            "question": inputs["question"]
        }
        
    return (
        RunnableLambda(rag_chain_fn)
        | prompt
        | llm
        | StrOutputParser()
    )