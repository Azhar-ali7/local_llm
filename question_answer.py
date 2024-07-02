from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def create_qa_chain(knowledge_base):
    llm = Ollama(model="gemma:2b-instruct-q4_0", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        retriever=knowledge_base.as_retriever(),
        llm=llm
    )
    return qa_chain

def ask_question(qa_chain, question):
    response = qa_chain.invoke({"query": question})
    return response['result']
