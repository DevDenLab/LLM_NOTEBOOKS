#This Python file creates a Chainlit UI,connects the UI to GGUF Model AHS_OPS_WPCS,Manages the Vectorstore operations,Handles Waitime Questions and
# Q&A over any PDF and Website on text data.

# 1.Import all the Necessary Packages/Modules
from llama_cpp import Llama
from langchain.chains import LLMChain,QAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_aiter_final_only import AsyncFinalIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain.callbacks.streaming_aiter_final_only import AsyncFinalIteratorCallbackHandler
from web_scraper_and_vector_store import Waitime_API_Scrape,wait_times_Scrape_dynamic,scrape_waitime_vectorstore,find_urls,Scrape_VectorStore,get_response_and_save,check_url_type,get_response_and_savePDF,ScrapePDF_VectorStore

#2.Declare the global Variables with corrosponding values
DB_FAISS_PATH = 'vectorstores\db_faiss'

#3.0 Prompt Template for LLMchain without Context and Memory Buffer.
custom_prompt_llmchain="""
 Below is an instruction that describes a task. Write a response that appropriately completes the request from the given context.
    ### Instruction:
    {question}
    ### Response:
    """

def set_custom_prompt_1():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_llmchain,
                            input_variables=["question"])
    return prompt

#3.1 creating Prompt Template including the conversation history and Context
custom_prompt_template="""
 Below is an instruction that describes a task. Write a response that appropriately completes the request from the given context.
    ### Chat History:
    {chat_history}
    ## Context:
    {context}
    ### Instruction:
    {question}
    ### Response:
    """
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=["chat_history","context","question"])
    return prompt


#4. Function for Loading the model
def load_llm():
    llm = CTransformers(
    model="H:\/6.WPCS_OPS_MERGED\OPS_WPCS_Q4.gguf", 
    model_type="llama",
    callbacks=[AsyncFinalIteratorCallbackHandler()],
    config={
    "temperature":0.15,
    # "max_new_tokens":512,
    # verbose=True,
    # streaming=True,
    "context_length":1800,
    "top_k":30,
    "repetition_penalty":1.2}
)
    return llm
#5. Function for Creating LLMchain
def create_chain():
    llm_chain = LLMChain(
            llm=load_llm(),
            prompt=set_custom_prompt_1(),
            verbose=True,
        )
    return llm_chain

#5. Function for Creating RetrievalQAChain
def create_chain_qa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    qa_chain = RetrievalQA.from_chain_type(llm=load_llm(),
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 3,"fetch_k":30}),
                                       verbose=True,
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': set_custom_prompt(),"verbose":True,"memory":ConversationBufferMemory(memory_key="chat_history",input_key="question",max_token_limit=150,return_messages=True)}
                                       )
    return qa_chain

# if __name__=="__main__":
#     chain=create_chain()
#     a=chain.run("How can I access the Service Status Page for HLA?")

#6. Chainlit Session Initialisation
@cl.on_chat_start
async def start():
    chain = create_chain_qa()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to AHS Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

#7. Chainlit on-user-message logic.
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER","Response","Result"]
    )
    cb.answer_reached = True
    if find_urls(message):
        url=find_urls(message)
        url_type=check_url_type(url[0])
    if "waittime" in message:
        filepath=Waitime_API_Scrape()
        scrape_waitime_vectorstore(filepath)
        chain = cl.user_session.get("chain")
        chain = create_chain_qa()
        cl.user_session.set("chain", chain)
        res = await chain.acall(message, callbacks=[cb])
    elif find_urls(message) and url_type=="SITE":
        path=get_response_and_save(url[0])
        _=Scrape_VectorStore(path)
        query=message.replace(url[0],"")
        print(query)
        res = await chain.acall(query, callbacks=[cb])
    elif find_urls(message) and url_type=="PDF":
        filepath=get_response_and_savePDF(url[0])
        ScrapePDF_VectorStore(filepath)
        query=message.replace(url[0],"")
        print(query)
        res = await chain.acall(query, callbacks=[cb])
    else:
        res = await chain.acall(message, callbacks=[cb])

