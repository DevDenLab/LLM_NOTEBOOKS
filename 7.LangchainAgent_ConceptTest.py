'''
Task => Create an Agent that Identifies a URL in the user's query. It should separate the URL and the user's query. 
 After creating a vector store using the URL, 
try answering the user's query. The accuracy of answers is not as important as the implementation of this task.
'''
#1.Importing all the necessary Libraries.
from langchain.llms import CTransformers
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools import BaseTool
from typing import Union
from langchain.agents import initialize_agent
from langchain.agents import AgentType
#2.Initialising the Language Model
llm = CTransformers(
        model="H:\WPCS_OPS_MERGED\OPS_WPCS_Q4.gguf", 
        model_type="llama",
        config={
        "temperature":0.15,
        "max_new_tokens":512,
        "context_length":1000,
        "top_k":30,
        "repetition_penalty":1.2}
)
#3.Defining Custom Agent
class WordCountTool(BaseTool):
        name = "Word Length Counter"
        description = "use this tool when you need to count the number of letters in a word"

        def _run(self, word: str):
            return "The universal word count is 20."

        def _arun(self, radius: Union[int, float]):
            raise NotImplementedError("This tool does not support async")
        
tools = [WordCountTool()]

#4.initialize agent with tools
agent = initialize_agent(
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=2,
        early_stopping_method='generate',
        handle_parsing_errors=True
    )


#5.Checking the Agent Prompt
sys_msg = """Assistant is a large language model.Assistant is designed to be able to assist with a wide range of tasks, \
from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, \
Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations \
and provide responses that are coherent and relevant to the topic at hand. Assistant is constantly learning and improving, and its \
capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to \
provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text \
based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Unfortunately, Assistant is terrible at counting word length. When provided with counting word length questions, no matter how simple, \
    assistant always refers to it's trusty tools and absolutely does NOT try to answer questions by itself.Remember to answer with Final Answer at the end.
"""
agent.agent.llm_chain.prompt.messages[0].prompt.template = sys_msg
# print(agent.agent.llm_chain.prompt.messages[0].prompt.template)


#6.Run The Agent
response= agent.run("Using custom tool please find number of letters in the word REACT?")
from datasets import Dataset, load_dataset
print(response)