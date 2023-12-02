from llama_cpp import Llama
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
llm = Llama(model_path="H:\Downloads\AHS_OPS_GGUF\ggml-model-q8_0.gguf")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
DB_FAISS_PATH = 'vectorstores/db_faiss'
db = FAISS.load_local(DB_FAISS_PATH, embeddings)
question="What technology does the technology team at Health Link maintain?"
docs = db.similarity_search(question,k=3)
context_list=[]
for i in docs:
    context_list.append(i.page_content)
context="\n".join(context_list)
# print(context)
prompt=f"""Below is an instruction that describes a task. Write a response that appropriately completes the request from the given context
    ### Instruction:
    {question}
    ### Context:
    {context}
    ### Response:
    """
output = llm(prompt, max_tokens=256, stop=["</s>"], echo=True)
print(output)
# for out in output:
#     completionfragement=copy.deepcopy(out)
#     print(completionfragement["choices"][0]["text"])