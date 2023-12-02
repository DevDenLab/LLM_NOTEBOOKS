from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader,TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH="data/"
DB_FAISS_PATH="vectorstores/db_faiss"

#create vector database
def create_vector_db():
    #Create a DirectoryLoader to load all PDFs from the DATA_PATH. Use the PyPDFLoader to load each PDF.
    loader=DirectoryLoader(DATA_PATH,glob="*.txt",loader_cls=TextLoader)
    documents=loader.load()
    #shared overlapping text gives some continuity between chunks and context.
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=80)
    # text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)

    texts=text_splitter.split_documents(documents)# all the splitted text is here,text chunks
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})#creating the embeddings
    db=FAISS.from_documents(texts,embeddings)#using this embedding model,create all the embedding and store it 
    db.save_local(DB_FAISS_PATH)

if __name__=="__main__":
        create_vector_db()
