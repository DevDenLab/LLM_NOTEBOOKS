#1.Import alla necessary packages
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import re
import os
import requests
import json
import numpy as np
from pathlib import Path
from langchain.document_loaders import PyPDFLoader,DirectoryLoader,TextLoader
from selenium import webdriver
from selenium.webdriver.common.by import By

#1.Take an Input text string and returns list of all the found urls in that string
def find_urls(text):
    pattern = re.compile(r'\b(?:https?|ftp):\/\/\S+')
    return pattern.findall(text)

#2.Takes URL string and finds the type of that URL(Site/Pdf)
def check_url_type(url):
    response = requests.head(url)  # Use a HEAD request to fetch only headers, not the entire content
    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' in content_type:
        return 'PDF'
    elif 'html' in content_type or 'text' in content_type:
        return 'SITE'
    else:
        return 'Unknown'

#3.Function which takes Url string as input and generates the filename for storing the file in directory.
def cleanUrl(url: str):
    cleaned_url = url.replace("https://", "").replace("/", "-").replace(".", "_")

    # Add ".pdf" extension for PDF files
    if cleaned_url.endswith("_pdf"):
        return cleaned_url + ".pdf"
    else:
        return cleaned_url    

#4.Function which fetches the PDF data from its URL,create a pdf file in local directory and save the content in it.
def get_response_and_savePDF(url: str):
    filepath = Path('scrape') / cleanUrl(url)
    response = requests.get(url)
    filepath.write_bytes(response.content)
    return cleanUrl(url)

#5. Function which takes the Pdf file created using the above function and create smaller documents/segements and save it in vector store.
def ScrapePDF_VectorStore(filepath):
    DATA_PATH="scrape/"
    loader=DirectoryLoader(DATA_PATH,glob=filepath,loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    db=FAISS.from_documents(texts,embeddings)
    DB_FAISS_PATH="vectorstores/db_faiss"
    db.save_local(DB_FAISS_PATH)
   
#6. Fetches the static Html file data save that to a directory
def get_response_and_save(url: str):
    response = requests.get(url)

	# create the scrape dir (if not found)
    if not os.path.exists("./scrape"):
        os.mkdir("./scrape")

	# save scraped content to a cleaned filename
    parsedUrl = cleanUrl(url)
    with open("./scrape/" + parsedUrl + ".html", "wb") as f:
        f.write(response.content)
    
    return "./scrape/" + parsedUrl + ".html"

#7. Create vector embeddings of the fetched html file above and save it in vector store with filtering only text data and removing the tags using BSHTMLLoader.
def Scrape_VectorStore(url):
    loader = BSHTMLLoader(url,open_encoding="utf8")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=35,
    )
    documents = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})
    DB_FAISS_PATH = 'vectorstores\db_faiss'
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)
    return embeddings

#8. Scraping Waitimes API
def Waitime_API_Scrape():
    response = requests.get("https://www.albertahealthservices.ca/Webapps/WaitTimes/api/waittimes")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        text_output = ""

        for city, categories in data.items():
            for category, entries in categories.items():
                text_output += format_hospital_info(city, category, entries)

        # Print or save the formatted text
        filepath="waitime.txt"
        output_file_path = "scrape/"+filepath
        with open(output_file_path, "w") as file:
            file.write(text_output)
    return filepath        
#9. Formatting Waitimes API json response to appropriate format
def format_hospital_info(city, category, entries):
    formatted_text = f"\n{city} - {category}:\n"
    for entry in entries:
        formatted_text += f"  Name: {entry['Name']}\n"
        formatted_text += f"  Wait Time: {entry['WaitTime']}\n"
        formatted_text += f"  URL: {entry['URL']}\n"
        formatted_text += f"  Note: {entry['Note']}\n"
        formatted_text += "\n"
    return formatted_text

text_output = ""
#10. Fetches the dynamic content of Waitimes page using Dynamic scraping and save the scraped data in waitime text file.
def wait_times_Scrape_dynamic():
    # Set up a headless browser
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    # Navigate to the ASPX page
    driver.get('https://www.albertahealthservices.ca/waittimes/Page14230.aspx')

    # Locate the dropdown element
    select_element  = driver.find_element(By.ID, "dd-city-ab")

    # Get all options from the dropdown
    options = select_element.find_elements(By.TAG_NAME, "option")
    all_data=""
    # Extract and print the values
    for option in options:
        driver.execute_script("arguments[0].click();", option)

        # Wait for a brief moment to ensure the page has loaded the new data
        driver.implicitly_wait(1)
            # Replace 'your_class_name' with the actual class name of the data you want to scrape
        class_name_wt_well = 'well.wt-well'#!Task=>this could be improved.dont give just this class name,specific class name could help improve memory utilisation.

        try:
            # Find all elements with the specified class
            wt_well_elements = driver.find_elements(By.CLASS_NAME, class_name_wt_well)

            # Iterate through the found elements and print/scrape the data
            for wt_well in wt_well_elements:
                # Extract relevant information
                wt_times = wt_well.find_element(By.CLASS_NAME, 'wt-times').text
                hospital_name = wt_well.find_element(By.CLASS_NAME, 'hospitalName').text
                hospital_desc = wt_well.find_element(By.CLASS_NAME, 'wt-description.langDirection').text
                texty=f"Hospital Description:{hospital_desc} & Waittime for {hospital_name}:{wt_times}"
                if texty!="Hospital Description: & Waittime for :":
                    print(texty)
                    all_data += texty
                    all_data+="\n\n\n"
        except Exception as e:
            print(f"Error: {e}")
    filepath="waitime.txt"
    with open("scrape/"+filepath, "w") as file:  # Use "a" for append mode
        file.write(all_data)
    # Close the browser
    driver.quit()
    return filepath

#11. Creating the vector embeddings of the dynamically scraped content above from the text file to the vector store.
def scrape_waitime_vectorstore(filepath):
    DB_FAISS_PATH="vectorstores/db_faiss"
    DATA_PATH="scrape/"
    #Create a DirectoryLoader to load all PDFs from the DATA_PATH. Use the PyPDFLoader to load each PDF.
    loader=DirectoryLoader(DATA_PATH,glob=filepath,loader_cls=TextLoader)
    documents=loader.load()
    #shared overlapping text gives some continuity between chunks and context.
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
    # text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)

    texts=text_splitter.split_documents(documents)# all the splitted text is here,text chunks
    # print(texts)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})#creating the embeddings
    new_db=FAISS.from_documents(texts,embeddings)#using this embedding model,create all the embedding and store it 
    # db.save_local(DB_FAISS_PATH)
    old_db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    Removing_vectorstore_docs(old_db)
    deleted_old_docs_db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    
    deleted_old_docs_db.merge_from(new_db)
    deleted_old_docs_db.save_local(DB_FAISS_PATH)

#12. Removing the old documents from the vectorstore from metadata.
def Removing_vectorstore_docs(vectorstore):
    DB_FAISS_PATH="vectorstores/db_faiss"
    id_to_remove = []
    target_metadata={'source': 'scrape\\waitime.txt'}
    for _id, doc in vectorstore.docstore._dict.items():
        to_remove = True
        # print(doc)
        for k, v in target_metadata.items():
            if doc.metadata[k] != v:
                to_remove = False
                break
        if to_remove:
            id_to_remove.append(_id)
            
    
    docstore_id_to_index = {
        v: k for k, v in vectorstore.index_to_docstore_id.items()
    }
    n_removed = len(id_to_remove)
    n_total = vectorstore.index.ntotal
    vectors_to_remove = []#for removing the embeddings.
    for _id in id_to_remove:
        # remove the document from the docstore
        del vectorstore.docstore._dict[
            _id
        ]
        # remove the embedding from the index
        ind = docstore_id_to_index[_id]
        vectors_to_remove.append(ind) ### Modification here ########################
        # remove the index to docstore id mapping
        del vectorstore.index_to_docstore_id[
            ind
        ]
    # reorder the mapping
    vectorstore.index.remove_ids(
        np.array(vectors_to_remove, dtype=np.int64)
    )  #
    vectorstore.save_local(DB_FAISS_PATH)

# if __name__=="__main__":
#     DB_FAISS_PATH = 'vectorstores\db_faiss'
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                         model_kwargs={'device': 'cpu'})
#     vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings) 


#     Removing_vectorstore_docs(vectorstore)
#     retriever=vectorstore.as_retriever(search_kwargs={'k': 4,"fetch_k":30})
#     retrieved_docs = retriever.invoke(
#         "what is the waiting time for Red Deer Regional Hospital in hours at AHS?"
# )
#     print(retrieved_docs)
