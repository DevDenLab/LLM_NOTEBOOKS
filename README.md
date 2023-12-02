# Table of Contents
1. [OBJECTIVES OF THE RESEARCH](#objective)
2. [VISUALISATIONS](#process-architecturediagram)
3. [RESULTS](#resultsdemo-of-final-model)
4. [TOOLS USED](#development-tools-used)
5. [Project Structure](#project-directory-structure)
6. [IMPORTANT LINKS](#some-important-links)
7. [TESTING CHATBOT LOCALLY](#how-to-test-the-chatbot-locally-on-your-machine)
8. [How to convert Pytorch.bin(Model) to GGUF format](#how-to-convert-pytorchbin-to-gguf-format)
9. [Q&A](#qa)
10. [TERMINOLOGIES](#terminologies)
11. [RESOURCES](#resources)
___
# OBJECTIVE:
To create a Question Answering Chatbot using the data scraped from collection of webpages from [Insite Operations website](https://insite.albertahealthservices.ca/811/Page4802.aspx). It should have the following Functionalities:
- [x] Answer Not-so-complicated Questions Related to the AHS operations information.
- [x] Understand the User-query(Even if its grammatically Incorrect) and give coherent Responses.
- [x] Should be able to give Correct URLs/Links/Email/Phone no/etc..(vulnerable aspects where model tends to hallucinate).
- [x] Should be able to run on CPU/Consumer Hardware.
- [x] Can Answer related to WPCS team members(In progress).
- [x] Can remember past history or conversation history.(In progress).
- [x] Able to Scrape the Waitimes Data on runtime and Include that In the Response[Manually].
- [ ] Able to Scrape the Waitimes Data on runtime and Include that In the Response[Dynamically](In progress).
- [ ] Can Restrict answering the unrelated or inappropriate questions using RLHF/DPO(In progress).
- [ ] Can it stream the answer to CLI and to GUI(In progress).
- [ ] Can the entire process of finetuning from any website data be automized using a pipeline(Dont Know).
- [ ] Can it reply to multiple users at the same time(Dont know).

___
# Process Architecture/Diagram:
|![alt text](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/AHS_BOT%20-%20Page%201.png)|
|:-:|
|High-level diagram of ***LLM Fientuning Process***|
___
|![alt text](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/AHS_BOT_Inference.png)|
|:-:|
|High-level diagram of **LLM Inference Process**|
___
|![alt text](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/End_to_End_Process.png)|
|:-:|
|High-level diagram of **End-To-End Process**|
___
# Results/Demo of Final Model:
|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/test-1.png)|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/test-2.png)|
|:-:|:-:|
|Test-1(**Success**)|Test-2(**Success**)|
____
|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/test-3.png)|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/test-4.png)|
|:-:|:-:|
|Test-3(**Success**)|Test-4(**Success**)|
____
|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/test-5.png)|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/test-6.png)|
|:-:|:-:|
|Test-5(**Success**)|Test-6(**Success**)|
___
|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/Issue-3.png)|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/issues-2.png)|
|:-:|:-:|
|Test-7(**Issue/Resolved**)-Generated correct response with correct url but did not stop and continue generating additonal related info(**Resolved**)|Test-8(**Issue/Resolved**)-It answers the questions by providing the right url but also outputs the urls which have not been asked in a query(**Resolved**)|
___
|![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/Conversation.jpeg)|
|:-:|
|Test-9(**Success**)-A Complete Conversation Chat with User. The first response demonstrate the Model's capability to understand the User Query and Give brief , speicific and coherent Response. The second response states that this model does not hallucinates the URL/LINKS/etc.. The Third Response explains how Language model can Interact with API of Waitime to give Updated Info about waitime to the user in real time. The fourth response tells the model's comphrensive ability in understanding scrambled text question and answering Accurately. The fifth response suggests that It can take ainto account the conversation history and doesnot stop at first question but also answer the subsequent question in the user query. |
___
# Features of Final Model:
* Can Process upto 2048 input tokens(words).
* Low memory usage of 1887.56 MB(1.9GB).
* Can be run on low-end CPU/consumer hardware.
* Comes with a chailit GUI.
* Occassionally generates additional content after final answer(An issue).
* Takes upto 2-4 mins in Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, 2694 Mhz, 2 Core(s), 2 Logical Processor(s).
* has 3.43 B parameters .
* its an Quantised/Compressed version of the originally finetuned model.
* uses <b>llamatokenizer</b> for tokenizing the input query.
* Number of words in its own vocabulary=32000 words that it can understand and generate.
* Language Model Architecture is based on <b>LLama</b>,which is decoder-only transformer.

___   
# Development Tools used:
1. **pytorch**-Deep Learning Framework.
2. **python**-interpreted programming language
3. **Huggingface**-for storing model files and dataset
4. **Transformers**-for working with LLMs
5. **TRL**-for doing supervised finetuning
6. **PEFT**-for applying parameter efficient finetuning config
7. **bitsandbytes**-for internal conversions of model weights to different precisions and different types.
8. **accelerator**-for managing computation resources during training/finetuning and testing/inference.
9. **langchain**-for easily accessing the finetuned model,connecting llm with vector store,etc..
10. **llamcpp**-c++ implementation for doing inference on llama models,it can be used with python bindings,for faster processing.
11. **datasets**-for converting pandas dataframe to dataset object for training.
12. **Kaggle and Colab**- These platforms offers free GPU usage with some restrictions.Kaggle provides two 15 GBs gpus and Colab provide 1 16 Gbs GPU for Training/Finetuning/Inference.
13. **Chainlit**- Its a Framework/Library which gives a ready made GUI to integrate our LLM with with minimal effort.
14. **FAISS**- Vector database developed by Meta developers.
___  
# SOME IMPORTANT LINKS:
1. GGUF MODEL Created:
https://huggingface.co/Tatvajsh/AHS_OPS_GGUF_V_1.0/blob/main/OPS_WPCS_Q4.gguf
2. BASE LLM Used:
https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main
3. Dataset Format Used:
https://huggingface.co/datasets/tatsu-lab/alpaca?row=0
4. Dataset For Finetuning:
https://huggingface.co/datasets/Tatvajsh/AHS_Operations/viewer/default/train
5. Dataset For Reinforcement Learning Using Human Feedback:
https://huggingface.co/datasets/Tatvajsh/DPO_AHS_OPS_WPCS_MIX_V_1.0

___
# How to Test the Chatbot Locally on Your Machine:

To run the Chatbot locally on your machine, follow these steps:

1. **Download Code File:**
   - Download the [Python code file](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/3.Chainlit%2BGGUF_Inference%2BLangchain%2BVectorStore%2BQ%26A_web.py) and place it in a separate directory.

2. **Install Python:**
   - Install the [Python programming language](https://www.python.org/downloads/).
   - Make sure to select the option to add Python to your system's PATH during installation.

3. **Create a Virtual Environment:**
   - Navigate to the directory where the Python file is stored.
   - Execute the following commands in order:
     ```bash
     cd path/to/your/directory
     python -m venv myenv
     myenv\Scripts\activate
     pip install -r requirements.txt
     ```

4. **Download Dependencies File:**
   - Download the [dependencies file](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/requirements.txt) and place it in the same directory as the Python file.

5. **Download Dataset File:**
   - Download the [dataset file](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/DATASET/AHS_OPS_TEXT(combined_text_2).txt) and place it in a directory named "data" within the same directory as other files.

6. **Download Model:**
   - Download the [model file](https://huggingface.co/Tatvajsh/AHS_OPS_GGUF_V_1.0/blob/main/OPS_WPCS_Q4.gguf) and place it in the same directory.

7. **Download Ingest File:**
   - Download the [ingest file](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/Ingest.py) and run the following command:
     ```bash
     python ingest.py
     ```
   - Ensure that the file structure includes an empty "demo" folder, Ingest.py, the main Python file, the dependencies file, the model, and a "data" directory within the "demo" folder.

8. **Update Model Path:**
   - Change the following line in the Python file to point to the model file in your directory:
     ```python
     model_path = "OPS_MODEL_Q4.gguf"
     ```

9. **Run the Chatbot:**
   - Execute the following command to test the model on the command line:
     ```bash
     chainlit run 3.Chainlit+GGUF_Inference+Langchain+VectorStore+Q&A_web.py
     ```

10. **Access the Chatbot:**
    - A webpage will open automatically, allowing you to query the bot as needed.
    - Note: Response generation may take 2-5 minutes, depending on your system specifications.

---

If you encounter any issues or need assistance, feel free to:
- [Raise an issue](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/issues) in the repository's issues section.
- For additional help, contact Tatva Joshi at [Tatva.joshi@albertahealthservices.ca](mailto:Tatva.joshi@albertahealthservices.ca).

# How to convert Pytorch.bin to GGUF format:
1. pip install llama-cpp-python
2. Download convert.py from https://github.com/ggerganov/llama.cpp/blob/master/convert.py (llamacpp repo)
3. Hit Below Command to start Converting:
    1. python convert.py <path to OpenLLaMA directory> (use context of 1024 or 2048 ,I kept 8bits size(instead of full 16 bits) thus q8_0 for outtype and keep all the files as it is in that directory)
    - EXAMPLE: python convert.py H:\Downloads\AHS_OPS_GGUF  --ctx 2048 --outtype q8_0
4. For 4-bit int quantised version of the GGUF model use this command:
    1. quantize.exe ggml-model-f16.gguf ahs_ops_q8.gguf q4_0

___
# Project Directory Structure

I. [**1.FINETUNE_AHS_OPS_WPCS.ipynb:**]
   - Main notebook containing the logic for fine-tuning the Chatbot using AHS, OPS, and WPCS data.

II. [**2.INFERENCE_ADAPTER_OPENLLAMA_LANGCHAIN_VECTORSTORE.ipynb**]
   - Code to do an Inference without using Langchain and using transformer library text Generation API.

III. [**3.Chainlit+GGUF_Inference+Langchain+VectorStore+Q&A_web.py**]
   - The Main Code file to start a localhost with Chainlit UI and connects that to the local llm and run all the functions using the help of Langchain.

IV. [**4.GGUF_INFERENCE(LlamaCPP+GPU+VectorEmb+Fastest).ipynb**]
   - Notebook file which includes the code file for doing Inference on GGUF file using llamacpp on GPU.
V.  [**5.LLamacpp+CPU+VectorStore+Inference.py**]
   - Notebook file which includes the code file for doing Inference on GGUF file using llamacpp on CPU.

VI. [**6.DPO_RLHF.ipynb**]
   - Code file to do Reinforcement Learning using Human Feedback.

VII. [**7.LangchainAgent_ConceptTest.py**]
   - Testing code for checking How an Agent Works

VIII. [**Ingest.py**]
   - Python script for ingesting Text data Inside Vectorstore.

IX. [**Web_Scraper_and_Vector_Store_utility.py**]
   - Python script serving as a utility for web scraping and vector storage. It handles data preprocessing, cleaning, and organization tasks.

## Directories:

X. [**DATASET/**]
   - Directory for All the Datasets used in this Project

XI. [**static/**]
   - Directory to store all the Images used in the Github Repo file

___
# Terminologies:
1. **Finetuning**
   > As LLMs are pretrained on large amount of generic text data,which helps it to learn general language understanding,grammar and context. Finetuning leverages this general knowledge and refines the model to achieve better performance and udnerstanding in a specific domain. This process usually entails training the model further on a smaller,targeted dataset that is relevant to the desired task or subject matter.
   > >  **For Example**, I used finetuning on a pretrained language model named openllama to leverage its general language understanding capability for the downstream task of Q&A(on organisation data). Beacuse the pretrained model will not have any information about the private data of any organisation.
2. **Inference**
   > An Inference refers to the model's ability to generate predictions or responses on the context and input it has been given. When the model is given a prompt,it uses its understanding of language and context to generate a response that is relevant and appropriate. SO When we say we are doing inference on the model,it simply means that we are giving the model a prompt and expecting it to genrate an appropriate answer/response from the given prompt.
3. **Autoregressive Language Model**
    > Autoregressive refers to the nature of the text generation process where the model predicts one token at a time,considering the previously generated tokens.
        In the context of the transformers,during text generation,the model starts by predicting the first token,then conditions its prediction for the next 
        token on the previously generated ones. So, when It's mentioned that the decoder processes tokens in **an autoregressive manner**, it means it predicts 
        each token based on the preceding tokens it has generated in the sequence.
4. **Retrieval Augmented Generation(RAG)**
   > Retrieval-augmented generation (RAG) for large language models (LLMs) aims to improve prediction quality by using an external datastore at inference time to build a richer prompt that includes some combination of context, history, and recent/relevant knowledge.See this [image](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/End_to_End_Process.png) to have better understanding.
5. **Tokenizer**
   > The main purpose of using a tokenizer with a language model like LLM (Large Language Model) is to convert text into data that can be processed by the model. Models can only process numbers, so tokenizers need to convert our text inputs into numerical data.To get better Idea,see below Image:
  ![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/Tokenizer.png)
6. **Transformer**
7. **Embeddings**
   > An embedding is a representation of a word or token in a continuous vector space.In language models, embeddings are learned representations of words or tokens that capture semantic meaning.For example, in the sentence "The cat is cute," each word ("The," "cat," "is," "cute") would have its own embedding vector. **Another Example**:let's consider the words "king" and "queen." In a well-trained language model, the embeddings for these words might be arranged in the vector space in such a way that the vector for "queen" is in a direction similar to the vector for "king," reflecting the semantic relationship between these words. In my case, The embeddings of each word has length of 3200,that means if I have given a word,lets say,"action",then this word would be converted to the embedding of the size 3200. Which we can not comprehend,but its can be comprehended by the neural netowowk quite easily. Why do I need to convert the word to embedding?->Follow the Explanation of how transformers work in Explanation section(Coming Soon...)
   >> In a nutshell,The primary goal of these embeddings is to capture the semantic (meaningful) relationships between words or tokens. This means that words with similar meanings or that often appear in similar contexts will have embeddings that are close together in the vector space. The model learns to represent not just the surface form of the words but also the underlying semantic relationships between them.
   >> For Example,![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/sent_trans-1.png)
   >> ![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/sent-trans-2.png)
9. **Vector store/database**
   > Vector databases are designed to efficiently store and retrieve **vectors** or **embeddings**. They provide fast similarity searches and are commonly used in applications where finding similar vectors or embeddings is a key requirement. Just like in my case, where I am retrieving similar records from the vector store and passing those similar records along with the query to the language model,just so to improve the accuracy of responses from llm.
10. **Parameter Efficient Finetuning**
11. **Sentence-transformers**
12. **Zero-shot learning,One-shot learning,Few-shot learning**
    > In the context of large language models, **zero-shot learning** refers to the model’s ability to generate responses to prompts it has never seen during training. For instance, if you ask a language model to write a poem about a topic it has never been trained on, it can still generate a reasonable poem. This is because it has learned the general task of “writing a poem” during training and can apply this knowledge to new topics. **One-shot learning** in large language models refers to the model’s ability to understand and respond to a prompt based on a single example provided in the prompt. For example, if you provide a single example of a task in the prompt, such as “Translate the following English text to French: ‘Hello, how are you?’”, the model can understand the task and generate the correct response, even though it has only seen one example of the task. **Few-shot** learning in large language models refers to the model’s ability to understand and respond to a prompt based on a few examples provided in the prompt. For example, if you provide a few examples of a task in the prompt, such as “Translate the following English texts to French: ‘Hello, how are you?’, ‘Good morning, have a nice day.’”, the model can understand the task and generate the correct responses for each example.
13. **Quantisation**
14. **Structured Dataset**
15. **Unstructured/Raw Dataset**
16. **Vector Embeddings vs Static Embeddings vs Context-aware Embeddings**
17. **Finetuned Adapters**
18. **Training/Fine-tuning Prompt Template**
19. **Vector store**
20. **Hyperparameters**
21. **Text Generation Config**
22. **Instruction tuning**
___
# Machine Learning Lingo:
1. Semi-supervised Learning:
   > Semi-supervised learning is a machine learning paradigm where a model is trained on a dataset that contains both labeled and unlabeled data. The model learns from the labeled examples and generalizes its knowledge to make predictions on the unlabeled data.
   >> Suppose we have a dataset of news articles, with only a fraction of them labeled with categories (e.g., sports, politics). A semi-supervised language model could use the labeled articles to learn the relationships between words and topics, and then apply that knowledge to categorize the unlabeled articles.
2. Self-Supervised Leanring:
   > Self-supervised learning is a training paradigm where a model generates its own labels from the input data, without requiring external annotations. The model is designed to predict certain parts of the input from other parts, effectively creating a supervisory signal from the data itself.
   >> In natural language processing, a self-supervised language model might be trained to predict the next word in a sentence based on the context provided by the preceding words. The model doesn't need labeled data; it learns to understand language by predicting missing parts within the input it receives.
3. Unsupervised Learning:
   > Unsupervised learning involves training a model on data without explicit labels or categories. The algorithm tries to find patterns, relationships, or structures in the data without the guidance of labeled examples.
   >> Unsupervised learning involves training a model on data without explicit labels or categories. The algorithm tries to find patterns, relationships, or structures in the data without the guidance of labeled examples.
4. Reinforcement Learning:
   > Unsupervised learning involves training a model on data without explicit labels or categories. The algorithm tries to find patterns, relationships, or structures in the data without the guidance of labeled examples.
   >> Consider a conversational AI system. The model interacts with users, and based on the feedback (rewards or penalties) it receives from the users, it adjusts its responses to improve the overall quality of the conversation. The goal is to learn a policy that leads to more satisfying interactions over time.
5. Supervised Learning:
   > Supervised learning is a machine learning paradigm in which a model is trained on a labeled dataset, where each training example consists of input-output pairs. The model learns to map the input data to the corresponding output by generalizing from the labeled examples.
   >> Consider training a language model for sentiment analysis. The dataset includes text samples (input) along with their corresponding sentiment labels (output), such as "positive," "negative," or "neutral." The model learns to associate certain patterns in the text with specific sentiment labels during training. Once trained, it can predict the sentiment of new text it has not seen before.
___
# Q&A:
1. **Why did I use OpenLlama 3b v2?**
   > I started out my research by testing this given [language model](https://huggingface.co/TheBloke/orca_mini_3B-GGML/blob/main/orca-mini-3b.ggmlv3.q4_0.bin) to me. It was not working well with our AHS data. And a technique called Finetuning was required to teach that model about AHS data,so that It could answer questions related to the AHS more accurately. So for that, Created a dataset and tried finetuning the language model with that. But,later,I stumbled upon a concept of quantisation,which is a technique to reduce the size of a pretrained large language model. So, when a quantisation is done on any language model,it reduces it size and one more **Important thing** is that this newly created compressed model can not be finetuned for some technical deep reasons. As previously said, I used **orca-mini-3b.ggmlv3.q4_0** language model,which is quantised version of the original pretrained large language model [**OpenLLaMA-3b-v2**](https://github.com/openlm-research/open_llama). So its clear,that I could not finetune the orca model mentioned above but I could finetune the Openllama model. That is the sole reason about how I went with using Openallama for this Research.
3. **Why did I use Pytorch?**
   > Pytorch comes with user-firendly APIs(functions) which are Easy to use. Whereas Tensroflow comes with APIs complex and verbose,making the development not so easier. Secondly, Pytorch code is easy to debug beacsue of the previous reason and Tensorflow code is slighlty requries more effort to debug. One point against Pytorch is that it is super helpful or the best suited Deep Learning framework for the development process. However,as we may all know that after development step deployment step follows,which makes the code that we made, available for all users to use. Upon a short research,I found that Tensorflow has many integration with cloud providers and it has best utilities for putting our developed stuff to production server. So, Tensorflow wins that aspect.
4. **Why am I doing both finetuning and RAG while doing an Inference?**
   - Beacuse Neither doing alone finetuning nor doing alone RAG were fullfilling the objective of generating correct URLs.When I tried the RAG after finetuning,out of the blues,it started working. I am still finding the reasons out why does it work like that. But,I believe that If I keep the same prompt template during finetuning and DUring Inference,The model's neurons associated with the asked info in its weights gets activated and it goes with the URL which is provided in the context and It will not try making one up. That's the reason I have been able to find out.
   - I also belive that given the restrictions in the computation power,as the 3 billion parameter models are not well-pretrained on very large dataset,so when we finetune them ,they tend to give incorrect url.But I believe the big models with 30-50 billion parameters will tend to give correct urls just upon finetuning and the additional step of including RAG as well will not be required.
   - But there is also one point that,If we have access to resources which can run 30-60 billion parameter models,then we dont have to finetune the model.BEacuse,we can pass a context from vector embeddings to the model while generating answer from query and It will not hallucinate.
   - The necessity here of finetuning was that the base model even with passed context was not generating correct urls.Its like passing an answer with the question but even with that,it was not able to answer it rightly,This was the reason prompted me to finetune the model with data sample of urls as example in dataset to finetune with.
5. **What are parameters while mentioning 3B parameters model and hwo do they adjust inside the memory of gpu or cpu?**
   - Parameters are the weights or connection values in neural network. Each and every connection or in technical term **Weights** have their own value,which is learnt during the pretraining/training process. So when its said that the model has 3B parameteres,it means that it has 3*10^9 paramters or 3,000,000,000 wights inside its architecture or this many parameters forms the entire Language model backbone.
   - One more important thing, each of these parameters are stored in a FP32 datatype,which is called full-precision of any floating point number and which tells us that the weight is stored in highest possible precision or representation of the given weight.FP32 datatype takes 32 bits(4 bytes) to store a given weight.
   - We need to load the model inside the memory first to query it. For that,this type of calculation is used to fit it inside the GPU/CPU memory:
       -  For Example, If we have 3B parameter model and we are trying to load that in GPU of 16GB of VRAM. Each parameter inside has datatype FP32,so each of them will take 32 bits to store inside a memory. So,if 1 paramter takes 32 bits(4 bytes) for to be stored,then how many bytes will 3B(3*10^9) will take?. Thats simple,right?. It will take 3B multiply by 4 bytes=12 billion Bytes=12 Gigabytes=12 GBs in our GPU memory,which I believe would be able to store or load the model without crashing the runtime.
6. **How much time does it take to run inside GPU of T10 type with 14 GBs VRAM?**
     - It only takes 2 to 15 seconds depending upon the length of the input,which is super fast.
7. **Why GPUs are preferred over CPUs while Doing an Inference on or while running a LLM?**
    - All Neural network calculations are done in the form of Matrix Multiplication and CPUs have very limited number of Cores where those calculations cannot be done faster or parallelly. Beacuse CPUs are not optimized to do that with only 32-64 cores. Whereas, modern GPUS are well-optimized to do the Matrix Multiplication much faster with parallelism using thousands of GPU processing cores specially optimized for that task. Would not now making use of GPU make sense?
8. **What is LoRA and QLoRA?**
    - **Lora** stands for **low rank adaptation of large language models**. It’s used for finetuning a pretrained model by using a mechanism called Matrix Decomposition into a lower rank. For Example, while we are doing a finetuning of a llm with our dataset then we freeze all layers of pretrained model and add additional lora layers at specified attention layers and only these layers will be trained during finetuning. In these layers, a matrix of the rank that we mention in the config will be applied.
    - **Qlora** stands for **Quantized and Low-rank adaptation of Large Language model**. It uses the idea of Lora and extends it with quantization techniques to reduce the size of the model during loading and during the computation of forward and backward pass inside the memory. Shockingly, it even works with 4-bit integer datatype for loading then dequantizing it to float16 for computation and then convert it back to the type we specify for storing that.
9. **what is Alpaca Format and Instruction Tuning?**
   - The newer techniques to finetune the pretrained model include the dataset in [the format of Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca?row=0), which I have used during the finetuning process. Its basically a format containing 2 columns namely Question/Instruction and Answer. The first one will contain the Questions asking for specific information and Latter will contain Answer to that question. Then This information will be combined with the below prompt template and under a new column named “text”, which we will send for training to the model.
        - **Prompt Template**:
            ![](https://github.com/TatvaJoshi/LLM_NOTEBOOKS/blob/main/static/images/prompt-template.png)
         
10. **What is Quantization and how is it achieved?**
    - An uncompressed version of the model (the full-precision model) has their weights parameters stored in FP32 datatype (takes about 4 bytes or 32 bits for storing one parameter). By changing the datatype from FP32 to FP16(half-precision), the research has found that there is a very negligible difference in performance of the language model by doing that. We may as well go for FP8(8-bit floating point) or 4-bit Int/floating point, resulting in incredible reduction of model size.
    - Fortunately, All of above could be automatically handled by using libraries called bitsandbytes and accelerator during model loading and finetuning
    - If we have a finetuned model in .bin format and we want to quantise it using 4bit and store in efficient format called GGUF then follow this [section](#how-to-convert-pytorchbin-to-gguf-format).
  
11. **What is GGML/GGUF file format?**
    - **Georgie Garganov’s Unified format(GGUF)** is the file format for storing large language model for inference. It is a successor of GGML file format eliminating several of the issues of the GGML file format such as GGUF is now Extensible, backward-compatible, faster loading and processing time, Model Architectural information is now available to access from gguf file itself and lastly, no longer limited to just only llama models,last but not least,Single-file deployment for easy distribution and loading without external files
   
12. **Why did I use Q4_0 which is 4bit int datatype for creating final version of the model?,which is in GGUF file format**
    - Given the task to make the model retrieve response much faster, I had to convert the model to this format as it provides the a lot faster processing and retrieval of response but I just had to compromise a very tiny bit of accuracy of the model.Thats the reason.

13. **What is A Transformer Architecture in Large Language Models? And How Does it work?**
14. **Whats the difference between Encoder-only trasnformers and Decoder-only Trasnformers?**
15.  **How does the Attention mechanism work inside the Trasnformer Architecture?**
16.  **What are logits and How do we decide the decoding strategy to convert those logits to Output tokens/words?**
17.  **Why do we use Tokenizer during Finetuning and Inference process?**
18.  **What is Sentence Transformer and how is it different from the Trasnformer used in Language Model?**
19.  **Does Embedding layer get updated during the finetuning process?**
20.  **How exactly the finetuning using LoRA adapters work under the hood using matrix decomposition algorithm?**
21.  **How to Evaluate the Results of a Language model,what are the different evaluation methods?**
   
___
# Resources:
1. To understand the fundamental concept and building block of any Language Model, especially the **Transformer**, check out this insightful blog:
[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
2. To understand what happens During Training and During Inference,Follow this Wonderfully Crafted Video By Huggingface Developer:https://www.youtube.com/watch?v=IGu7ivuy1Ag&t=1388s&pp=ygUddW5kZXJzdGFuZCB0cmFuc2Zvcm1lciBtb2RlbCA%3D
3. To Understand Structure of Any LLM Repository in Huggingface and to Understand Different files in any Language Model project,Follow this Amazing Small and Easy-to-complete Course By Huggingface:https://huggingface.co/learn/nlp-course/chapter9/1?fw=pt
