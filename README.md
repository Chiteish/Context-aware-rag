RAG MODEL CREATED TO ANSWER THE USER QUERY BASED ON THE PDF UPLOADED BY THE USER

HERE THE SAMPLE INPUT PDF USED IS "cybersecurity.pdf"

FRAMEWORK USED:
"LANGCHAIN"

LOADING PDF:
At first it will be loading the pdf file with the help of "PYPDF" library that will be used to extract the text from the pdf 

CHUNKING THE PDF:
The new lines("\n")are replaced with the space to form a clean text datas and those text datas are further splitted into chunks of size 800 words with overlap of 150 using "RECURSIVE CHARACTER TEXT SPLITTER"

EMBEDDING THE PDF: 
Those chunked datas are embedded into numerical values using the HUGGINGFACE EMBEDDINGS of model named "sentence-transformers/all-MiniLM-L6-v2"

VECTOR STORAGE OF EMBEDDED DATAS:
The input numerical values are get stored in the vector database using the "FAISS"

RETRIEVER:
Retriver is used to fetch the most relevant datas from the vector database based on the user query where this uses default "similarity search" to fetch the most relevant datas

As in this rag model it used to return top 3 relevant datas from the database 

Those 3 top datas are combined to form a perfect ans for the provided query, this is done by sending the 3 datas to the LLM model to ans the query 

PROMPT TEMPLATE:
This file is used for giving prompt to the LLM model to ans the query by using the "prompt template" imported from langchain.core.prompts

RAG PIPELINE:
It is the core of the rag model where this file has "LLM,RETRIEVER,INMEMORYCHATMESSAGEHISTORY"

LLM MODEL that I used here from OLLAMA is "mistral"

Retriver is also include here to process with the user query

InMemoryChatMessageHistory is used to store the chat history of the user , the previous conversation history is stored in memory and based on that if the ques is asked then i uses that memory to ans this

RETRIVER.INVOKE:
This part which used to vectorize the user query and it wont store to database where this enables the vectorized query to fetch the most relevant datas from the vector database

Main.py
This file is used to run the rag model , where this is the file to execute and call all the functions in this model 

DATA FOLDER:
This folder is used to store the pdf file that is used as input to the rag model