from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    
    # create loader object
    loader = PyPDFLoader(file_path)
    
    # load the document
    documents = loader.load()

    return documents

    