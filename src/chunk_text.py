from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def split_documents(documents):

    # clean text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    return chunks