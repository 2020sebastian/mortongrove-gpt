from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import requests
import streamlit as st

def get_wiki_data(title):
    url = f"https://mortongrove.municipalcodeonline.com/book/print?type=ordinances&name={title}"
    data = requests.get(url)
    return Document(
        page_content=str(data.content),
        metadata={"source": f"https://mortongrove.municipalcodeonline.com/book?type=ordinances#name={title}"},
    )

sources = [
    get_wiki_data("Title_10_BUILDING_AND_CONSTRUCTION_REGULATIONS"),
]

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

chain = load_qa_with_sources_chain(OpenAI(temperature=0))

def print_answer(question):
    return(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=1),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )


st.header("`Morton Grove building code`")
st.info("`Hello! I am a ChatGPT connected to the Morton Grove building code.`")
query = st.text_input("`Please ask a question:` ","How do I dispose of construction debris?")
result = print_answer(query)
st.info("`%s`"%result)