from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Get the directory of this script
base_dir = os.path.dirname(__file__)
# Create the full path to the PDF
pdf_path = os.path.join(base_dir, 'dl-curriculum.pdf')

loader = PyPDFLoader(pdf_path)

docs = loader.lazy_load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[0])