from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
sentence =["This is a test sentence.","kolkata is a city in india","the capital of india is delhi"]

embedding_vector = embedding.embed_documents(sentence)
print(embedding_vector) 