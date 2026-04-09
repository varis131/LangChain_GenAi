from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
sentence = "This is a test sentence."

embedding_vector = embedding.embed_query(sentence)
print(str(embedding_vector))  