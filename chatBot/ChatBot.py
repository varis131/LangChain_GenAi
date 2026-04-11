import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)
chat_history = []

while True:
    user_input = input("Enter your question (or 'exit' to quit): ")
    chat_history.append({"role": "user", "content": user_input})
    if user_input.lower() == "exit":
        break

    response = model.invoke(chat_history)
    chat_history.append({"role": "assistant", "content": response.content})
    print("Response:", response.content)

print(chat_history)    