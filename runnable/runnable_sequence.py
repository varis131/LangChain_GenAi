from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from typing import Literal

load_dotenv()

#model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='write the joke about my friend {name}',
    input_variables=['name']
)
prompt2 = PromptTemplate(
    template='explain the joke {text}',
    input_variables=['text']
)

chain=RunnableSequence(prompt1, model, parser, prompt2, model, parser)
result = chain.invoke({'name': 'Samarth'})
print(result)

