from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1=PromptTemplate(
    template="explain this {topic} in detail",
    input_variables=["topic"],
)

prompt2=PromptTemplate(
    template="give the summary of the {text} explanation in three line",
    input_variables=["text"],
)

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "cricket"})
print(result)
chain.get_graph().print_ascii()