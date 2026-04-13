from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#1st prompt template
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#2nd prompt template
template2=PromptTemplate(
    template='Summarize the following text in 5 lines: {text}',
    input_variables=['text']
)

parser=StrOutputParser()
chain=template1 | model | parser | template2 | model | parser
output=chain.invoke({'topic':'Climate Change'})
print(output)
