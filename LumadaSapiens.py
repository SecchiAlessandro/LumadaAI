import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import openai  # Update to import openai directly
import os
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_community.document_loaders import TextLoader, DirectoryLoader, JSONLoader
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Correct way to set API key
openai.api_key = os.getenv("COHERE_API_KEY")  # Correct way to set API key
openai.api_key = os.getenv("TAVILY_API_KEY")  # Correct way to set API key



llm = ChatOpenAI(model="gpt-4o")


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


#tools

from typing import Annotated, List, Tuple, Union, Iterator

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

python_repl_tool = PythonREPLTool()

#load scraped websites for RAG tools

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key

    def clean_text(self, text: str) -> str:
        # Remove \n, \xa0, and multiple spaces
        text = text.replace('\n', ' ')  # Replace \n with a space
        text = text.replace('\xa0', ' ')  # Replace non-breaking space with a normal space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()  # Remove leading and trailing spaces


    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs=[]
        # Load JSON file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            # Iterate through 'pages'

            for item in data:
                title = item['title']
                url = item['url']
                html_content = item['html']

                # Ensure the html_content is a string
                if isinstance(html_content, str):
                    # Clean the html content
                    cleaned_html_content = self.clean_text(html_content)
                else:
                    raise TypeError(f"Expected 'html' to be a string, but got {type(html_content).__name__}")
                
                
                # Create metadata
                metadata = dict(
                    source=url,
                    title=title
                )



                # Create a Document object with the html content and metadata
                docs.append(Document(page_content=cleaned_html_content, metadata=metadata))
        return docs

#load data
file_path_he='./hitachi_energy.json'
file_path_hr='./hitachi_rail.json'
file_path_gl='./hitachi_globallogic.json'
file_path_as='./hitachi_astemo.json'
file_path_ma='./hitachi_machinery.json'
file_path_ht='./hitachi_hightech.json'
loader_he = JSONLoader(file_path=file_path_he)
loader_hr = JSONLoader(file_path=file_path_hr)
loader_gl = JSONLoader(file_path=file_path_gl)
loader_as = JSONLoader(file_path=file_path_as)
loader_ma = JSONLoader(file_path=file_path_ma)
loader_ht = JSONLoader(file_path=file_path_ht)
docs_he = loader_he.load()
docs_hr = loader_hr.load()
docs_gl = loader_gl.load()
docs_as = loader_as.load()
docs_ma = loader_ma.load()
docs_ht = loader_ht.load()

# Split

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=10)
new_docs_he = text_splitter.split_documents(docs_he)
new_docs_hr = text_splitter.split_documents(docs_hr)
new_docs_gl = text_splitter.split_documents(docs_gl)
new_docs_as = text_splitter.split_documents(docs_as)
new_docs_ma = text_splitter.split_documents(docs_ma)
new_docs_ht = text_splitter.split_documents(docs_ht)


vectorstore_he = Chroma.from_documents(
    documents=new_docs_he,
    collection_name="HitachiEnergy-rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever_he = vectorstore_he.as_retriever()

vectorstore_hr = Chroma.from_documents(
    documents=new_docs_hr,
    collection_name="HitachiRail-rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever_hr = vectorstore_hr.as_retriever()

vectorstore_gl = Chroma.from_documents(
    documents=new_docs_gl,
    collection_name="GloabalLogic-rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever_gl = vectorstore_gl.as_retriever()

vectorstore_as = Chroma.from_documents(
    documents=new_docs_as,
    collection_name="HitachiAstemo-rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever_as = vectorstore_as.as_retriever()

vectorstore_ma = Chroma.from_documents(
    documents=new_docs_ma,
    collection_name="HitachiMachinary-rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever_ma = vectorstore_ma.as_retriever()

vectorstore_ht = Chroma.from_documents(
    documents=new_docs_ht,
    collection_name="HitachiHighTech-rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever_ht = vectorstore_ht.as_retriever()


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#tool RAG on Hitachi Energy
@tool
def HitachiEnergy(state):

    """Use this tool to retrieve information when the question is related to high voltage trasmission products, switchgear, power electronics, transformer, grid automation and other products and solutions of Hitachi Energy."""
    
    state = str(state)

    print('-> Calling RAG on Hitachi Energy ->')
    question = state
    print('Question: ', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever_he, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result

#tool RAG on Hitachi Rail
@tool
def HitachiRail(state):

    """Use this to retrieve the information when the questions are related to manufacturing rail systems, including high-speed trains, metros, and signaling solutions and other products of Hitachi Rail."""
  
    state = str(state)

    print('-> Calling RAG on Hitachi Rail->')
    question = state
    print('Question: ', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever_hr, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result

#tool RAG on Global Logic
@tool
def GlobalLogic(state):

    """Use this to retrieve information when the questions are related to digital product engineering, software solutions and other digital products of globallogic."""
   
    state = str(state)

    print('-> Calling RAG on GlobalLogic ->')
    question = state
    print('Question:', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever_gl, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result


#tool RAG on Hitachi Astemo
@tool
def HitachiAstemo(state):

    """Use this to retrieve information when the questions are related to the automotive and transportation systems, producing advanced components for vehicles, including powertrains, chassis systems, and advanced driver-assistance systems (ADAS) and other products of Hitachi Astemo."""
   
    state = str(state)

    print('-> Calling RAG on HitachiAstemo ->')
    question = state
    print('Question:', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever_gl, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result


#tool RAG on Hitachi Machinery
@tool
def HitachiMachinery(state):

    """Use this to retrieve information when the questions are related to heavy equipment for the construction and mining industries such as excavators, wheel loaders, dump trucks, and cranes and other solutions of Hitachi COnstruction Machinery."""
   
    state = str(state)

    print('-> Calling RAG on HitachiMachinery ->')
    question = state
    print('Question:', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever_gl, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result


#tool RAG on Hitachi HIghTech
@tool
def HitachiHighTech(state):

    """Use this to retrieve information when the questions are related to high-tech solutions, including semiconductor manufacturing equipment, analytical instruments, and life sciences technology and other solutions of Hitachi High Tech."""
   
    state = str(state)

    print('-> Calling RAG on HitachiHighTech ->')
    question = state
    print('Question:', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever_gl, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    result = retrieval_chain.invoke(question)
    return result

  
#agent node

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

#supervisor

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

members = ["HitachiEnergy", "HitachiAstemo", "HitachiMachinery", "HitachiHighTech", "HitachiRail", "GlobalLogic" , "Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Use HitachiEnergy, HitachiAstemo, HitachiMachinery, HitachiHighTech, HitachiRail, GlobalLogic tools when questions "
     "are related to their companies products and solutions. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    #| llm.bind_tools(tools=[function_def], function_call="auto")

    
    | llm.bind_functions(functions=[function_def], function_call="route")

    | JsonOutputFunctionsParser()
)


import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

HitachiEnergy_agent = create_agent(
    llm,
    [HitachiEnergy],
    "Use this tools when questions are related to hitachi energy products and solutions such as electrical components.",
)
HitachiEnergy_node = functools.partial(agent_node, agent=HitachiEnergy_agent, name="HitachiEnergy")

HitachiAstemo_agent = create_agent(
    llm,
    [HitachiAstemo],
    "to the automotive and transportation systems, producing advanced components for vehicles, including powertrains, chassis systems, and advanced driver-assistance systems (ADAS) and other products of Hitachi Astemo",
)
HitachiAstemo_node = functools.partial(agent_node, agent=HitachiAstemo_agent, name="HitachiAstemo")

HitachiMachinery_agent = create_agent(
    llm,
    [HitachiMachinery],
    "Use this tools when questions are related to to heavy equipment for the construction and mining industries such as excavators, wheel loaders, dump trucks, and cranes and other solutions of Hitachi COnstruction Machinery.",
)
HitachiMachinery_node = functools.partial(agent_node, agent=HitachiMachinery_agent, name="HitachiMachinery")

HitachiHighTech_agent = create_agent(
    llm,
    [HitachiHighTech],
    "Use this tools when questions are related to high-tech solutions, including semiconductor manufacturing equipment, analytical instruments, and life sciences technology and other solutions of Hitachi High Tech.",
)
HitachiHighTech_node = functools.partial(agent_node, agent=HitachiHighTech_agent, name="HitachiHighTech")

HitachiRail_agent = create_agent(
    llm,
    [HitachiRail],
    "Use this tools when questions are related to hitachi rail products and solutions such as trains.",
)
HitachiRail_node = functools.partial(agent_node, agent=HitachiRail_agent, name="HitachiRail")

GlobalLogic_agent = create_agent(
    llm,
    [GlobalLogic],
    "Use this tools when questions are related to globalllogic solutions such as digital.",
)
GlobalLogic_node = functools.partial(agent_node, agent=GlobalLogic_agent, name="GlobalLogic")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("HitachiEnergy", HitachiEnergy_node)
workflow.add_node("HitachiRail", HitachiRail_node)
workflow.add_node("GlobalLogic", GlobalLogic_node)
workflow.add_node("LumadaAI", supervisor_chain)
workflow.add_node("HitachiAstemo", HitachiAstemo_node)
workflow.add_node("HitachiMachinery", HitachiMachinery_node)
workflow.add_node("HitachiHighTech", HitachiHighTech_node)



#edges

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "LumadaAI")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("LumadaAI", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("LumadaAI")

graph = workflow.compile()




    
# user_query = input("what are you looking for?  ")
# for s in graph.stream({"messages":[user_query]}):
    

#     if "__end__" not in s:
#         print(s)
#         print("----")



def LumadaAI(user_query):

    # user_query = input("what are you looking for?  ")
    response = []
    for s in graph.stream({"messages":[user_query]}):
    
        if "__end__" not in s:
            print(s)
            
            response_str = str(s)  # Adjust as necessary, e.g., extract specific fields if s is a dict
            combined_str = f"{response_str}\n\n  -----"
            response.append(combined_str)
            
            

    # Combine all parts of the response
    final_response = "\n\n".join(response)  # Double newline for added spacing
    return final_response




if __name__ == "__LumadaAI__":
    LumadaAI()








# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(content="is globallogic a startup?. Use RAG agent first, then with researcher agent tell me the top 3 startup at the moment")
#         ]
#     }
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")







