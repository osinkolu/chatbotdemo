import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.tools import tool
from datetime import datetime
# import dotenv
import os

# dotenv.load_dotenv()

openai_api_key=os.environ.get("openai_api_key")
groq_api_key=os.environ.get("groq_api_key")
claude_api_key=os.environ.get("claude_api_key")

search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


prompt = hub.pull("hwchase17/openai-functions-agent")


@tool
def time_tool(query: str) -> str:
    """
    A tool that tells the current time. Use this when a user asks for the current time.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time is {current_time}"

tools = [search,wikipedia, time_tool]

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if st.button("clear chat"):
    st.session_state.memory = []
    st.session_state.chat_memory = []
    
st.session_state.selected_model = st.selectbox("Choose Model to use.", ("Meta's Llama"))

print("Model is ",st.session_state.selected_model)
st.title("Victor Demo BOT")

if st.session_state.selected_model == "OpenAI's GPT":
    model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    model = model.bind_tools(tools)

elif st.session_state.selected_model == "Anthropic's Claude":
    model = ChatAnthropic(model='claude-3-opus-20240229',api_key = claude_api_key)
    model = model.bind_tools(tools)
    
elif st.session_state.selected_model == "Meta's Llama":
    model = ChatGroq(model='llama-3.1-70b-versatile',api_key = groq_api_key)


chain = RunnablePassthrough.assign(
                agent_scratchpad = lambda x: format_to_tool_messages(x["intermediate_steps"])
                ) | prompt | model | ToolsAgentOutputParser()
agent_executor = AgentExecutor(agent=chain, tools=tools,
                                           handle_parsing_errors=True,
                                           return_intermediate_steps=False)

for chat in st.session_state.chat_memory:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

if prompt:= st.chat_input("what is moving?"):
    with st.chat_message("user"):
        st.session_state.memory.append(HumanMessage(content=prompt))
        st.markdown(prompt)
        st.session_state.chat_memory.append({"role":"user", "message":prompt})

    with st.chat_message("ai"):
        response = agent_executor.invoke({"input": prompt, "chat_history":st.session_state.memory}, return_only_outputs=True)["output"]
        if st.session_state.selected_model == "Anthropic's Claude":
            response = response[0]["text"].split("</thinking>")[-1]
        st.markdown(response)
        st.session_state.memory.append(AIMessage(content=response))
        st.session_state.chat_memory.append({"role":"ai", "message":response})