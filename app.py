import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
# --- Nuevas importaciones para GitHub ---
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Agente con GitHub", page_icon="🐙")
st.title("Mi Agente con acceso a GitHub 🐙")

st.sidebar.header("Configuración")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
github_token = st.sidebar.text_input("GitHub Access Token", type="password") # Pide el token de GitHub

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="¡Hola! Puedo buscar en internet y leer tus repositorios de GitHub. ¿Qué hacemos hoy?")]

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    if not openai_api_key or not github_token:
        st.info("Por favor, añade tus API Keys de OpenAI y GitHub en la barra lateral.")
        st.stop()

    # Configurar variable de entorno requerida por el Wrapper de LangChain
    os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_token

    # --- Inicializar Herramientas de GitHub ---
    github = GitHubAPIWrapper()
    github_toolkit = GitHubToolkit.from_github_api_wrapper(github)
    github_tools = github_toolkit.get_tools() # Esto trae herramientas como GetIssue, CreatePR, etc.

    search_tool = DuckDuckGoSearchRun()
    
    # Juntamos todas las herramientas
    tools = [search_tool] + github_tools

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente técnico. Puedes buscar en internet y usar la API de GitHub para revisar repositorios, issues y archivos."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.chat_message("assistant"):
        with st.spinner("Trabajando..."):
            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": st.session_state.messages[:-1]
            })
            st.write(response["output"])
            st.session_state.messages.append(AIMessage(content=response["output"]))
