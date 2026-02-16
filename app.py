import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
import wikipedia


wikipedia.set_user_agent("MiAgenteLangchain/1.0 (https://github.com/tu-usuario/tu-repo; tu_correo@ejemplo.com)")
# --- 1. Herramienta Personalizada del Notebook ---
@tool
def conchita_coins(input: float) -> float:
    """Use this tool to convert USD to Conchita Academy coins"""
    return 1.3*(float(input))

# --- 2. Funciones de Memoria del Notebook ---
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def ensure_string_output(agent_result: dict) -> dict:
    output_value = agent_result.get('output')
    if isinstance(output_value, list):
        concatenated_text = ""
        for item in output_value:
            if isinstance(item, dict) and item.get('type') == 'text':
                concatenated_text += item.get('text', '')
            elif isinstance(item, str):
                concatenated_text += item
        agent_result['output'] = concatenated_text
    elif not isinstance(output_value, str):
        agent_result['output'] = str(output_value)
    return agent_result

# --- 3. Interfaz de Streamlit ---
st.set_page_config(page_title="Agente Conchita EDEM", page_icon="🎓")
st.title("Agente LangChain EDEM 🎓")

st.sidebar.header("Configuración")
google_api_key = st.sidebar.text_input("Google API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="¡Hola! Puedo buscar en Wikipedia, DuckDuckGo, y convertir USD a Conchita coins. ¿En qué te ayudo?")]

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    if not google_api_key:
        st.info("Por favor, añade tu Google API Key en la barra lateral para continuar.")
        st.stop()

    # --- 4. Lógica del Agente (basada en el Notebook) ---
    # Modelo
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=google_api_key)
    
    # Herramientas
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search, wikipedia, conchita_coins]

    # Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Based on user query and the chat history, look for information using DuckDuckGo Search and Wikipedia and then give the final answer"),
        ("placeholder", "{history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Creación del Agente usando langchain_classic
    agent = create_tool_calling_agent(chat, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Añadiendo la memoria
    agent_executor_with_formatted_output = agent_executor | RunnableLambda(ensure_string_output)
    agent_with_history = RunnableWithMessageHistory(
        agent_executor_with_formatted_output,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # --- 5. Ejecución ---
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            # Usamos un session_id fijo para este usuario en Streamlit
            response = agent_with_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "sess1"}}
            )
            
            output_text = response["output"]
            st.write(output_text)
            st.session_state.messages.append(AIMessage(content=output_text))
