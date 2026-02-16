import streamlit as st
import wikipedia
from datetime import datetime
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

# --- CONFIGURACIÓN PARA EVITAR BLOQUEOS DE WIKIPEDIA ---
wikipedia.set_user_agent("AgenteStreamlitEDEM/1.0 (https://github.com/tu-usuario/tu-repo; alumno@edem.es)")

# --- 1. HERRAMIENTA PERSONALIZADA ---
@tool
def conchita_coins(input: float) -> float:
    """Use this tool to convert USD to Conchita Academy coins"""
    return 1.3 * float(input)

# --- 2. FUNCIONES DE MEMORIA ---
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

# --- 3. INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Agente Inteligente", page_icon="🤖")
st.title("Mi Agente LangChain 🤖")

st.sidebar.header("Configuración")
google_api_key = st.sidebar.text_input("Google API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="¡Hola! Puedo buscar en internet, en Wikipedia y convertir USD a Conchita coins. ¿En qué te ayudo hoy?")
    ]

# Mostrar historial de mensajes en la pantalla
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

# --- 4. INTERACCIÓN Y LÓGICA DEL AGENTE ---
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Guardar y mostrar el mensaje del usuario
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    if not google_api_key:
        st.info("Por favor, añade tu Google API Key en la barra lateral para continuar.")
        st.stop()

    # Inicializar Modelo y Herramientas
    chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=google_api_key)
    search = DuckDuckGoSearchRun()
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search, wiki_tool, conchita_coins]

    # Obtener fecha actual para evitar "ceguera temporal"
    fecha_actual = datetime.now().strftime("%A, %d de %B de %Y")

    # Crear el Prompt con inyección de contexto
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""Eres un asistente útil, experto y muy persistente. 
        Hoy es {fecha_actual}. Usa esta fecha exacta como tu referencia en el tiempo para saber qué eventos son pasados, actuales o futuros (próximos).
        Si usas DuckDuckGo y no encuentras la respuesta a la primera, vuelve a buscar usando términos o palabras clave diferentes antes de rendirte.
        Basado en la consulta del usuario y el historial de chat, busca la información necesaria y da una respuesta final clara y directa."""),
        ("placeholder", "{history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Ensamblar el Agente
    agent = create_tool_calling_agent(chat, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Conectar el Agente con la Memoria
    agent_executor_with_formatted_output = agent_executor | RunnableLambda(ensure_string_output)
    agent_with_history = RunnableWithMessageHistory(
        agent_executor_with_formatted_output,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Ejecutar y mostrar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando e investigando..."):
            response = agent_with_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "sesion_unica_usuario"}}
            )
            
            output_text = response["output"]
            st.write(output_text)
            st.session_state.messages.append(AIMessage(content=output_text))
