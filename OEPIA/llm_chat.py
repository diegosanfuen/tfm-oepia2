import sys
from pathlib import Path
import os, yaml
import datetime
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import logging
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, Agent, AgentExecutor
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.tools.base import BaseTool
from langchain.schema.prompt_template import BasePromptTemplate
from typing import Sequence, Any

# Cargar variables de entorno
load_dotenv()

# Importar clases personalizadas
sys.path.insert(0, os.environ['PROJECT_ROOT'])
from Sesiones.sesiones import ManejadorSesiones as ses
from FaissOPEIA import carga as fcg
from OEPIA.Utiles import Prompts as prompts
from OEPIA.Utiles import Utiles as utls

# Configurar logging
PATH_BASE = Path(os.getenv('PROJECT_ROOT'))
date_today = datetime.datetime.today().strftime("%Y_%m_%d")
logging.basicConfig(
    filename=PATH_BASE / 'logs' / f'logs_{date_today}.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Inicializar el modelo y la memoria
try:
    modelo = os.getenv('LLM_MODEL', 'default_model')  # Valor por defecto si no está definido
    temperature = float(os.getenv('LLM_TEMPERATURE', 0.7))  # Valor por defecto si no está definido
    llm = Ollama(model=modelo, temperature=temperature)
except Exception as e:
    logger.error(f'Error al cargar el modelo: {e}')
    sys.exit()

# Inicializar el manejador de sesiones
try:
    sesiones = ses()
except Exception as e:
    logger.error(f'Error al cargar la base de datos de sesiones: {e}')
    sys.exit()

# Generar token de sesión
token = sesiones.generate_token()

# Configurar la memoria de la conversación
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Crear la plantilla del prompt
prompt_template = ChatPromptTemplate.from_template(prompts.obtenerPROMPTTemplatePrincipalOEPIA())

# Crear la cadena de conversación con memoria
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)


# Herramienta personalizada
def obtener_boe_texto(url: str) -> str:
    # Implementar la lógica para obtener el texto del BOE desde la URL
    return "Texto del BOE"


# Definir las herramientas de LangChain
HERRAMIENTAS = [
    Tool(
        name="ObtenerTextBOE",
        func=obtener_boe_texto,
        description="Obtiene el texto del BOE desde una URL."
    )
]


# Clase personalizada del agente ReAct
class ReActAgent(Agent):
    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        return PROMPT_AGENTE

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        if len(tools) != 1:
            raise ValueError("El número de herramientas es inválido.")

    @property
    def _agent_type(self) -> str:
        return "react"

    @property
    def finish_tool_name(self) -> str:
        return "Finish"

    @property
    def observation_prefix(self) -> str:
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        return "Thought: "


# Crear una instancia del agente
agent = ReActAgent.from_llm_and_tools(
    llm,
    HERRAMIENTAS,
)

# Definir el ejecutor del agente
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=HERRAMIENTAS,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=int(os.getenv('AGENTE_PDF_N_REINTENTOS', 3)),
    return_messages=True,
)

# Crear el chain final combinando la cadena de conversación y el agente
llmApp = conversation_chain | agent_executor


# Función para interactuar con el modelo y mantener la sesión
def interact(user_input):
    global token
    answer = "<h1>SE PRODUJO UN ERROR</h1>"

    if "@resetear_sesion" in user_input.lower():
        token = sesiones.generate_token()
        answer = "Sesión reseteada"
    elif "@ver_historial" in user_input.lower():
        answer = sesiones.obtener_mensajes_por_sesion(token)
    elif "usa el agente para" in user_input.lower():
        try:
            response = agent_executor.run(user_input + " " + str(sesiones.obtener_mensajes_por_sesion(token)))
            answer = str(response)
            sesiones.add_mensajes_por_sesion(token, user_input)
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(answer)
        except Exception as e:
            logger.error(f'Error al invocar el LLM: {e}')
    else:
        try:
            response = llmApp.invoke({"input": user_input, "history": memory.load_memory_variables(token)["history"]})
            answer = str(response)
            sesiones.add_mensajes_por_sesion(token, user_input)
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(answer)
        except Exception as e:
            logger.error(f'Error al invocar el LLM: {e}')

    return answer


# Definir la interfaz de Gradio
with gr.Blocks() as iface:
    with gr.Row():
        texto_entrada = gr.Textbox(label="Ingresa tu mensaje", placeholder="Escribe aquí...", lines=10)
        historial_previo = gr.Textbox(label="Historial", value="", visible=False)

    texto_entrada.change(fn=format_links, inputs=texto_entrada, outputs=historial_previo)


def interactuar_con_llm(texto, historial_previo):
    global history
    historial_previo = historial_previo + str(history)
    texto_limpio = texto.strip()
    respuesta = interact(texto_limpio)

    html_wrapper = f"""
    <div class="container">
        <details>
            <summary>Historial {datetime.datetime.today().strftime('%H:%M:%S')}</summary>
            <div class="content">
                <p>{history}</p>
            </div>
        </details>
    </div>
    """

    if historial_previo:
        nuevo_historial = f"\n<h3><u>USUARIO:</h3></u><pre> {texto_limpio}</pre>\n\n<h3><u>OEPIA:</u></h3> <div><p>{respuesta}</p></div><br><br>{html_wrapper}\n\n"
    else:
        nuevo_historial = f"\n<h3><u>USUARIO:</u></h3><pre> {texto_limpio}</pre>\n\n<h3><u>OEPIA:</u></h3> <div><p>{respuesta}</p></div>\n\n"

    history = nuevo_historial
    return nuevo_historial


iface = gr.Interface(
    fn=interactuar_con_llm,
    inputs=[texto_entrada, historial_previo],
    outputs=gr.Markdown(label="Historial de la conversación"),
    title="<img src ='https://diegosanfuen.github.io/staticsTFM/logo/logo.png' />OEPIA: La IA especializada en ofertas de Empleo Público",
    description="Escribe un mensaje y presiona 'Submit' para interactuar con el modelo de lenguaje.",
    live=False,
    css=CSS,
    article=utls.obtener_ayuda_oepia(),
    thumbnail=True,
    allow_flagging="manual",
    flagging_options=["Incorrecto", "Irrelevante", "Ofensivo"],
    flagging_dir="flagged_data",
)

iface.launch(share=True)
