# Importamos librerias necesarias
import sys
from pathlib import Path
import os, yaml
import datetime
import logging
import re
from dotenv import load_dotenv
import gradio as gr
from typing import Sequence, Any
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import Tool, Agent, AgentOutputParser, AgentExecutor
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.tools.base import BaseTool
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate  # Asegúrate de importar esto

# Cargamos las variables de entorno
load_dotenv()
sys.path.insert(0, os.environ['PROJECT_ROOT'])

from Sesiones.sesiones import ManejadorSesiones as ses
from FaissOPEIA import carga as fcg
from OEPIA.Utiles import Prompts as prompts
from OEPIA.Utiles import Utiles as utls

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
date_today = datetime.datetime.today().strftime("%Y_%m_%d")

# Configuración básica del logger
log_level = getattr(logging, config['logs_config']['level'].upper(), logging.INFO)
logging.basicConfig(filename=PATH_BASE / config['logs_config']['ruta_salida_logs'] / f'logs_{date_today}.log',
                    level=log_level,
                    format=config['logs_config']['format'])

logger = logging.getLogger()

# CSS para Gradio
CSS = utls.obtenerCSSOEPIAInterfaz()

# Configuración del modelo
try:
    modelo = config['llm_oepia']['parameters_modelo']['llm_model']
    temperature = config['llm_oepia']['parameters_modelo']['temperature']
    assistant_name = config['llm_oepia']['parameters_modelo']['nombre_asistente']
    llm = Ollama(model=modelo, temperature=temperature)
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar el modelo {modelo} : {e}')
    exit()

# Inicializar sesiones
try:
    sesiones = ses()
except Exception as e:
    logger.error(f'Un Error se produjo al intentar cargar la base de datos de sesiones: {e}')
    exit()

# Generar token de sesión
token = ses.generate_token()
prompt_template = ChatPromptTemplate.from_template(prompts.obtenerPROMPTTemplatePrincipalOEPIA())
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever_inst = fcg()
retriever_faiss = retriever_inst.inialize_retriever()
retrieval_chain = create_retrieval_chain(retriever_faiss, document_chain)


# Definimos las herramientas de langchain
def obtener_boe_texto(url):
    return utls.obtener_boe_texto(url)


HERRAMIENTAS = [
    Tool(
        name="ObtenerTextBOE",
        func=obtener_boe_texto,
        description="Trae el texto del BOE dado una URL al PDF que contiene la oferta de empleo",
    )
]

AGENTE_FEW_SHOT_EJEMPLOS = [
    """
    Question: ¿Obtén el BOE del enlace que te paso?
    Thought: Necesito localizar la url proporcionada por el usuario, la localizamos y es 
    https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf
    Action: ObtenerTextBOE["https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf"]
    Observation: "texto del BOE": Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ....    
    Action: Finish["El BOE contiene: Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ...."]
    """
]

AGENTE_FEW_SHOT_EJEMPLOS.extend([
    """
    Question: Descarga el BOE del texto anterior
    Thought: Necesito localizar del contexto la url proporcionada por el usuario, la localizamos y es 
    https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf
    Action: ObtenerTextBOE["https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf"]
    Observation: "texto del BOE": Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ....    
    Action: Finish["El BOE contiene: Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ...."]
""",
    """
    Question: Descarga el enlace al documento proporcionado 
    Thought: Necesito localizar la url proporcionada por el usuario, la localizamos y es 
    https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf
    Action: ObtenerTextBOE["https://www.boe.es/boe/dias/2024/05/02/pdfs/BOE-A-2024-8838.pdf"]
    Observation: "texto del BOE": Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ....    
    Action: Finish["El BOE contiene: Resolución de 25 de abril de 2024, del Instituto de la Cinematografía y ...."]
    """,
])

SUFIJO = """
    \nEres un sistema inteligente realizando una serie de pensamientos y ejecutando acciones para poder responder la pregunta del usuario.
    Pero es importante que sólo debes de usar este agente, cuando te solicitan o detectas que hay que descargar un BOE o un BOCYL.
    Cada acción es una llamada a una función: ObtenerTextBOE(url: str): str
    Por favor, entrega la respuesta sin usar caracteres que puedan causar problemas de parsing como comillas dobles o comas.
    Puedes usar la función cuando consideres necesario. Cada acción se realiza por separado. Contesta siempre en castellano. 
    Después sigue procesando la petición del usuario con las demás ordenes

    Vamos a empezar

    Question: {input}
    {agent_scratchpad}
"""

PROMPT_AGENTE = PromptTemplate.from_examples(
    examples=AGENTE_FEW_SHOT_EJEMPLOS,
    suffix=SUFIJO,
    input_variables=["input", "agent_scratchpad"],
)


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
            raise ValueError("The number of tools is invalid.")

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


agent = ReActAgent.from_llm_and_tools(llm, HERRAMIENTAS)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=HERRAMIENTAS,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=config['agentePDF']['n_reintentos'],
    return_messages=True,
)

llmApp = retrieval_chain | agent_executor


def chat(pregunta):
    global token
    answer = "<h1>SE PRODUJO UN ERROR</h1>"
    if "@resetear_sesion" in pregunta.lower():
        token = ses.generate_token()
        answer = "Sesión reseteada"
    elif "@ver_historial" in pregunta.lower():
        answer = sesiones.obtener_mensajes_por_sesion(token)
    else:
        try:
            response = llmApp.invoke({"input": pregunta,
                                      "context": str(sesiones.obtener_mensajes_por_sesion(token))})
            answer = str(response['answer'])
            sesiones.add_mensajes_por_sesion(token, str(pregunta))
            sesiones.add_mensajes_por_sesion(token, answer)
            logger.info(answer)
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar invocar el LLM: {e}')
    return answer


history = ""


def interactuar_con_llm(texto, historial_previo):
    global history
    historial_previo = historial_previo + str(history)
    texto_limpio = texto.strip()
    respuesta = chat(texto_limpio)
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
    inputs=[gr.Textbox(label="Ingresa tu mensaje", placeholder="Escribe aquí...", lines=10),
            gr.Textbox(label="Historial", value="", visible=False)],
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
