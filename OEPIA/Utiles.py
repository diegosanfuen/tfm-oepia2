import requests
import pdfplumber
import io, sys, os, yaml
from pathlib import Path
import logging
import datetime


sys.path.insert(0, os.environ['PROJECT_ROOT'])
from Sesiones.sesiones import ManejadorSesiones as ses

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['llm_oepia']['ruta'])
date_today = datetime.datetime.today().strftime("%Y_%m_%d")

# Configuración básica del logger
log_level = None
match config['logs_config']['level']:
    case 'DEBUG':
        log_level = logging.DEBUG
    case 'WARN':
        log_level = logging.WARNING
    case 'WARNING':
        log_level = logging.WARNING
    case 'ERROR':
        log_level = logging.ERROR
    case _:
        log_level = logging.INFO

logging.basicConfig(filename=PATH_BASE / config['logs_config']['ruta_salida_logs'] / f'logs_{date_today}.log',
                    level=log_level,
                    format=config['logs_config']['format'])


class Prompts:
    """
    Clase con métodos Helper que facilitan los prompts al script principal del LLM
    """

    @staticmethod
    def obtenerPROMPTTemplatePrincipalOEPIA():
        """
        Método estatico que devuelve el PROMPT al script principal del LLM
        :return: SystemPrompt para el LLM principal
        """
        return """
            Te llamas OEPIA, y eres un asistente chat, tienes las siguientes misiones importantes:            
            * Ayudar al usuario para encontrar las mejores ofertas de empleo público que coincidan con mi perfil. pero para ello tienes acceso a una base de datos provista por FAISS.
            * Deberás de identificar las oportunidades de empleo público más relevantes que se adapten al perfil de usuario, localizando ofertas de la base de datos provista por FAISS.
            * Proporciona detalles sobre los requisitos y el proceso de solicitud para cada puesto.
            * Ofrece consejos sobre cómo mejorar mi aplicación y aumentar mis posibilidades de éxito.
            * Cuando te pregunten por las ofertas públicas de empleo directamente otroga prioridad al los datos facilitados por la base de datos del RAG.

            * Es importante que los resultados sean precisos y actualizados porque la competencia para puestos de empleo público es alta y los plazos de solicitud suelen ser estrictos. Agradezco tu ayuda en este proceso vital para mi carrera profesional.
            * No te inventes información ni rellenes los datos vacios. Si no tienes ofertas que cumplan el criterio di que no tienes. Como eres un chat amigable :) también tienes la capacidad de reponder a preguntas no relaccionadas con las ofertas de empleo público.

            <context>
            {context}
            </context>

            Question: {input}
            """


class Utiles:
    """
    Clase utilities o herramientas necesarias para la ejecución del script principal del LLM
    """

    @staticmethod
    def obtenerPROMPTTemplatePrincipalOEPIA():
        """
        Método estatico que devuelve el PROMPT al script principal del LLM
        :return:
        """
        return """
            Te llamas OEPIA, y eres un asistente chat, tienes las siguientes misiones importantes:            
            * Ayudar al usuario para encontrar las mejores ofertas de empleo público que coincidan con mi perfil. pero para ello tienes acceso a una base de datos provista por FAISS.
            * Deberás de identificar las oportunidades de empleo público más relevantes que se adapten al perfil de usuario, localizando ofertas de la base de datos provista por FAISS.
            * Proporciona detalles sobre los requisitos y el proceso de solicitud para cada puesto.
            * Ofrece consejos sobre cómo mejorar mi aplicación y aumentar mis posibilidades de éxito.
            * Cuando te pregunten por las ofertas públicas de empleo directamente otroga prioridad al los datos facilitados por la base de datos del RAG.

            * Es importante que los resultados sean precisos y actualizados porque la competencia para puestos de empleo público es alta y los plazos de solicitud suelen ser estrictos. Agradezco tu ayuda en este proceso vital para mi carrera profesional.
            * Es immportante que contestes siempre en Castellano, Catalán o Español
            * No te inventes información ni rellenes los datos vacios. Si no tienes ofertas que cumplan el criterio di que no tienes. Como eres un chat amigable :) también tienes la capacidad de reponder a preguntas no relaccionadas con las ofertas de empleo público.

            <context>
            {context}
            </context>

            Question: {input}
            """

    @staticmethod
    def obtenerCSSOEPIAInterfaz():
        """
        Método estático que devuelve la página de estilos para el interfaz de OEPIA será usado por Gradio
        :return:
        """
        return """
            <style>
                .gr-textbox { 
                    width: 100%; 
                }
            
                .interface-container {
                    background-color: #000; 
                    border-radius: 30px; 
                }
                button {
                    background-color: #4CAF50; 
                    color: dark-blue; 
                    padding: 10px; 
                    border: none; 
                    border-radius: 20px; 
                }
            
                input[type='text'], textarea {
                    border: 2px solid #ddd; 
                    padding: 8px; 
                    font-size: 16px; 
                    font-color: dark-gray
                }
                label {
                    color: #555; 
                    font-weight: bold; 
                    margin-bottom: 5px; 
                }
            
                .output-container {
                    background-color: #DDD; 
                    padding: 15px; 
                    border-radius: 5px; 
                }
            
            </style>
    """

    @staticmethod
    def obtener_boe_texto(url):
        """
        Método estático que es la herramienta a la que llama el AgentePDF utilzando en el script principal del LLM
        :param url: La url del PDF a parsear
        :return: Texto completo del PDF
        """
        response = requests.get(url)

        # Asegurarse de que la solicitud fue exitosa
        if response.status_code == 200:
            # Abrir el PDF desde un stream de bytes
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ''
                # Extraer texto de cada página
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        else:
            print("Error al descargar el archivo")
        return text

    @staticmethod
    def obtener_ayuda_oepia():
        """
        Método estático que devuelve el codigo HTML estático que ayuda a manejar OEPIA
        :param
        :return: HTML de ayuda
        """
        MARKDOWN = """
        ## Manejo de OEPIA:
        
        ### Uso basico:
        Escribe frases que indiequen a OEPIA que quieres localizar como por ejemmplo:
        <pre>Buscame las ofertas de empleo público que tengan que ver con la profesión de arquitecto</pre>
        <pre>Localiza todas las ofertas de empleo en el ambito sanitario y me las presentas en formato json indicando la url.</pre>
        
        ### Metafunciones:
        <pre>@resetear_sesion: Resetea a OEPIA es como si volviese a conocerte</pre>
        <pre>@ver_historial: OEPIA te muestra todo lo que ha conversado contigo</pre>
        
        ### Más información sobre el proyecto
        [Portal RAG OEPIA](http://ragoepia.atwebpages.com/)
        """
        return MARKDOWN

    @staticmethod
    def obtenerPROMPTMemoriaContextoOEPIA(token, k=0):
        """
        Método que devuleve un PROMPT que usa la cadena para mantener la memoria del contexto,
        recibe el parámetro k, que es número de elementos a recuperar de la base de datos de sesiones.
        :param k: Número de elementos a recuperar de la bbd de sesiones k=0 todos
        :param token: token id_sesion
        :return: el prompt de contexto más los elementos recuperados
        """
        try:
            sesiones = ses()
        except Exception as e:
            logger.error(f'Un Error se produjo al intentar cargar la base de datos de sesiones: {e}')
            exit()
        prompt = """
        <context>
        Ten en cuenta la siguiente información como contexto, pero no la incluyas en tus respuestas, 
        si se te solicita una operación concreta omite el conexto: 
        """ + str('\n'.join(sesiones.obtener_mensajes_por_sesion(token, k))) + "</context>\n"
        return prompt

