import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime, sys
import string
from urllib.parse import urlparse, urlunparse
import re
from pathlib import Path
import logging, os, yaml, time

# Continuar con el resto de tu código de sumy después de esta descarga
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Idioma del texto
language = "spanish"

# Introducir esta variable de entorno en el lanzador
# os.environ['PROJECT_ROOT'] = r'/content/tfm-oepia'

# Abrir y leer el archivo YAML
with open(Path(os.getenv('PROJECT_ROOT')) / 'config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

PATH_BASE = Path(config['ruta_base'])
directorio_proyecto = os.path.dirname(Path(PATH_BASE) / config['scrapping']['ruta'])
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

# Creamos el logger
logger = logging.getLogger()


class DescargaBOCyL:
    """
    Clase que permite la descarga del BOCyL en lo referente a las Resoluciones relacionadas con las convocatorias de Oposiciones
    Para instanciar la clase:
    MiClase = DescargaBOCyL()
    Para fijar el Offset
    MiClase.establecer_offset(offset)
    """


    def __init__(self):
        """
        Generador de la clase no recibe parámetros
        establece las variables internas
        fecha_actual, url_patron, dominio u dataset con los boes
        """
        # Obtiene la fecha y hora actual
        self.fecha_actual = datetime.datetime.now()
        self.url_patron = string.Template(config['scrapping']['fuentes']['BOCYL']['patron'])
        self.dominio = config['scrapping']['fuentes']['BOCYL']['url']
        self.dataset_bocyls = pd.DataFrame({'url': [],
                                            'titulo': [],
                                            'texto': [],
                                            'resumen': [],
                                            })
        logger.info("-------------------------------------------------------------------------------------")
        logger.info("-----------------------------------OBTENCION DE DATOS BOCYL-----------------------------")
        logger.info("-------------------------------------------------------------------------------------")

        self.folder_paquete = config['scrapping']['ruta']
        self.folder_data = config['scrapping']['descarga_datos']
        self.folder_paquete = config['scrapping']['ruta']
        self.name_file_output = config["scrapping"]["fuentes"]["BOCYL"]["fichero_csv"]
        self.separator_name = config["scrapping"]["fuentes"]["separador"]
        self.limit = config["scrapping"]["fuentes"]["limitacion_descargas"]
        self.time_wait = config["scrapping"]["fuentes"]["tiempo_entre_descargas"]
        self.headers = config['scrapping']['headers']
        self.timeout = config['scrapping']['timeout']

        # Idioma del texto
        self.language = "spanish"

        # Número de oraciones en el resumen
        self.num_sentences = config['scrapping']['n_sentences_summary']

    def quitar_etiquetas_html(self, cadena_html: str) -> str:
        """
        Método Helper para la eliminación de etiquetas HTML de los textos parseados
        uso:
        Entrada: Texto con etiquetas HTML
        Salida: Mismo Texto sin etiquetas HTML
        self.quitar_etiquetas_html(Texto)
        """
        # Parsear la cadena HTML
        soup = BeautifulSoup(cadena_html, 'html.parser')
        # Obtener solo el texto sin etiquetas HTML
        texto = soup.get_text(separator='')
        texto = texto.replace('[', '')
        texto = texto.replace(']', '')
        texto = re.sub('<.*?>', '', texto)
        return texto

    def generar_resumen(self, texto: str) -> str:
        """
        Genera un resumen del texto teniendo en cuenta la parametrizacion n_sentences_summary y lo devuelve como salida
        :param texto: Texto a resumir
        :return: Texto resumido
        self.generar_resumen(texto)
        """

        # Inicializar el parser
        parser = PlaintextParser.from_string(texto, Tokenizer(self.language))
        stemmer = Stemmer(self.language)

        # Inicializar el sumarizador
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)

        try:
            # Generar el resumen
            summary = summarizer(parser.document, self.num_sentences)
        except Exception as e:
            logger.error(f"Hubo un problema al realizar el resumen {e}")

        texto_resumido = ""
        for sentence in summary:
            if texto_resumido == "":
                texto_resumido = str(sentence)
            else:
                texto_resumido = texto_resumido + "\n" + str(sentence)

        return texto_resumido


    def establecer_offset(self, offset: int):
        """
        Método que estalece el OFFSET definido como el número de días a partir de la fecha
        actual desde la que se quiere descargar los BOES
        Si instanciamos
        MiClase.establecer_offset(5)
        Inspeccionaremos los BOES de hace 5 días
        Entrada: Offset Es un entero
        Salida: Variables internas de la clase (URLS de los BOES)
        """
        fecha_calculada = self.fecha_actual - datetime.timedelta(days=offset)
        anio = fecha_calculada.year
        mes = str(fecha_calculada.month).zfill(2)
        dia = str(fecha_calculada.day).zfill(2)
        fecha = {'anio': anio,
                 'mes': mes,
                 'dia': dia}
        self.url_busqueda = self.url_patron.substitute(anio=fecha['anio'],
                                                       mes=fecha['mes'],
                                                       dia=fecha['dia'])

    def buscar_urls_xmls(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos las URLS
        de las disposiciones relativas a las ofertas de empelo público es decir 
        Sección II B del BOE
        Uso
        self.buscar_urls_xmls()
        """

        url = self.url_busqueda
        parsed_url = urlparse(url)

        dominio = parsed_url.netloc


        response = requests.get(url)
        html_content = response.content

        soup = BeautifulSoup(html_content, 'html.parser')

        regex = re.compile(".*otros formatos.*")

        enlaces_con_titulo = []
        for enlace in soup.find_all('a'):
            if regex.search(str(enlace)):
                enlaces_con_titulo.append(enlace)

        lista_urls = []
        for enlace in enlaces_con_titulo:
            # Realizamos una serie de transformaciones a la URL
            href_transformado = str(enlace["href"]).replace(
                'html', 'xml').replace(
                'do', 'xml')

            url_obtenida = f'https://{dominio}/{href_transformado}'

            parsed_url = urlparse(url_obtenida)

            parsed_url_lista = list(parsed_url)
            path_url = parsed_url_lista[2].split('/')
            path_url[1] = 'boletines'
            parsed_url_lista[2] = "/".join(path_url)

            # Convertir la lista de nuevo a un objeto ParseResult
            parsed_url_modificada = urlparse(urlunparse(parsed_url_lista))
            url_obtenida = urlunparse(parsed_url_modificada)
            lista_urls.append(url_obtenida)

        self.lista_urls = lista_urls

    def obtener_lista_xmls(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos los XMLs
        de las disposiciones relativas a las ofertas de empelo público es decir 
        Sección II B del BOE
        Uso
        self.obtener_lista_xmls()
        """
        lista_respuestas = []
        for url in self.lista_urls:
            # url = 'https://bocyl.jcyl.es/boletines/2024/04/29/xml/BOCYL-D-29042024-1.xml'
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
            except requests.exceptions.ConnectTimeout as e:
                logger.error(f"La conexión ha excedido el tiempo máximo de espera. {e}")

            lista_respuestas.append(response.text)
        self.lista_xmls = lista_respuestas

    def obtener_lista_titulos(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos los titulos
        de las disposiciones relativas a las ofertas de empelo público es decir 
        Sección II B del BOE
        Uso
        self.obtener_lista_titulos()
        """
        lista_titulos = []
        for XML in self.lista_xmls:
            soup = BeautifulSoup(XML, "xml")
            titulo = soup.find("titulo")
            lista_titulos.append(titulo.get_text())
        self.lista_titulos = lista_titulos

    def obtener_lista_textos(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos los textos
        de las disposiciones relativas a las ofertas de empelo público es decir 
        Sección I B del BOCyL
        Uso
        self.obtener_lista_textos()
        """
        lista_textos = []
        for XML in self.lista_xmls:
            textos = ""
            soup = BeautifulSoup(XML, "xml")
            text = soup.find_all("texto")
            lista_textos.append(str(text))
        self.lista_textos = lista_textos

    def obtener_lista_urls_pdf(self):
        """
        Con los parámetros obtenidos de establecer_offset, localizamos las urls pdfs
        de las disposiciones relativas a las ofertas de empelo público es decir 
        Sección II B del BOE
        Uso
        self.obtener_lista_urls_pdf()
        """
        lista_urls_pdf = []
        for XML in self.lista_xmls:
            textos = ""
            soup = BeautifulSoup(XML, "xml")
            url_pdf = soup.find_all("url_pdf")
            lista_urls_pdf.append(f'{self.dominio}{str(self.quitar_etiquetas_html(str(url_pdf)))}')
        self.lista_urls_pdf = lista_urls_pdf

    def dividir_texto_en_chunks(self, texto,
                                longitud_chunk=config["scrapping"]["max_chunk_length"]):
        return [texto[i:i + longitud_chunk] for i in range(0, len(texto), longitud_chunk)]

    def generar_dataset(self) -> int:
        """
        Con los parámetros obtenidos de establecer_offset, generamos el dataset pandas
        de las disposiciones relativas a las ofertas de empelo público es decir 
        Sección II B del BOE
        Uso
        self.generar_dataset()
        Salida: Conteo de filas del dataset
        """
        self.buscar_urls_xmls()
        self.obtener_lista_xmls()
        self.obtener_lista_titulos()
        self.obtener_lista_textos()
        self.obtener_lista_urls_pdf()
        dataset_capturado = pd.DataFrame({'url': self.lista_urls_pdf,
                                          'titulo': self.lista_titulos,
                                          'texto': self.lista_textos})

        dataset_capturado['texto'] = dataset_capturado['texto'].apply(self.quitar_etiquetas_html)
        # dataset_capturado['resumenW'] = dataset_capturado['texto'].apply(self.generar_recorte)

        filas_expandidas = []
        for index, row in dataset_capturado.iterrows():
            chunks = self.dividir_texto_en_chunks(row['texto'])
            for chunk in chunks:
                nueva_fila = row.to_dict()
                nueva_fila['resumenW'] = chunk
                filas_expandidas.append(nueva_fila)

        dataset_capturado = pd.DataFrame(filas_expandidas)
        texto_separador = "\nURL: "
        try:
            dataset_capturado['resumen'] = dataset_capturado.apply(
                lambda row: f"{row['resumenW']}{texto_separador}{row['url']}", axis=1)
            dataset_capturado.drop('resumenW', axis=1, inplace=True)
            dataset_capturado['texto'] = ['' for i in range(len(dataset_capturado))]
        except Exception as e:
            logger.error(f"No existen BOCyLs para el día de hoy {e}")
            dataset_capturado = pd.DataFrame({'url': [],
                                              'titulo': [],
                                              'texto': [],
                                              'resumen': [],
                                            })


        self.dataset_bocyls = pd.concat([self.dataset_bocyls, dataset_capturado], ignore_index=True)
        return self.dataset_bocyls.shape[0]

    def obtener_dataset_final(self):
        """
        Finalmente devolvemos a la rutina principal el contenido del dataset completo
        MiClase.obtener_dataset_final()
        Salida: Dataset Completo
        """
        return self.dataset_bocyls

    def guardar_dataset_final(self):
        """
        Guarda en formato CSV en la ruta indicada en el fichero de configuracion
        MiClase.guardar_dataset_final()
        """
        fecha_actual = datetime.datetime.now()
        sufijo_fecha = fecha_actual.strftime("%Y%m%d")
        self.dataset_bocyls.to_csv(
            f'{directorio_proyecto}/{self.folder_paquete}/{self.folder_data}/{self.name_file_output}_{sufijo_fecha}.csv',
            sep=self.separator_name)

    def initialize_download(self):
        """
        Método que ejecuta toda la cadena de procesos para descargase los BOCyLs y guardarlo en
        formato csv en la ruta y con las configuraciones del config.xml

        Ejemplo de uso de la clase
        MiObjeto = DescargaBOCyL
        MiObjeto.initialize_download()
        """
        if config['scrapping']['enabled_scapping'] == 0:
            logger.info("La descarga via webscrapping fue deshabilitada")
            sys.exit()

        i = 0

        while True:
            self.establecer_offset(i)
            if (self.generar_dataset() > self.limit):
                break
            time.sleep(self.time_wait)
            i += 1
        self.guardar_dataset_final()
