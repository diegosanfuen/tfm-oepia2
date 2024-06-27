import sys, os
sys.path.insert(0, os.environ['PROJECT_ROOT'])

import glob

import unittest
import shutil
from Sesiones.sesiones import ManejadorSesiones as ManejadorSesiones

class TestSesiones(unittest.TestCase):

    def test_generate_token(self):
        try:
            sesiones = ManejadorSesiones()
            token1 = sesiones.generate_token()
            token2 = sesiones.generate_token()
            assert token1 != token2
            token3 = sesiones.generate_token(length=15)
            assert len(token3) == 15
        except:
            raise AssertionError

    def test_probar_sesiones(self):
        try:
            sesiones = ManejadorSesiones()
            token = "1234567890acbd"
            sesiones.add_mensajes_por_sesion(token, str(f"PruebaMensage: test"))
            assert len(sesiones.obtener_mensajes_por_sesion(token)) > 0
        except:
            raise AssertionError
