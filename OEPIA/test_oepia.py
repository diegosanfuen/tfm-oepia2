import unittest
from Utiles import Prompts as promt
from Utiles import Utiles as utls

# Clase de test
class TestOepia(unittest.TestCase):

    def test_PROMPTTemplatePrincipalOEPIA(self):
        self.assertEqual(len(promt.obtenerPROMPTTemplatePrincipalOEPIA() > 1), True)

    def test_PROMPTTemplatePrincipalOEPIA(self):
        self.assertEqual(len(utls.obtenerPROMPTTemplatePrincipalOEPIA() > 1), True)

    def test_obtenerCSSOEPIAInterfaz(self):
        self.assertEqual(len(utls.obtenerCSSOEPIAInterfaz() > 1), True)

    def test_obtener_boe_texto(self):
        self.assertEqual(len(utls.obtener_boe_texto("https://www.boe.es//boe/dias/2024/05/23/pdfs/BOE-A-2024-10363.pdf") > 1), True)

     def test_obtener_ayuda_oepia(self):
         self.assertEqual(len(utls.obtener_ayuda_oepia() > 1), True)

     def test_obtenerPROMPTMemoriaContextoOEPIA(self):
         self.assertEqual(len(utls.obtenerPROMPTMemoriaContextoOEPIA('1234567890acbd') > 1), True)

# Ejecutar los tests
if __name__ == '__main__':
    unittest.main()