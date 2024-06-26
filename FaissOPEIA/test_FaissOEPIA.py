import sys, os
sys.path.insert(0, os.environ['PROJECT_ROOT'])

import unittest
import shutil
from FaissOepia.carga import carga as carga
# from FaissOepia.ingesta import ingesta as ingesta

class TestFaiss(unittest.TestCase):

    def test_carga(self):
        try:
            origen = os.environ['PROJECT_ROOT'] + "/FaissOEPIA/test/*"
            destino = os.environ['PROJECT_ROOT'] + "/FaissOEPIA/db/"
            shutil.copy(origen, destino)
            BDVect = carga()
            retriever = BDVect.getRetriver()
            assert 'retriever' in globals() or 'retriever' in locals()
            os.remove(os.environ['PROJECT_ROOT'] + "/FaissOEPIA/db/*")
        except:
            raise AssertionError



