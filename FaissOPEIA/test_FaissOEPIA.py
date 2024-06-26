import sys, os
sys.path.insert(0, os.environ['PROJECT_ROOT'])

import unittest
import shutil
from FaissOPEIA.carga import carga as carga
# from FaissOepia.ingesta import ingesta as ingesta

class TestFaiss(unittest.TestCase):

    def test_carga(self):
        try:
            origen = os.environ['PROJECT_ROOT'] + "/FaissOPEIA/tests/*.pkl"
            destino = os.environ['PROJECT_ROOT'] + "/FaissOPEIA/db/"
            shutil.copy(origen, destino)
            BDVect = carga()
            retriever = BDVect.getRetriver()
            assert 'retriever' in globals() or 'retriever' in locals()
            os.remove(os.environ['PROJECT_ROOT'] + "/FaissOPEIA/db/bbdd_vecrtorial.pkl")
            os.remove(os.environ['PROJECT_ROOT'] + "/FaissOPEIA/db/retriever.pkl")
        except:
            raise AssertionError



