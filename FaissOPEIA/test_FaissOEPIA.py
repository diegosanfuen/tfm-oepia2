import sys, os
sys.path.insert(0, os.environ['PROJECT_ROOT'])

import glob

import unittest
import shutil
from FaissOPEIA.carga import carga as carga
# from FaissOepia.ingesta import ingesta as ingesta

class TestFaiss(unittest.TestCase):

    def test_carga(self):
        try:
            origenes = os.environ['PROJECT_ROOT'] + "/FaissOPEIA/tests"
            destino = os.environ['PROJECT_ROOT'] + "/FaissOPEIA/db/"
            for origen in glob.glob(origenes + "/*.pkl"):
                shutil.copy(origen, destino)
            BDVect = carga()
            retriever = BDVect.getRetriver()
            assert 'retriever' in globals() or 'retriever' in locals()
            os.remove(os.environ['PROJECT_ROOT'] + "/FaissOPEIA/db/bbdd_vecrtorial.pkl")
            os.remove(os.environ['PROJECT_ROOT'] + "/FaissOPEIA/db/retriever.pkl")
        except:
            raise AssertionError



