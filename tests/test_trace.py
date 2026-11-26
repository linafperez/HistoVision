"""Regression checks for the optional in-memory observer."""
import pathlib,unittest
import numpy as np
from app.servicio import analizar
from app.trazabilidad import RegistroEtapas

ROOT=pathlib.Path(__file__).resolve().parents[1]

class TraceTests(unittest.TestCase):
 @unittest.skipUnless((ROOT/"data").exists(),"Se necesitan los recortes locales de data/.")
 def test_all_categories_identical_with_observation(self):
  folders=sorted((ROOT/'data').glob('*'))
  self.assertEqual(len(folders),7)
  for folder in folders:
   with self.subTest(category=folder.name):
    sample=next(folder.glob('*.png'))
    content=sample.read_bytes()
    before=analizar(content,folder.name,sample.name,registrar_etapas=False)
    after=analizar(content,folder.name,sample.name)
    self.assertEqual({key:after[key] for key in before},before)
    self.assertTrue(after['etapas'])
    self.assertTrue(after['imagenes_etapas'])
    self.assertTrue(all(step['llamadas'] or step['estado']=='no_ejecutada' for step in after['etapas']))
    self.assertEqual(len([step for step in after['etapas'] if step['id'].endswith('-01')]),1)
    for step in after['etapas']:
     for call in step['llamadas']:
      for picture in call['entrada']+call['salida']:
       self.assertIn(picture['imagen_id'],after['imagenes_etapas'])

 def test_snapshot_is_independent_from_in_place_mutation(self):
  from app.trazabilidad import _arrays
  source=np.zeros((8,8),dtype=np.uint8)
  saved=_arrays(source)[0][1]
  source[:]=123
  self.assertFalse(saved.any())

 def test_inactive_context_restores_functions(self):
  import cv2
  original=cv2.bilateralFilter
  with RegistroEtapas(np.zeros((128,128,3),np.uint8),'canine_lung_cancer'):
   self.assertIsNot(cv2.bilateralFilter,original)
  self.assertIs(cv2.bilateralFilter,original)

if __name__=='__main__':unittest.main()
