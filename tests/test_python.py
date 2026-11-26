import unittest,io
import numpy as np
from PIL import Image
from caracterizacion import caracterizar
from app.servicio import validar_imagen,analizar

def encode(image):
    stream=io.BytesIO();Image.fromarray(image).save(stream,format="PNG");return stream.getvalue()

class MorfometriaTests(unittest.TestCase):
    def test_eight_measurements_and_empty(self):
        empty=np.zeros((128,128),bool)
        result={r['id']:r['valor'] for r in caracterizar(empty,empty)}
        self.assertEqual(len(result),8)
        self.assertEqual(result['area_celular'],0)
        self.assertIsNone(result['circularidad'])
        self.assertIsNone(result['eje_mayor'])
        self.assertIsNone(result['relacion_cromatina_celula'])

    def test_set_difference_and_spatial_consistency(self):
        cell=np.zeros((16,16),bool);cell[3:11,3:11]=True
        chrom=np.zeros_like(cell);chrom[7:13,7:13]=True
        values={r['id']:r['valor'] for r in caracterizar(cell,chrom)}
        self.assertEqual(values['area_citoplasmatica'],48)
        self.assertIsNone(values['relacion_cromatina_celula'])

    def test_moments_rectangle_and_diagonal_connectivity(self):
        cell=np.zeros((32,32),bool);cell[10:16,8:20]=True
        chrom=np.zeros_like(cell);chrom[11,11]=True;chrom[12,12]=True
        values={r['id']:r['valor'] for r in caracterizar(cell,chrom)}
        self.assertAlmostEqual(values['eje_mayor'],4*np.sqrt(12**2/12))
        self.assertAlmostEqual(values['eje_menor'],4*np.sqrt(6**2/12))
        self.assertEqual(values['componentes_cromatinicos'],1)
        self.assertAlmostEqual(values['relacion_cromatina_celula'],200/72)

    def test_disk_and_fragmentation(self):
        yy,xx=np.indices((128,128));circle=(xx-64)**2+(yy-64)**2<=20**2
        values={r['id']:r['valor'] for r in caracterizar(circle,np.zeros_like(circle))}
        self.assertTrue(.92<values['circularidad']<1.05)
        self.assertTrue(39<values['eje_mayor']<41)
        fragmented=circle.copy();fragmented[:4,:4]=True
        self.assertTrue(all(r['valor'] is not None for r in caracterizar(fragmented,circle)))

class InputTests(unittest.TestCase):
    def test_invalid(self):
        for data in (b'bad', encode(np.zeros((16,20,3),np.uint8)),encode(np.zeros((16,16),np.uint8)),
                     encode(np.zeros((16,16,4),np.uint8))):
            with self.assertRaises(ValueError):validar_imagen(data)

    def test_square_resampling(self):
        rgb,shape,warnings=validar_imagen(encode(np.zeros((256,256,3),np.uint8)))
        self.assertEqual(rgb.shape,(128,128,3))
        self.assertEqual(shape,[256,256])
        self.assertTrue(warnings)
        with self.assertRaises(ValueError):analizar(b'bad','unknown','x.png')

if __name__=="__main__":unittest.main()
