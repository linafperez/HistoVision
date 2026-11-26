"""Assets must cover the real operation inventory and remain presentation figures."""
from pathlib import Path
import json
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


class FigureTests(unittest.TestCase):
 def test_every_operation_has_a_composed_figure(self):
  catalog=json.loads((ROOT/'assets/methods/catalog.json').read_text(encoding='utf8'))
  operations={node.get('data-operation') for path in (ROOT/'assets/diagrams').glob('*.svg')
              for node in ET.fromstring(path.read_text(encoding='utf8')).iter() if node.get('data-operation')}
  self.assertEqual(set(catalog),operations)
  self.assertEqual(len(operations),85)
  for ident,item in catalog.items():
   with self.subTest(operation=ident):
    tree=ET.fromstring((ROOT/item['figure']).read_text(encoding='utf8'))
    blocks=[node for node in tree.iter() if node.get('data-method-block') is not None]
    self.assertGreaterEqual(len(blocks),2)
    self.assertLessEqual(len(blocks),6)
    selected=[node for node in tree.iter() if node.get('data-selected')=='true']
    self.assertEqual([node.get('data-stage-operation') for node in selected],[ident])
    self.assertLessEqual(len([node for node in tree.iter() if node.get('data-parameter-slot') is not None]),2)
    self.assertFalse(any(node.get('role')=='button' or node.tag.endswith(('script','foreignObject')) for node in tree.iter()))
    labels=' '.join(tree.itertext())
    for word in ('previous','prepared','return','pair_evidence','contour5','details','mode'):
     self.assertNotRegex(labels,r'\b'+word+r'\b')
    if item.get('formula'):
     formula=ET.fromstring((ROOT/item['formula']['image']).read_text(encoding='utf8'))
     self.assertTrue(any(node.tag.endswith('path') for node in formula.iter()))

 def test_different_methods_keep_distinct_presentations(self):
  catalog=json.loads((ROOT/'assets/methods/catalog.json').read_text(encoding='utf8'))
  self.assertNotEqual(catalog['mastocitoma-06']['profile'],catalog['mama-05']['profile'])
  self.assertNotEqual(catalog['neuroendocrino-06']['formula'],catalog['sarcoma-04']['formula'])
  self.assertNotEqual(catalog['pulmon-08']['profile'],catalog['pulmon-10']['profile'])


if __name__=='__main__':unittest.main()
