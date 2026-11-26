"""Checks of real mask dependencies and control flow in explanatory diagrams."""
import unittest
from app.flujo import describir_flujo
from pipelines_histologia._procesamiento import small_holes, grad, corte_espacial
from pipelines_histologia.seleccion import select_body, select_chromatin
from pipelines_histologia._mama import breast_evolution


def _mask_dependency(first, second, allowed, fill):
    result = first | second
    if fill:
        result &= allowed
    return result


class FlowTests(unittest.TestCase):
    def test_mutated_mask_and_both_branch_inputs_reach_output(self):
        flow = describir_flujo(_mask_dependency)
        self.assertTrue(any('Unir máscaras' in n['titulo'] for n in flow['nodos']))
        self.assertTrue(any('intersección' in n['titulo'] for n in flow['nodos']))
        output = next(n['id'] for n in flow['nodos'] if n['tipo']=='output')
        dependencies = [e for e in flow['conexiones'] if e['hasta']==output and e['tipo']=='data']
        self.assertEqual(len(dependencies),2)  # unchanged or intersected result
        labels = {e['etiqueta'] for e in flow['conexiones']}
        self.assertIn('Sí',labels)
        self.assertIn('No',labels)

    def test_real_methods_have_different_conditions_and_cycles(self):
        holes = describir_flujo(small_holes)
        text = '\n'.join(n['expresion'] for n in holes['nodos'])
        self.assertIn('area <= max_area',text)
        self.assertIn('FRAME | ~allowed',text)
        self.assertIn('Repetir',{e['etiqueta'] for e in holes['conexiones']})
        spatial = describir_flujo(corte_espacial)
        self.assertTrue(any('maximum_flow' in n['expresion'] for n in spatial['nodos']))
        self.assertNotEqual(holes['nodos'],spatial['nodos'])
        chrom = '\n'.join(n['expresion'] for n in describir_flujo(select_chromatin)['nodos'])
        body = '\n'.join(n['expresion'] for n in describir_flujo(select_body)['nodos'])
        self.assertIn('best_score + required_gain',chrom)
        self.assertIn('not issues',body)
        self.assertIn("body['score']",body)

    def test_conditional_expression_has_two_explicit_routes(self):
        flow=describir_flujo(grad)
        self.assertEqual(len([n for n in flow['nodos'] if n['tipo']=='condition']),1)
        self.assertIn('Sí',{e['etiqueta'] for e in flow['conexiones']})
        self.assertIn('No',{e['etiqueta'] for e in flow['conexiones']})

    def test_iteration_callback_constraints_are_explicit(self):
        flow=describir_flujo(breast_evolution)
        callback=next(n for n in flow['nodos'] if n['tipo']=='callback')
        self.assertIn('level[excluded] = 0',callback['expresion'])
        self.assertIn('level[chrom] = 1',callback['expresion'])
        self.assertTrue(any(e['desde']==callback['id'] and e['etiqueta']=='constrain'
                            for e in flow['conexiones']))

    def test_native_mask_and_image_inputs_are_separate(self):
        import cv2
        flow=describir_flujo(cv2.grabCut,'grabCut')
        inputs=[n['id'] for n in flow['nodos'] if n['tipo']=='input']
        self.assertEqual(len(inputs),3)
        self.assertTrue(all(any(e['desde']==node and e['tipo']=='data' for e in flow['conexiones'])
                            for node in inputs))


if __name__=='__main__':
    unittest.main()
