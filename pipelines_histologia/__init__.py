"""Segmentación automática de recortes de figuras mitóticas RGB."""
from importlib import import_module

import cv2
import numpy as np

from .calidad import _labels, evidence, quality

CATEGORIAS = {
    'canine_cutaneous_mast_cell_tumor': '_mastocitoma',
    'canine_lung_cancer': '_pulmon',
    'canine_lymphoma': '_linfoma',
    'canine_soft_tissue_sarcoma': '_sarcoma',
    'human_breast_cancer': '_mama',
    'human_melanoma': '_melanoma',
    'human_neuroendocrine_tumor': '_neuroendocrino',
}


def _normalizar_rgb(rgb):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ValueError('Se requiere una imagen RGB H × W × 3 de al menos 8 × 8.')
    if image.dtype.kind == 'f':
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Las imágenes RGB de punto flotante deben estar en [0, 1].')
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        image = np.rint(image.astype(np.float32) * (255 / 65535)).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError('Use RGB uint8, uint16 o punto flotante en [0, 1].')
    if image.shape[:2] != (128, 128):
        mode = cv2.INTER_AREA if min(image.shape[:2]) >= 128 else cv2.INTER_LINEAR
        image = cv2.resize(image, (128, 128), interpolation=mode)
    return np.ascontiguousarray(image)


def segmentar(rgb, categoria, *, evaluar_sensibilidad=True):
    """Procesa un recorte RGB con el evento de interés cerca del centro.

    Devuelve máscaras booleanas en la cuadrícula de entrada, indicadores y método.
    La evaluación de sensibilidad repite el método con un cambio pequeño de luz y enfoque.
    Las áreas y distancias de los indicadores se miden en la cuadrícula de 128 × 128.
    """
    if categoria not in CATEGORIAS:
        raise ValueError('Categoría desconocida. Consulte CATEGORIAS.')
    image = _normalizar_rgb(rgb)
    source = evidence(image)
    route = import_module('.' + CATEGORIAS[categoria], __package__).segmentar_recorte
    if np.ptp(source['light']) < .022 and np.ptp(source['lab'][...,1:]) < 2.:
        blank = np.zeros((128,128),dtype=bool)
        record = {'cell':blank, 'chromatin':blank, 'omitted':blank}
        method = 'Sin evidencia en el recorte'
    else:
        record, method = route(image, source)
    perturbation = None
    if evaluar_sensibilidad:
        if not record['cell'].any() and not record['chromatin'].any() and method == 'Sin evidencia en el recorte':
            perturbation = record
        else:
            softened = cv2.GaussianBlur(image, (3,3), .42)
            softened = cv2.addWeighted(softened, .986, softened, 0, 2.5)
            perturbation, _ = route(softened, evidence(softened))
    omitted = record.get('omitted',np.zeros((128,128),bool)) & ~record['chromatin']
    proximity = cv2.dilate(record['cell'].astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))).astype(bool)
    ids, areas, _ = _labels(omitted)
    ambiguity = int(np.count_nonzero((areas >= 18) &
                    (np.bincount(ids[proximity],minlength=len(areas)+1)[1:] >= np.maximum(7,.20*areas))))
    indicators = quality(source, record, perturbation, ambiguity)
    shape = np.asarray(rgb).shape[:2]
    indicators['cuadricula_evaluacion'] = [128,128]
    indicators['remuestreada'] = shape != (128,128)
    outputs = []
    for key in ('cell','chromatin'):
        mask = record[key]
        if shape != (128,128):
            mask = cv2.resize(mask.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
        outputs.append(mask.astype(bool,copy=True))
    return dict(mascara_celular=outputs[0], mascara_cromatinica=outputs[1],
                indicadores=indicators, metodo=method)
