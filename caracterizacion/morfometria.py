import cv2
import numpy as np

LABELS = (
    ("area_celular", "Área celular estimada", "px²"),
    ("area_cromatinica", "Área cromatínica", "px²"),
    ("area_citoplasmatica", "Área citoplasmática estimada", "px²"),
    ("eje_mayor", "Eje mayor", "px"),
    ("eje_menor", "Eje menor", "px"),
    ("circularidad", "Circularidad", ""),
    ("relacion_cromatina_celula", "Relación cromatina/célula", "%"),
    ("componentes_cromatinicos", "Componentes cromatínicos", ""),
)


def caracterizar(celular, cromatinica):
    """Mide máscaras en su malla computacional.

    Elipse equivalente por segundos momentos centrales de todos los píxeles,
    con corrección de 1/12 por la extensión de cada píxel cuadrado. Longitudes
    completas de ejes = 4 sqrt(autovalor). Perímetro por Crofton de cuatro
    direcciones, incluidos huecos y todas las componentes.
    """
    from skimage.measure import perimeter_crofton
    cell, chrom = np.asarray(celular), np.asarray(cromatinica)
    if cell.ndim != 2 or cell.shape != chrom.shape or cell.dtype != bool or chrom.dtype != bool:
        raise ValueError("Se requieren dos máscaras booleanas bidimensionales de igual tamaño.")
    area, stain = int(cell.sum()), int(chrom.sum())
    values = dict(area_celular=area, area_cromatinica=stain,
                  area_citoplasmatica=int((cell & ~chrom).sum()),
                  eje_mayor=None, eje_menor=None, circularidad=None,
                  relacion_cromatina_celula=None, componentes_cromatinicos=0)
    if chrom.size:
        count, _, _, _ = cv2.connectedComponentsWithStats(chrom.astype(np.uint8), connectivity=8)
        values["componentes_cromatinicos"] = int(count - 1)
    if area:
        moments = cv2.moments(cell.astype(np.uint8), binaryImage=True)
        covariance = np.array([[moments["mu20"], moments["mu11"]],
                               [moments["mu11"], moments["mu02"]]]) / area
        axes = 4 * np.sqrt(np.maximum(0, np.linalg.eigvalsh(covariance) + 1/12))
        values["eje_menor"], values["eje_mayor"] = map(float, axes)
        perimeter = float(perimeter_crofton(cell, directions=4))
        if perimeter > 0:
            values["circularidad"] = float(4*np.pi*area/perimeter**2)
        if not (chrom & ~cell).any():
            values["relacion_cromatina_celula"] = float(100*stain/area)
    return [{"id": key, "nombre": name, "valor": values[key], "unidad": unit}
            for key, name, unit in LABELS]
