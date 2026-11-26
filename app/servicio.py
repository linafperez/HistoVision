"""Validación de entrada y coordinación del análisis local."""
import base64
import hashlib
import io
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pipelines_histologia import segmentar, CATEGORIAS
from caracterizacion import caracterizar

ADVERTENCIAS = {
 "cromatina_insuficiente": "La delimitación cromatínica podría ser incompleta.",
 "posible_material_vecino": "Puede haberse incluido material de una estructura vecina.",
 "discordancia_mascaras": "Hay cromatina fuera de la región celular. La relación cromatina/célula no está disponible.",
 "limite_celular_debil": "La delimitación celular presenta incertidumbre. Interprete sus dimensiones como estimaciones.",
 "sensibilidad_excesiva": "La segmentación cambia ante pequeñas variaciones de la imagen.",
 "estructura_truncada": "Una estructura toca el borde del recorte. Sus dimensiones pueden estar truncadas.",
 "evidencia_celular_insuficiente": "La imagen ofrece evidencia insuficiente para delimitar el cuerpo celular.",
}


def validar_imagen(content):
    if not content or len(content) > 20 * 1024 * 1024:
        raise ValueError("Seleccione un PNG de hasta 20 MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG":
                raise ValueError("El formato admitido es PNG.")
            if image.mode not in ("RGB", "RGBA"):
                raise ValueError("El PNG debe tener tres canales RGB o cuatro canales RGBA.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Seleccione un PNG estático de una sola imagen.")
            width, height = image.size
            if width < 8 or height < 8 or max(width, height) > 4096:
                raise ValueError("Las dimensiones deben estar entre 8 y 4096 píxeles por eje.")
            if width != height:
                raise ValueError("Seleccione un recorte cuadrado. Las imágenes rectangulares requieren un recorte previo.")
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.mode == "RGBA":
                if image.getextrema()[3] != (255, 255):
                    raise ValueError("El PNG contiene transparencia. Seleccione un recorte RGB opaco.")
                image = image.convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("El archivo PNG está dañado o no se puede decodificar.") from error
    notices = []
    if width != 128:
        notices.append(f"Entrada de {width} × {height} px remuestreada a 128 × 128 px. Las mediciones corresponden a esta malla.")
        rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA if width > 128 else cv2.INTER_LINEAR)
    return rgb, [width, height], notices


def _png(rgb):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("No se pudo generar la visualización.")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def analizar(content, categoria, nombre, *, registrar_etapas=True):
    if categoria not in CATEGORIAS:
        raise ValueError("Seleccione uno de los siete tipos tumorales.")
    rgb, original_size, notices = validar_imagen(content)
    trace = None
    if registrar_etapas:
        from .trazabilidad import RegistroEtapas
        trace = RegistroEtapas(rgb, categoria)
        with trace:
            result = segmentar(rgb, categoria, evaluar_sensibilidad=True)
    else:
        result = segmentar(rgb, categoria, evaluar_sensibilidad=True)
    cell, chrom = result["mascara_celular"], result["mascara_cromatinica"]
    visualizations = [rgb]
    for mask, tint, alpha in ((cell, (0, 220, 240), .35), (chrom, (255, 220, 0), .52)):
        overlay = rgb.astype(np.float32).copy()
        overlay[mask] = (1-alpha)*overlay[mask] + alpha*np.asarray(tint)
        visualizations.append(np.rint(overlay).astype(np.uint8))
    notices.extend(text for key, text in ADVERTENCIAS.items()
                   if result["indicadores"]["alertas"].get(key))
    response = dict(nombre=str(nombre), categoria=categoria, resolucion=[128,128],
                resolucion_original=original_size, metodo=result["metodo"],
                caracteristicas=caracterizar(cell, chrom), advertencias=notices,
                indicadores=result["indicadores"],
                huellas_mascaras={key: hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
                                  for key, mask in (("celular", cell), ("cromatinica", chrom))},
                imagenes=[_png(image) for image in visualizations])

    if trace is not None:
        from pathlib import Path

        slug = CATEGORIAS[categoria][1:]
        diagram = next((Path(__file__).resolve().parents[1] / "assets" / "diagrams").glob("*_"+slug+".svg"))
        response["etapas"] = trace.exportar(diagram.read_text(encoding="utf8"))
        response["imagenes_etapas"] = trace.images
        response["flujos_etapas"] = trace.flows
    return response

