"""Selección de contornos a partir de la evidencia del recorte."""
from .calidad import body_evidence, chrom_evidence


def inspect_record(source, record):
    chrom = chrom_evidence(source, record['chromatin'])
    body = body_evidence(source, record['cell'], record['chromatin'])
    issues = []
    if not chrom['area']:
        issues.append('cromatina_vacía')
    if not body['area']:
        issues.append('cuerpo_vacío')
    if body['borde_efectivo_px'] < 28 or body['borde_exterior'] < .30 or body['cuerpo_extra_nuclear'] < .25:
        issues.append('envolvente_sin_apoyo')
    if body['vecino'] > .13 or body['claro'] > .10 or body['fuga'] > .30:
        issues.append('intrusión_o_fragmentación')
    if body['prolongacion'] > .29 or (body['fracción_distal'] > .35 and
            body['vecino'] > .018 and body['contraste_exterior'] < .08):
        issues.append('prolongación_o_vecindad_dudosa')
    if body['discordancia'] > .28:
        issues.append('discordancia_extensa')
    return chrom, body, issues


def select_body(source, records):
    """Order estimates by contour evidence and image-derived consistency."""
    ratings = []
    for record in records:
        _, body, issues = inspect_record(source, record)
        ratings.append((not issues, body['score']))
    return records[max(range(len(records)), key=ratings.__getitem__)]


def select_chromatin(source, records, required_gain=.028):
    """The ordered proposals share a focus hypothesis; require a clear stain improvement to change it."""
    reference = records[0]
    current = chrom_evidence(source, reference['chromatin'])
    best = reference
    best_score = current['score']
    for proposal in records[1:]:
        evidence = chrom_evidence(source, proposal['chromatin'])
        if not evidence['area']:
            continue
        if evidence['score'] >= best_score + required_gain:
            best, best_score = proposal, evidence['score']
    return best
