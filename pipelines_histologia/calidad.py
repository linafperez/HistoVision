"""Evidencia de imagen y controles descriptivos de segmentación."""
import cv2
import numpy as np

CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _fraction(numerator, denominator):
    return float(numerator) / max(1, int(denominator))


def _ring(mask):
    return mask & ~cv2.erode(mask.astype(np.uint8), CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _distance(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _gray_gradient(channel, sigma):
    sample = cv2.GaussianBlur(channel, (0, 0), sigma)
    return cv2.magnitude(cv2.Scharr(sample, cv2.CV_32F, 1, 0),
                         cv2.Scharr(sample, cv2.CV_32F, 0, 1))


def _rescale(x):
    a, b = np.quantile(x, [.04, .97])
    return np.clip((x - a) / max(float(b - a), 1e-5), 0, 1)


def _labels(mask):
    count, ids, stats, centroid = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return ids, stats[1:count, cv2.CC_STAT_AREA], centroid[1:count]


def evidence(rgb):
    """Prepare shared physical cues once, regardless of the evaluated proposal."""
    filtered = cv2.bilateralFilter(rgb, 5, 17, 3)
    lab = cv2.cvtColor(filtered.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    light = lab[..., 0] / 100
    redness = lab[..., 1] / 65
    yellow = lab[..., 2] / 65
    r, g, b = np.moveaxis(filtered.astype(np.float32), -1, 0)
    br = (100*b / (1+r+g))*(256/(1+r+g+b))
    raw_dark = .55 * (1-light) + .45 * np.clip(br/135, 0, 1)
    contrast = np.maximum(0, raw_dark-cv2.GaussianBlur(raw_dark,(0,0),5.0))
    dark = .76 * _rescale(raw_dark) + .24 * _rescale(contrast)
    threshold, _ = cv2.threshold((dark*255).astype(np.uint8),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    dense = dark >= max(.56, threshold/255 + .10)
    groups, areas, centers = _labels(dense)
    yy, xx = np.indices(light.shape)
    edge_color = cv2.magnitude(_gray_gradient(redness, 1.5), _gray_gradient(yellow, 1.5))
    edge_light = _gray_gradient(light, 1.7)
    chroma = np.hypot(redness, yellow)
    pale = (light > .92) & (chroma < .14)
    edge = .55*_rescale(edge_color) + .45*_rescale(edge_light)
    edge = cv2.dilate(edge, CROSS)
    return dict(light=light,dark=dark,raw_dark=raw_dark,contrast=contrast,dense=dense,
                group=groups,areas=areas,centers=centers,xx=xx,yy=yy,pale=pale,lab=lab,
                either_edge=edge,edge_floor=max(.36,float(np.quantile(edge,.60))),
                frame=(yy==0)|(xx==0)|(yy==rgb.shape[0]-1)|(xx==rgb.shape[1]-1),
                center=(np.array(rgb.shape[1::-1])-1)/2)


def chrom_evidence(source, chrom):
    """Look for stain contrast, unsupported fringes, omitted connected material and other foci."""
    area = int(chrom.sum())
    if not area:
        return dict(area=0, contraste=0., soporte=0., borde_débil=1., posible_omisión=1.,
                    fracción_periférica=0., mediana_centro_px=None, score=-1.)
    distances = _distance(chrom)
    surround = (distances >= 1) & (distances <= 5)
    raw_dark, dark = source['raw_dark'], source['dark']
    mid = float(np.median(raw_dark[chrom]))
    contrast = mid-float(np.median(raw_dark[surround])) if surround.any() else 0.
    rim = _ring(chrom) & ~source['frame']
    strong = source['dense'] & (distances <= 4)
    outside_close = strong & ~chrom
    close = outside_close & (dark >= np.quantile(dark[chrom], .35))
    ids = source['group']
    if close.any():
        overlaps = np.bincount(ids[chrom], minlength=len(source['areas'])+1)
        valid = np.flatnonzero(overlaps >= 5)
        valid = valid[valid != 0]
        close &= np.isin(ids, valid)
    omitted = _fraction(close.sum(), area + close.sum())
    weak = _fraction((rim & (dark < .52)).sum(), rim.sum())
    support = float(np.median(dark[chrom]))
    centroid = np.array([source['xx'][chrom].mean(),source['yy'][chrom].mean()])
    location = float(np.linalg.norm(centroid-source['center']))
    at_frame = _fraction((rim & cv2.dilate(source['frame'].astype(np.uint8), CROSS).astype(bool)).sum(),rim.sum())
    # Penalize isolated dark cores that explain little of their connected staining.
    available = (source['dense'] | (dark >= .57)) & (distances <= 6)
    ids, sizes, _ = _labels(available | chrom)
    overlap = np.unique(ids[chrom])
    same = np.isin(ids, overlap) & available & ~chrom
    incompleteness = max(omitted,.72*_fraction(same.sum(), area + same.sum()))
    score = (.48*support + .30*np.clip(contrast/.22,0,1) - .24*incompleteness
             - .18*weak - .12*min(location/55,1.5) - .18*at_frame)
    return dict(area=area, contraste=contrast, soporte=support, borde_débil=weak,
                posible_omisión=incompleteness, fracción_periférica=at_frame,
                mediana_centro_px=location, score=float(score))


def body_evidence(source, body, chrom):
    """Measure the entire candidate contour with the same image cues for every method."""
    area = int(body.sum())
    outside = int((chrom & ~body).sum())
    mismatch = _fraction(outside,chrom.sum())
    if not area:
        return dict(area=0, fuera=outside, discordancia=mismatch, borde=0., borde_exterior=0.,
                    contraste_exterior=0., borde_efectivo_px=0, cuerpo_extra_nuclear=0.,
                    vecino=0., claro=0., fuga=1., prolongacion=0., fracción_distal=0., marco=False, partes=0, score=-1.)
    rim = _ring(body)
    away = _distance(chrom) > 3 if chrom.any() else np.ones(body.shape,bool)
    evaluable = rim & away & ~source['frame']
    strong_rim = evaluable & (source['either_edge']>=source['edge_floor'])
    exterior = _fraction(evaluable.sum(),rim.sum())
    contour = _fraction(strong_rim.sum(),evaluable.sum())
    dist_in, dist_out = _distance(~body), _distance(body)
    inner = (dist_in > 1) & (dist_in <= 4) & away
    outer = (dist_out > 1) & (dist_out <= 4) & away
    lab = source['lab'] / np.array([100., 60., 60.],dtype=np.float32)
    color_jump = float(np.linalg.norm(np.median(lab[inner],axis=0)-np.median(lab[outer],axis=0))) if inner.any() and outer.any() else 0.
    shell = body & ~chrom
    cytoplasm = _fraction(shell.sum(), area)
    bordering = bool((body & source['frame']).any())
    rival_parts = source['areas']
    group = source['group']
    body_touch = np.bincount(group[body],minlength=len(rival_parts)+1)[1:]
    origin_touch = np.bincount(group[chrom],minlength=len(rival_parts)+1)[1:]
    cuts = (rival_parts>=22) & (body_touch>=np.maximum(9,.25*rival_parts)) & (origin_touch<np.maximum(4,.12*rival_parts))
    if chrom.any():
        # The one-pixel dark ridge on the same focus is included in chromatin QA.
        proximity = _distance(chrom)[np.round(source['centers'][:,1]).astype(int),
                                     np.round(source['centers'][:,0]).astype(int)]
        cuts &= proximity>=5
    captured = float(body_touch[cuts].sum()) / max(28,area)
    pale = _fraction((body & source['pale']).sum(),area)
    _, parts, _ = _labels(body)
    fragment = _fraction(parts.sum()-parts.max(),area)
    extension, distal_fraction = 0., 0.
    if chrom.any():
        distance_core = _distance(chrom)
        # Compare distal support across directions around this proposal's own chromatin.
        # A broad ring is distinct from a long extension in only one direction.
        distal = body & (distance_core > max(18.,1.7*np.sqrt(chrom.sum()/np.pi)))
        if distal.any():
            dy = source['yy'][distal] - source['yy'][chrom].mean()
            dx = source['xx'][distal] - source['xx'][chrom].mean()
            angle = np.arctan2(dy,dx)
            sectors = []
            for turn in (0.,np.pi/8):
                bins = np.floor(((angle + turn) % (2*np.pi)) / (np.pi/4)).astype(int)
                sectors.append(np.bincount(bins,minlength=8).max())
            extension = _fraction(max(sectors),area)
            distal_fraction = _fraction(distal.sum(),area)
    score = (.35*contour*np.sqrt(exterior) + .19*np.clip(color_jump/.25,0,1) +
             .20*min(cytoplasm/.64,1) + .10*exterior - .50*mismatch -
             .65*min(captured,.65) - .45*pale - .12*bordering - .20*fragment - .55*extension)
    return dict(area=area, fuera=outside, discordancia=mismatch, borde=contour, borde_exterior=exterior,
                contraste_exterior=color_jump, borde_efectivo_px=int(evaluable.sum()),
                cuerpo_extra_nuclear=cytoplasm, vecino=captured, claro=pale, fuga=fragment,
                prolongacion=extension, fracción_distal=distal_fraction, marco=bordering, partes=int(len(parts)), score=float(score))


def sensitivity(mask, changed):
    """Symmetric change under a small, deterministic input perturbation."""
    return _fraction((mask ^ changed).sum(),(mask | changed).sum())


def quality(source, result, perturbed=None, ambiguity=0):
    body = body_evidence(source,result['cell'],result['chromatin'])
    chrom = chrom_evidence(source,result['chromatin'])
    dch = None if perturbed is None else sensitivity(result['chromatin'],perturbed['chromatin'])
    dce = None if perturbed is None else sensitivity(result['cell'],perturbed['cell'])
    geom_body = body['borde_efectivo_px'] < 25 or body['borde_exterior'] < .33
    signal_body = body['cuerpo_extra_nuclear'] < .22 or (body['borde'] < .30 and body['contraste_exterior'] < .08)
    statuses = {
        'cromatina_insuficiente': not chrom['area'] or chrom['posible_omisión'] > .25 or
                                 (chrom['contraste'] < .045 and chrom['borde_débil'] > .30),
        'posible_material_vecino': body['vecino'] >= .045 or body['prolongacion'] >= .20 or ambiguity >= 1,
        'discordancia_mascaras': body['fuera'] > 0,
        'limite_celular_debil': body['borde'] < .42 or geom_body,
        'sensibilidad_excesiva': None if perturbed is None else dch > .23 or dce > .32,
        'estructura_truncada': body['marco'] or bool((result['chromatin'] & source['frame']).any()),
        'evidencia_celular_insuficiente': (geom_body or signal_body or not body['area'] or
                                             float(np.quantile(np.hypot(source['lab'][...,1],
                                                                       source['lab'][...,2]),.85)) < 3.),
    }
    # Numeric diagnostics retain the distinction between descriptive scores and warnings.
    return dict(alertas={k:None if v is None else bool(v) for k,v in statuses.items()},
                cromatina=chrom, cuerpo=body,
                sensibilidad={'cromatina':dch,'cuerpo':dce}, competidores_ambiguos=int(ambiguity))
