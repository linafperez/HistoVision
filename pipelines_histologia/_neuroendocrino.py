"""Pipeline RGB consolidado para neuroendocrino."""


from ._procesamiento import (
    CENTER,
    FRAME,
    GLOBAL_FILTERS,
    MCP_Geometric,
    RADIAL,
    SEED,
    XX,
    YY,
    anchor_region,
    appearance6,
    associate5,
    attached,
    attached_clean,
    audited_result,
    base_maps,
    informacion_cromatinica,
    body_from_regions,
    corte_en_banda,
    boundary_rivals,
    breadth_first_order,
    choose_anchor,
    class_probabilities,
    cluster_options,
    color,
    foco_compacto,
    territorios_por_coste,
    components,
    contour5,
    coverage6,
    cv2,
    candidatos_oscuros,
    distance,
    ell,
    modelos_de_apariencia,
    expand,
    expit,
    energia_orientada,
    field_means,
    filters,
    ajustar_mezcla,
    grad,
    jerarquia_regional,
    interval,
    landscape_markers,
    log_densidad_mezcla,
    logsumexp,
    made,
    grafo_multicanal,
    mapas_apariencia,
    marker_image,
    maximum_flow,
    mean_shift_centers,
    merge_hierarchy,
    morphology,
    ndi,
    np,
    pair_evidence,
    parcel_graph,
    parcel_means6,
    foco_central,
    random_walker_regions,
    reject_union,
    scale,
    segmentation,
    shrink,
    small_holes,
    sparse,
    corte_espacial,
    split_components,
    spsolve,
    threshold_result,
    u8,
    upper_contacts,
    v4_contacts,
    mapas_tincion_multiescala,
    v4_means,
    v4_parcels,
    mapas_region_tincion,
    formar_resultado_regional,
)
from .seleccion import select_body, select_chromatin
from .calidad import chrom_evidence


def protected_watershed(rgb, markers):
    work = markers.copy()
    cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), work)
    return np.maximum(work, 0)


def preparar_random_walker(maps, mode, step=0):
    s = maps['response']
    response = .50 * s['H normalizada'] + .34 * s['Blue Ratio'] + .16 * s['Contraste local']
    tissue = maps['tissue']
    if mode == 'K-means espacial y Random Walker':
        _, p = cluster_options(maps, 'K-means', 5, 'LabHE+xy')
        high = (response >= .58 + .014 * step) & (p[..., :2].sum(axis=-1) >= .41)
        weak = (response >= .48 + .014 * step) & tissue & (p[..., :3].sum(axis=-1) >= .49)
    else:
        top, bounds, _ = threshold_result(response, 'Multi-Otsu')
        level = float(bounds[-1]) if len(bounds) else .62
        high = (response >= level + .014 * step) & tissue
        weak = (response >= level - .085 + .014 * step) & tissue
    candidate = attached(weak, high)
    parts = split_components(candidate, response, minimum=13)
    buffer = expand(FRAME, 2) | shrink(~tissue)
    markers = marker_image(parts, buffer)
    return response, candidate, parts, markers


def segmentar_random_walker(maps, prepared, mode, step=0):
    response, candidate, parts, markers = prepared
    vectors = np.dstack((maps['color_vector'], 0.35 * maps['xy']))
    parcels = v4_parcels(vectors, n=510)
    if mode == 'Multi-Otsu y Watershed':
        owner = protected_watershed(maps['smooth'], markers)
    else:
        owner, p = random_walker_regions(parcels, vectors, maps['edge4'], markers, 6.0 + 0.5 * step)
    primary = choose_anchor(parts, response, 30)
    empty = np.zeros(owner.shape, bool)
    if primary is None:
        return made(empty, empty, omitted=candidate, candidate=candidate, seeds=markers, territory=owner, domain=empty)
    anchor_idx = primary + 2
    base = owner == anchor_idx
    touching = expand(base, 2)
    accept, uncertain = ([primary], [])
    for i, part in enumerate(parts):
        if i == primary:
            continue
        gap, path = interval(parts[primary], part)
        territory = owner == i + 2
        contact = touching & territory | expand(territory, 2) & base
        midpoint = np.linalg.norm((np.argwhere(part).mean(axis=0) + np.argwhere(parts[primary]).mean(axis=0)) / 2 - CENTER)
        edge = np.quantile(maps['edge4'][contact], 0.6) if contact.any() else 1.0
        valley = np.quantile(response[path], 0.4) if path[0].size else 0.0
        seam = expand(parts[primary]) & part
        continuous = mode == 'Multi-Otsu y Random Walker' and seam.sum() >= 4 and (np.quantile(response[seam], 0.3) >= 0.59) and (np.quantile(maps['edge4'][seam], 0.65) < 0.42)
        if gap <= 10 and midpoint < 17 and (contact.sum() >= 6) and (edge < 0.44 and valley >= 0.35 or continuous):
            accept.append(i)
        elif gap < 13 and midpoint < 20:
            uncertain.append(i)
    domain = np.isin(owner, np.array(accept) + 2)
    competitors = (owner >= 2) & ~domain
    chrom = reject_union([parts[i] for i in accept], owner.shape)
    cell = attached_clean(domain, chrom, maps['tissue'] & ~competitors & ~FRAME, 8)
    omitted = reject_union([parts[i] for i in uncertain], owner.shape) | chrom & ~cell
    return made(cell, chrom & cell, competitors, omitted, candidate=candidate, seeds=markers, territory=owner, domain=domain, preconstraint=chrom.copy())


def preparar_territorios_neuro(maps, mode, step=0):
    if mode == 'Jerarquía de H-minima':
        field = jerarquia_regional(maps, 'H-minima', use_texture=False, limit=.425, step=step)
    elif mode == 'División multiescala y nivel vectorial':
        field = jerarquia_regional(maps, 'Felzenszwalb', use_texture=True, limit=.445, step=step, use_outer=True)
    else:
        field = jerarquia_regional(maps, 'H-maxima de espacio libre', use_texture=True, limit=.435, step=step, use_outer=True)
    score = .44 * maps['h'] + .29 * maps['ratio'] + .21 * maps['od'] + .06 * maps['local']
    independent = anchor_region(field['labels'], .58 * score + .42 * maps['e'], maps['tissue'])
    return score, field, independent


def segmentar_territorios_neuro(maps, prepared, mode, step=0):
    score, field, independent = prepared
    # Fused dark material is split only on large masks with distinct distance peaks.
    groups, candidate, strong = candidatos_oscuros(score, maps, .80, .129, 970, step)
    labels = field['labels']
    primary = foco_central(groups, score, labels, independent, 32, maps['l'])
    blank = np.zeros(score.shape, bool)
    if primary is None:
        return audited_result(blank, blank, uncertain=candidate, field=field, candidate=candidate, initial=blank)
    selected, unsure = associate5(maps, score, groups, primary, field, 13, .53, .14)
    chrom = reject_union([groups[idx] for idx in selected], score.shape)
    pending = reject_union([groups[idx] for idx in unsure], score.shape)
    competing, rival_cores = boundary_rivals(labels, groups, selected + unsure)
    core = np.bincount(labels[groups[primary]], minlength=int(labels.max()) + 1)
    accepted = np.bincount(labels[chrom], minlength=len(core))
    core[0], accepted[0] = 0, 0
    chosen = (accepted >= max(8, .21 * chrom.sum()))
    if core.any():
        chosen[core.argmax()] = True
    claim = chosen[labels]
    zone = body_from_regions(field, maps, claim, rival_cores, .375 + .014 * step, 2)
    admissible = maps['tissue'] & (~competing | claim) & ~FRAME
    if mode == 'Territorios conjuntos de cromatina y tejido':
        ownership, markers, costs = territorios_por_coste(maps, field, groups, selected, claim, step)
        competing = ownership > 2
        result = (ownership == 2) & maps['tissue']
        claim = result.copy()
        admissible = maps['tissue'] & ~competing
        field = {**field, 'competitive': ownership}
    elif mode == 'División multiescala y nivel vectorial':
        result = contour5(maps, zone, admissible, shrink(claim, 2), field['edge'],
                          field['vector'], 12 + 2 * step, prototype_count=2)
    else:
        result = zone & admissible
    cell = small_holes(attached(result, claim), admissible, 9)
    return audited_result(cell, chrom, competing, pending, field=field, candidate=candidate,
                          initial=claim, strong=strong)


def nucleo_por_intensidad(maps, previous):
    chromatin = previous['chromatin']
    signal = .40 * maps['ratio'] + .36 * maps['od'] + .24 * (1 - maps['l'])
    result = chromatin.copy()
    anchor = foco_compacto(previous)
    core_table = []
    for _, group, area, _ in components(chromatin, 1):
        # Review weak low-staining fringe only when a large component spans distinct intensity modes.
        values = signal[group]
        if area < 400 or np.quantile(values, .90) - np.quantile(values, .15) < .22:
            continue
        high = np.quantile(values, .76)
        level, _ = cv2.threshold(u8(values), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        intrinsic = attached(group & (signal > level / 255 - .075), group & (signal >= high))
        removed = group & ~intrinsic
        dark_edge = (group & ~shrink(group)) & removed
        if intrinsic.sum() >= .66 * area and dark_edge.sum() >= .22 * removed.sum():
            result = (result & ~group) | intrinsic
        core_table.append((area, int(intrinsic.sum()), float(level / 255), int(dark_edge.sum())))
    return result, previous.get('candidate', chromatin) | chromatin, previous['omitted'] | (chromatin & ~result), previous['excluded'] & ~result, core_table


def nucleo_por_densidad(maps, previous):
    old = previous['chromatin']
    signal = .5*maps['od'] + .34*maps['ratio'] + .16*maps['h']
    robust = old.copy()
    rows = []
    for _, group, area, _ in components(old, 1):
        if area < 740:
            continue
        medium = cv2.GaussianBlur(signal,(0,0), .8)
        local = medium - cv2.GaussianBlur(medium,(0,0), 4.5)
        values = medium[group]
        reference = np.quantile(values, .4)
        strong = group & (medium >= np.quantile(values, .75))
        retained = attached(group & ((medium >= reference) | (local > .034)), strong)
        if .66 <= retained.sum()/area <= .94:
            robust = (robust & ~group) | retained
        rows.append((area, int(retained.sum()), round(float(reference),3)))
    return robust, previous.get('candidate', old) | old, previous['omitted'] | (old & ~robust), previous['excluded'] & ~robust, rows


def full_chromatin(record):
    chrom = record.get('preconstraint',record['chromatin'])
    return dict(record,chromatin=chrom,preconstraint=chrom,excluded_by_cell=chrom & ~record['cell'])


def new_context(maps, initial):
    anchor = foco_compacto(initial)
    if not anchor.any():
        coords = np.argwhere(initial['cell'])
        if len(coords):
            yy,xx = coords[np.argmin(np.sum((coords-CENTER)**2,axis=1))]
        else:
            yy,xx = np.unravel_index(np.argmax(maps['ratio']-.01*RADIAL),anchor.shape)
        anchor[yy,xx] = True
    return dict(maps=maps,previous=initial,base=informacion_cromatinica(initial),anchor=anchor,perturb=0)


def with_detection(record, detection, method):
    chrom,candidate,omitted,rivals,_ = detection
    return dict(record,chromatin=chrom,candidate=candidate,omitted=omitted,excluded=rivals,method=method)


def label_body(raw, detection, initial, method, graph=None, model=None):
    # Core contact and hole filling apply to new region estimates. Established bodies are kept intact here.
    has_graph = graph is not None
    result = formar_resultado_regional(raw,detection,initial,graph,model,cleanup=has_graph)
    result['method'] = method
    return result


def segmentar_recorte(rgb, source):
    terrain = mapas_region_tincion(rgb,use_texture=True)
    mode = 'Territorios conjuntos de cromatina y tejido'
    initial = segmentar_territorios_neuro(terrain,preparar_territorios_neuro(terrain,mode),mode)
    maps = mapas_apariencia(rgb)
    ctx = new_context(maps,initial)
    first = dict(initial,method='territorios de tejido y tinción')
    options = [first]
    core = with_detection(initial,nucleo_por_intensidad(maps,initial),'intensidad intracomponente')
    dense = with_detection(initial,nucleo_por_densidad(maps,initial),'núcleo denso')
    options.extend(proposal for proposal in (core,dense)
                   if chrom_evidence(source,proposal['chromatin'])['posible_omisión'] <= .065)
    preliminary = chrom_evidence(source,initial['chromatin'])
    alternatives = [initial]
    if preliminary['soporte'] < .60 or preliminary['borde_débil'] > .55:
        multiscale = mapas_tincion_multiescala(rgb)
        kind = 'Multi-Otsu y Random Walker'
        compact = full_chromatin(segmentar_random_walker(multiscale,preparar_random_walker(multiscale,kind),kind))
        options.append(dict(compact,method='Multi-Otsu y componentes de alta densidad'))
        alternatives.append(compact)
    focus = select_chromatin(source,options,required_gain=.068)
    detection = informacion_cromatinica(focus)
    vectors = np.dstack((maps['he6'],maps['lab6'][...,1:],.55*maps['std']))
    raw,band,model = corte_en_banda(ctx,detection,vectors,14)
    lattice = grafo_multicanal(ctx,vectors,n=380)
    rivals = modelos_de_apariencia(ctx,lattice,vectors,detection,order=2)
    cut = corte_espacial(lattice,rivals,smoothing=1.35,offset=.04)
    body = [label_body(item['cell'],detection,item,'territorios de tejido' if i == 0 else 'Random Walker')
            for i,item in enumerate(alternatives)]
    body.extend([label_body(raw,detection,initial,'banda de cobertura local',band,model),
                 label_body(cut,detection,initial,'competidores con GMM propios',lattice,rivals)])
    result = select_body(source,body)
    return result, f"Cromatina por {focus['method']}. Cuerpo por {result['method']}"
