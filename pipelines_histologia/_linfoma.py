"""Pipeline RGB consolidado para linfoma."""


from ._procesamiento import (
    CENTER,
    FRAME,
    HAS_INCLUSIVE_MAX,
    MCP_Geometric,
    RADIAL,
    SEED,
    XX,
    YY,
    all_dark6,
    anchor_region,
    appearance6,
    associate5,
    associate_components,
    associate_v2,
    attached,
    attached_clean,
    audited_result,
    base_maps,
    informacion_cromatinica,
    body_from_regions,
    corte_en_banda,
    boundary_rivals,
    breadth_first_order,
    build_markers,
    candidate_groups6,
    cell_raw,
    choose_anchor,
    chromatin_candidates,
    chromatin_response,
    color,
    foco_compacto,
    compatible_groups,
    components,
    coverage6,
    cv2,
    candidatos_oscuros,
    dark_signals6,
    dilate,
    disk,
    distance,
    distance_to,
    ell,
    erode,
    modelos_de_apariencia,
    estimate_cell,
    expand,
    expit,
    feature,
    energia_orientada,
    filters,
    ajustar_mezcla,
    fixed_mask6,
    grad,
    jerarquia_regional,
    historical_holes,
    historical_objects,
    image_features,
    interval,
    landscape_markers,
    log_densidad_mezcla,
    logsumexp,
    made,
    grafo_multicanal,
    mapas_apariencia,
    marker_image,
    maximum_flow,
    measure,
    fusion_adyacente,
    merge_hierarchy,
    morphology,
    ndi,
    np,
    pair_evidence,
    parcel_graph,
    parcel_means6,
    pd,
    preprocess,
    foco_central,
    propose_components,
    random_walker_regions,
    refine_cell,
    region_contacts,
    region_means,
    regions,
    reject_union,
    robust_scale,
    scale,
    segment_image,
    segmentation,
    shrink,
    signature,
    small_holes,
    sparse,
    corte_espacial,
    split_components,
    spsolve,
    u8,
    upper_contacts,
    v4_contacts,
    mapas_tincion_multiescala,
    v4_means,
    v4_parcels,
    mapas_region_tincion,
    formar_resultado_regional,
)
from .seleccion import inspect_record, select_chromatin


CONFIG = {'map': 'he_blend', 'threshold': 'multiotsu', 'sigma': 0.55, 'clahe': False, 'offset': 0.1, 'high_min': 0.6, 'high_max': 0.84, 'hysteresis_width': 0.1, 'min_area': 5, 'split_area': 850, 'anchor_radius': 30, 'group_gap': 7, 'group_radius': 25, 'cell_method': 'geodesic', 'color_weight': 0.12, 'edge_weight': 14, 'seed_expand': 1, 'compactness': 0.0005, 'geodesic_bias': 0.9, 'contour_iterations': 12}


SETTINGS = {'stain_mix': (0.55, 0.3), 'edge_mix': (0.47, 0.32, 0.21), 'sigma': 0.65, 'context_sigma': 7, 'local_boost': 0.39, 'high_range': (0.6, 0.86), 'offset': 0.12, 'hysteresis': 0.09, 'fragment_area': 5, 'split_area': 820, 'anchor_radius': 29, 'gap': 9, 'group_radius': 25, 'pair_center': 10, 'barrier': 0.53, 'stain_drop': 0.19, 'bridge_drop': 0.27, 'ring': 6, 'ring_similarity': 0.55, 'seed_edge': 0.45, 'grabcut_iterations': 3}


PROPOSED_CELL = 'grabcut_previo'


def compact_nuclei(score, tissue, shift=0, minimum=9):
    roi = (RADIAL < 34) & tissue
    values = score[roi] if roi.sum() >= 15 else score[tissue]
    if values.size < 15:
        return []
    level, _ = cv2.threshold(np.rint(values * 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high = np.clip(level / 255 + .08 + shift, .54, .82)
    strong = (score >= high) & tissue
    weak = (score >= high - .095) & tissue
    recovered = attached(weak, strong)
    return components(recovered, minimum)


def legacy_first(rgb):
    return segment_image(rgb, CONFIG)


def legacy_second(rgb, first=None):
    f = preprocess(rgb, SETTINGS)
    labels, high = propose_components(f, SETTINGS)
    if PROPOSED_CELL == 'grabcut_previo':
        first = legacy_first(rgb) if first is None else first
        labels, chrom = first['labels'], first['chromatin']
        ambiguous = np.zeros_like(chrom)
    else:
        chrom, _, ambiguous = associate_v2(labels, f, high, SETTINGS)
    raw, markers = cell_raw(f, chrom, labels, ambiguous, SETTINGS, 'grabcut')
    return dict(cell=refine_cell(raw, chrom, f, markers), chromatin=chrom,
                labels=labels, ambiguous=ambiguous, excluded=markers['neighbors'])


def lymph_maps(rgb):
    maps = base_maps(rgb, (7, 22, 4))
    local = np.maximum(0, maps['h'] - cv2.GaussianBlur(maps['h'], (0, 0), 5))
    maps['score'] = .65 * maps['h'] + .22 * (1 - maps['l']) + .13 * scale(local)
    broad = scale(np.sqrt(grad(maps['a'], 1.6)**2 + grad(maps['b'], 1.6)**2) +
                  .5 * grad(maps['e'], 1.6), 6, 99)
    maps['relief'] = .42 * maps['edge'] + .58 * broad
    return maps


def lymph_parcels(maps, method='Cuencas con asociación distribuida', perturb=0):
    relief, tissue = maps['relief'], maps['tissue']
    if method == 'Regiones Felzenszwalb':
        vector = np.dstack((maps['l'], 1.45 * maps['a'], 1.45 * maps['b'], .65 * maps['e']))
        return segmentation.felzenszwalb(vector, scale=37 + 4 * perturb, sigma=.65,
                                         min_size=23, channel_axis=-1) + 1
    ridge = (relief > np.quantile(relief[tissue], .70)) if tissue.any() else ~tissue
    clearance = distance(ridge | ~tissue | FRAME)
    local_max = clearance >= cv2.dilate(clearance, ell(4))
    points = []
    for _, peak, _, _ in components(local_max & (clearance >= 2.45), 1):
        pixels = np.argwhere(peak)
        y, x = pixels[np.argmax(clearance[peak])]
        points.append((float(clearance[y, x]), int(y), int(x)))
    markers = np.zeros(tissue.shape, np.int32)
    for _, y, x in sorted(points, reverse=True):
        if not markers[max(0, y - 5):y + 6, max(0, x - 5):x + 6].any():
            markers[y, x] = int(markers.max()) + 1
    if not markers.any() and tissue.any():
        markers[np.unravel_index(np.argmax(clearance * tissue), tissue.shape)] = 1
    return segmentation.watershed(relief, markers, mask=tissue)


def lymph_associate(maps, parcels, perturb=0, distributed=True):
    score = maps['score']
    regions_nuc = compact_nuclei(score, maps['tissue'], .013 * perturb, minimum=7)
    empty = np.zeros(score.shape, bool)
    if not regions_nuc:
        return empty, empty, empty, empty, region_means(parcels, [score]).ravel()[parcels]
    # Rank independently segmented regions using central stain support.
    central = np.exp(-.5 * (RADIAL / 18)**2)
    high = np.logical_or.reduce([item[1] for item in regions_nuc])
    rank = np.bincount(parcels.ravel(), weights=(central * score * high).ravel())
    rank /= np.maximum(np.bincount(parcels.ravel())**.28, 1)
    rank[0] = 0
    if rank.max() == 0:
        return empty, empty, empty, high, score
    primary = int(np.argmax(rank))
    base = parcels == primary
    anchor = max(regions_nuc, key=lambda item: (item[1] & base).sum())[1]
    core_mean = np.quantile(score[anchor], .78)
    candidates, rejected = [anchor], []
    for _, part, area, centroid in regions_nuc:
        if (part & anchor).any():
            continue
        gap, path = interval(anchor, part)
        midpoint = (centroid + np.argwhere(anchor).mean(axis=0)) / 2
        barrier = np.quantile(maps['relief'][path], .8) if path[0].size else 1
        paper = (~maps['tissue'][path]).mean() if path[0].size else 1
        stain = np.quantile(score[part], .80)
        overlap = (part & base).sum() / area
        shared = overlap > .34
        gap_limit = 16 if distributed else 14
        barrier_limit = .59 if distributed else .52
        supported = (gap <= gap_limit and np.linalg.norm(midpoint - CENTER) < 14.5
                     and barrier < barrier_limit + .02 * perturb)
        if not (part & FRAME).any() and stain >= core_mean - .18 and (shared or (supported and paper < .18)):
            candidates.append(part)
        else:
            rejected.append(part)
    exclusion = np.logical_or.reduce(rejected) if rejected else empty
    support = np.logical_or.reduce(candidates)
    (left, right), barrier, lengths = region_contacts(parcels, maps['relief'])
    means = region_means(parcels, [maps['l'], maps['a'], maps['b'], maps['e']])
    distance_color = np.linalg.norm(means[left] - means[right], axis=1)
    outside_support = exclusion & ~expand(support, 2)
    if distributed:
        count = np.bincount(parcels.ravel())
        rival_load = np.bincount(parcels[outside_support], minlength=len(rank))
        target_load = np.bincount(parcels[support], minlength=len(rank))
        competing = set(np.flatnonzero((rival_load >= np.maximum(5, .08 * count))
                                       & (rival_load >= 2 * target_load)))
    else:
        competing = set(np.unique(parcels[outside_support]))
    competing.discard(primary)
    edge_cost = .70 * barrier + .30 * distance_color
    adjacency = {idx: [] for idx in range(len(rank))}
    for i, j, cost, length in zip(left, right, edge_cost, lengths):
        if i and j and length >= 3:
            adjacency[i].append((j, cost))
            adjacency[j].append((i, cost))
    from heapq import heapify, heappush, heappop
    best = np.full(len(rank), np.inf)
    seed_ids = [primary]
    anchor_overlap = region_means(parcels, [anchor]).ravel()
    if distributed:
        own = np.bincount(parcels[anchor], minlength=len(rank))
        seed_ids.extend(int(i) for i in np.flatnonzero(
            (own >= max(8, .07 * anchor.sum())) & (anchor_overlap > .09)) if i and i != primary)
    best[seed_ids] = 0
    queue = [(0., idx) for idx in seed_ids]
    heapify(queue)
    while queue:
        cost, current = heappop(queue)
        if cost > best[current]:
            continue
        for neighbor, jump in adjacency[current]:
            if neighbor in competing or jump > .41 + .02 * perturb:
                continue
            # Expansion slows when the interface and the intensity shift both stand out.
            light_shift = abs(means[current, 0] - means[neighbor, 0])
            brightness_cost = .42 * max(0, light_shift - .14) if distributed else 0
            combined = cost + jump + .06 + brightness_cost
            if combined < best[neighbor]:
                best[neighbor] = combined
                heappush(queue, (combined, neighbor))
    max_path = np.full(len(rank), (.57 if distributed else .48) + .02 * perturb)
    supported_ids = np.unique(parcels[support])
    max_path[supported_ids] += .25
    reached = (best <= max_path) | (anchor_overlap > .16)
    reached[0] = False
    domain = reached[parcels] & maps['tissue'] & ~exclusion
    accepted = np.zeros_like(empty)
    omitted = np.zeros_like(empty)
    for part in candidates:
        fraction = (domain & part).sum() / part.sum()
        if fraction >= (.68 if distributed else .77):
            accepted |= part & domain
            if distributed and fraction < 1:
                omitted |= part & ~domain
        else:
            omitted |= part
    return domain, accepted, exclusion, omitted, np.exp(-best[parcels]).astype(np.float32)


def lymph_finish(domain, chrom, neighbors, omitted, membership, maps, parcels):
    anchored = attached(domain, chrom)
    cell = small_holes(anchored, maps['tissue'] & ~neighbors, 8)
    return made(cell, chrom, neighbors, omitted, regions=parcels, membership=membership, raw=domain)


def lymph_review_collapsed(result, maps):
    cell, chrom = result['cell'], result['chromatin']
    flank = cell & ~expand(chrom, 1)
    if cell.any() and flank.sum() >= .12 * chrom.sum():
        return result
    ref = legacy_second(maps['rgb'])
    ref_cell, ref_chrom = ref['cell'], ref['chromatin']
    if not ref_chrom.any() or (ref_chrom & ~ref_cell).any():
        return result
    signal = chrom if chrom.any() else result['omitted']
    if not signal.any() or (signal & ref_cell).sum() < .67 * signal.sum():
        return result
    ref_rim = ref_cell & ~shrink(ref_cell) & ~expand(ref_chrom, 2)
    edge = cv2.dilate(maps['edge'], ell(1))
    if not ref_rim.any() or (edge[ref_rim] >= .42).mean() < .40:
        return result
    return made(ref_cell, ref_chrom, ref['excluded'], result['omitted'] & ~ref_chrom,
                regions=result['regions'], membership=result['membership'], raw=result['raw'],
                reference_retained=True)


def run_lymph(rgb, method='Cuencas con control de colapso', perturb=0):
    maps = lymph_maps(rgb)
    parcels = lymph_parcels(maps, method, perturb)
    domain, chrom, neighbors, omitted, membership = lymph_associate(
        maps, parcels, perturb, method in ('Cuencas con asociación distribuida', 'Cuencas con control de colapso'))
    result = lymph_finish(domain, chrom, neighbors, omitted, membership, maps, parcels)
    return lymph_review_collapsed(result, maps) if method == 'Cuencas con control de colapso' else result


def lymph_cell_first(maps, mode, step=0):
    edge = .65 * maps['edge4'] + .35 * scale(grad(maps['e'], 1.8))
    tissue = maps['tissue']
    ridge = (edge > np.quantile(edge[tissue], .72)) if tissue.any() else ~tissue
    free = distance(ridge | ~tissue | FRAME)
    maxima = (free >= cv2.dilate(free, ell(5))) & (free > 2.6)
    parts = [mask for _, mask, _, _ in components(maxima)]
    markers = marker_image(parts, expand(FRAME, 2) | shrink(~tissue))
    vectors = maps['color_vector']
    if mode == 'Cuencas Canny':
        owner = segmentation.watershed(edge + .2 * maps['canny'], markers, mask=~FRAME)
        p = None
    else:
        parcels = v4_parcels(vectors, n=580)
        owner, p = random_walker_regions(parcels, vectors, edge, markers, 5.7 + .4 * step)
    return owner, markers, edge, p


def segmentar_cuencas_multicanal(maps, prepared, mode, step=0):
    owner, markers, barrier, probabilities = prepared
    s = maps['response']
    response = 0.5 * s['H normalizada'] + 0.25 * s['Blue Ratio'] + 0.25 * s['Contraste local']
    high = np.quantile(response[RADIAL < 34], 0.82) + 0.012 * step
    candidate = attached((response >= high - 0.105) & maps['tissue'], response >= high)
    parts = split_components(candidate, response, minimum=8, split=False)
    primary = choose_anchor(parts, response)
    blank = np.zeros(owner.shape, bool)
    if primary is None:
        return made(blank, blank, candidate=candidate, seeds=markers, territory=owner, domain=blank)
    anchor = parts[primary]
    counts = np.bincount(owner[anchor].ravel(), minlength=int(owner.max()) + 1)
    counts[:2] = 0
    if not counts.any():
        return made(blank, blank, omitted=anchor, candidate=candidate, seeds=markers, territory=owner, domain=blank)
    basins = np.flatnonzero(counts >= max(4, counts[2:].max() * 0.21))
    if mode == 'Cuencas Canny':
        basins = np.array([int(np.argmax(counts))])
    provisional = np.isin(owner, basins) & maps['tissue']
    (ii, jj), strength, length = v4_contacts(owner, barrier)
    touches = np.isin(ii, basins) & ~np.isin(jj, basins) | ~np.isin(ii, basins) & np.isin(jj, basins)
    neighbors = np.where(np.isin(ii, basins), jj, ii)
    accessible = neighbors[touches & (neighbors >= 2) & (strength <= 0.46) & (length >= 5)]
    expanded_region = provisional | np.isin(owner, accessible)
    accept, unsure, rivals = compatible_groups(parts, primary, response, barrier, maps['tissue'], expanded_region, 15)
    for idx in accept[1:]:
        part = parts[idx]
        hits = np.bincount(owner[part & expanded_region], minlength=len(counts))
        promising = np.flatnonzero(hits >= 0.25 * part.sum())
        basins = np.union1d(basins, promising[promising >= 2])
    domain = np.isin(owner, basins)
    excluded = reject_union([parts[i] for i in rivals], owner.shape)
    chrom = reject_union([parts[i] for i in accept], owner.shape)
    cell = attached_clean(domain, chrom, maps['tissue'] & ~excluded & ~FRAME, 6)
    pending = reject_union([parts[i] for i in unsure], owner.shape) | chrom & ~cell
    return made(cell, chrom & cell, excluded, pending, candidate=candidate, seeds=markers, territory=owner, domain=domain, preconstraint=chrom.copy())


def preparar_cuencas_jerarquicas(maps, mode, step=0):
    score = .47 * maps['h'] + .28 * maps['ratio'] + .13 * maps['od'] + .12 * maps['local']
    if mode == 'H-minima y asociación estricta':
        field = jerarquia_regional(maps, 'H-minima', limit=.39, step=step)
    else:
        field = jerarquia_regional(maps, 'H-maxima de espacio libre', limit=.43, step=step)
    field['initial_id'] = anchor_region(field['labels'], score, maps['tissue'])
    return score, field


def segmentar_cuencas_jerarquicas(maps, prepared, mode, step=0):
    score, field = prepared
    groups, candidate, strong = candidatos_oscuros(score, maps, .826, .122, 1000, step)
    labels = field['labels']
    primary = foco_central(groups, score, labels, field['initial_id'], 32, maps['l'])
    blank = np.zeros(score.shape, bool)
    if primary is None:
        return audited_result(blank, blank, uncertain=candidate, initial=blank, field=field,
                              candidate=candidate)
    strict = mode == 'H-minima y asociación estricta'
    linked, uncertain = associate5(maps, score, groups, primary, field, 16 if not strict else 13,
                                    .58 if not strict else .50, .12, strict)
    chrom = reject_union([groups[i] for i in linked], labels.shape)
    pending = reject_union([groups[i] for i in uncertain], labels.shape)
    competing, others = boundary_rivals(labels, groups, linked + uncertain)
    chosen = []
    for i in linked:
        counts = np.bincount(labels[groups[i]])
        counts[0] = 0
        if counts.max() == 0:
            continue
        main = counts.argmax()
        other = np.flatnonzero(counts >= .34 * groups[i].sum())
        chosen.extend([main, *other] if not strict else [main])
    initial = np.isin(labels, chosen)
    domain = body_from_regions(field, maps, initial, others, .39 + .015 * step, 2)
    final = attached(domain & (~competing | initial), initial)
    cell = small_holes(final, maps['tissue'] & ~competing, 8)
    return audited_result(cell, chrom, competing, pending, field=field, candidate=candidate,
                          initial=initial, strong=strong)


def asociacion_por_contacto(maps, previous, regions6):
    vote, signals, dark = all_dark6(maps, previous)
    old = previous['chromatin']
    anchor = foco_compacto(previous)
    selected, unsure, rivals = old.copy(), previous['omitted'].copy(), previous['excluded'].copy()
    score = .67 * signals['H&E'] + .33 * signals['Blue Ratio']
    if not anchor.any():
        return selected, dark | old, unsure, rivals, ()
    ids, counts = np.unique(regions6[anchor], return_counts=True)
    owned = ids[(ids >= 0) & (counts >= .20 * anchor.sum())]
    evidence = []
    for candidate in candidate_groups6(dark, score):
        candidate = candidate & ~old
        if candidate.sum() < 14 or RADIAL[candidate].min() > 37:
            continue
        gap, middle = interval(anchor, candidate)
        barrier = float(maps['edge6'][middle].mean()) if len(middle[0]) else 0.
        overlap = float(np.isin(regions6[candidate], owned).mean())
        color_shift = float(np.linalg.norm(np.median(maps['he6'][candidate], axis=0) -
                                           np.median(maps['he6'][anchor], axis=0)))
        if gap <= 10 and overlap >= .68 and barrier <= .28 and color_shift <= .26:
            selected |= candidate
        elif gap <= 16 and color_shift <= .38:
            unsure |= candidate
        else:
            rivals |= candidate
        evidence.append((int(candidate.sum()), round(gap, 1), round(barrier, 3),
                         round(overlap, 2), round(color_shift, 3)))
    return selected, dark | old, unsure & ~selected, rivals & ~selected, evidence


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
    basin_maps = mapas_tincion_multiescala(rgb)
    mode = 'Cuencas con control de cobertura'
    basins = full_chromatin(segmentar_cuencas_multicanal(basin_maps, lymph_cell_first(basin_maps, mode), mode))
    terrain = mapas_region_tincion(rgb)
    mode = 'Cuencas y cobertura revisada'
    multi = segmentar_cuencas_jerarquicas(terrain, preparar_cuencas_jerarquicas(terrain, mode), mode)
    local = run_lymph(rgb, 'Cuencas con control de colapso')
    chrom_routes = [dict(out, method=method) for out,method in
                    ((basins,'cuencas multicanal'),(multi,'jerarquía de cuencas'),(local,'cuencas locales'))]
    seed = select_chromatin(source, chrom_routes, required_gain=.034)
    ctx = new_context(mapas_apariencia(rgb), seed)
    maps = ctx['maps']
    vectors = np.dstack((maps['he6'],.75*maps['lab6'][...,1:],.40*maps['std']))
    graph = grafo_multicanal(ctx,vectors,n=375)
    fused = fusion_adyacente(graph,vectors,.85)
    contact = asociacion_por_contacto(maps,seed,fused[graph['labels']])
    associated = with_detection(seed, contact, seed['method'] + ' y asociación RAG')
    focus = select_chromatin(source,[seed,associated],required_gain=.013)
    detection = informacion_cromatinica(focus)
    raw, band_graph, density = corte_en_banda(ctx,detection,vectors,11)
    # Competing bodies may belong to distinct nearby foci. Keep cohesive pairs among
    # the choices and compare the additional association as a separate hypothesis.
    choices = [(record, record['method'], record['method']) for record in chrom_routes]
    associated = label_body(seed['cell'],detection,seed,seed['method'])
    choices.append((associated,focus['method'],seed['method']))
    transition = label_body(raw,detection,seed,'corte de transición',band_graph,density)
    choices.append((transition,focus['method'],'corte de transición'))

    def support(choice):
        chrom, body, issues = inspect_record(source,choice[0])
        return not issues, .74*body['score'] + .26*chrom['score']

    result, stain_method, body_method = max(choices,key=support)
    return result, f'Cromatina por {stain_method}. Cuerpo por {body_method}'
