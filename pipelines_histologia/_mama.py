"""Pipeline RGB consolidado para mama."""


from skimage.segmentation import morphological_chan_vese
from ._procesamiento import (
    CENTER,
    FRAME,
    GLOBAL_FILTERS,
    HAS_INCLUSIVE_MAX,
    MCP_Geometric,
    RADIAL,
    SEED,
    XX,
    YY,
    appearance6,
    associate_components,
    associate_v2,
    attached,
    audited_result,
    base_maps,
    informacion_cromatinica,
    corte_en_banda,
    breadth_first_order,
    build_markers,
    cell_raw,
    chromatin_candidates,
    chromatin_response,
    class_probabilities,
    cluster_options,
    color,
    foco_compacto,
    components,
    coverage6,
    cv2,
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
    grad,
    historical_holes,
    historical_objects,
    image_features,
    log_densidad_mezcla,
    logsumexp,
    made,
    grafo_multicanal,
    mapas_apariencia,
    maximum_flow,
    mean_shift_centers,
    measure,
    morphology,
    ndi,
    np,
    parcel_graph,
    parcel_means6,
    pd,
    preprocess,
    propose_components,
    refine_cell,
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
    threshold_result,
    u8,
    mapas_tincion_multiescala,
    mapas_region_tincion,
    formar_resultado_regional,
)
from .seleccion import select_body


CONFIG = {'map': 'he_blend', 'threshold': 'hysteresis', 'sigma': 0.55, 'clahe': False, 'offset': 0.1, 'high_min': 0.53, 'high_max': 0.82, 'hysteresis_width': 0.12, 'min_area': 5, 'split_area': 850, 'anchor_radius': 30, 'group_gap': 15, 'group_radius': 32, 'cell_method': 'geodesic', 'color_weight': 0.18, 'edge_weight': 12, 'seed_expand': 1, 'compactness': 0.0005, 'geodesic_bias': 0.97, 'contour_iterations': 8}


SETTINGS = {'stain_mix': (0.53, 0.29), 'edge_mix': (0.4, 0.41, 0.19), 'sigma': 0.55, 'context_sigma': 8, 'local_boost': 0.32, 'high_range': (0.52, 0.86), 'offset': 0.1, 'hysteresis': 0.12, 'fragment_area': 5, 'split_area': 980, 'anchor_radius': 29, 'gap': 15, 'group_radius': 33, 'pair_center': 14, 'barrier': 0.6, 'stain_drop': 0.23, 'bridge_drop': 0.27, 'ring': 9, 'ring_similarity': 0.47, 'seed_edge': 0.45, 'grabcut_iterations': 3}


PROPOSED_CELL = 'grabcut'


def constrained_support(maps, seeds, exterior, border):
    if not seeds.any():
        return np.zeros_like(seeds), np.zeros(seeds.shape, np.float32)
    edge_cost = 1 + 12 * border**1.8 + 4 * (~maps['tissue'])
    forward, _ = MCP_Geometric(edge_cost).find_costs([tuple(row) for row in np.argwhere(seeds)])
    backward, _ = MCP_Geometric(edge_cost).find_costs([tuple(row) for row in np.argwhere(exterior)])
    relative = ((backward - forward) / (forward + backward + 1e-6)).astype(np.float32)
    return relative >= -.05, relative


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


def breast_maps(rgb):
    maps = base_maps(rgb, (9, 26, 5))
    edge_components = []
    for sigma in (.75, 2.1):
        edge_components.append(scale(np.sqrt(grad(maps['a'], sigma)**2 + grad(maps['b'], sigma)**2
                                              + .55 * grad(maps['e'], sigma)**2), 5, 99))
    maps['stop'] = .24 * maps['edge'] + .41 * edge_components[0] + .35 * edge_components[1]
    return maps


def breast_initial(maps, reference):
    chrom = reference['chromatin']
    neighbors = (reference['labels'] > 0) & ~expand(chrom, 1) & ~reference['ambiguous']
    blocked = ~maps['tissue'] | neighbors | FRAME
    if not chrom.any():
        return chrom, chrom, blocked, np.zeros(chrom.shape)
    starts = [tuple(point) for point in np.argwhere(shrink(chrom, 1))]
    if not starts:
        starts = [tuple(point) for point in np.argwhere(chrom)]
    cost = 1 + 9 * maps['stop']**1.7
    cost[~maps['tissue']] *= 3.
    target_cost, _ = MCP_Geometric(cost).find_costs(starts)
    seeds_out = shrink(neighbors, 1) | expand(FRAME, 2) | shrink(~maps['tissue'], 1)
    competitor_cost, _ = MCP_Geometric(cost).find_costs([tuple(p) for p in np.argwhere(seeds_out)])
    balance = (competitor_cost - .77 * target_cost) / (competitor_cost + target_cost + 1e-6)
    allowed = (balance >= -.08) & ~blocked
    # A detected chromatin pixel is constrained as foreground during evolution.
    allowed[chrom] = True
    initial = attached((balance > .075) & allowed, chrom)
    return chrom, initial, ~allowed, balance


def breast_evolution(maps, chrom, initial, excluded, mode='Geodésico con interior estable', perturb=0):
    if not chrom.any() or not initial.any():
        return initial.copy()

    stable = shrink(initial, 4) if mode == 'Geodésico con interior estable' else np.zeros_like(chrom)

    def constrain(level):
        level[excluded] = 0
        level[stable] = 1
        level[chrom] = 1

    if mode == 'Chan–Vese':
        image = .52 * maps['l'] + .33 * (1 - maps['e']) + .15 * (1 - maps['h'])
        raw = segmentation.morphological_chan_vese(image, 20 + 2 * perturb, initial, smoothing=2,
                                                    lambda1=1.05, lambda2=1, iter_callback=constrain)
    else:
        stop = 1 / (1 + 7 * maps['stop']**1.4)
        restrained = mode == 'Geodésico con interior estable'
        raw = segmentation.morphological_geodesic_active_contour(
            stop, (12 if restrained else 20) + 2 * perturb, initial, smoothing=1,
            threshold=.37, balloon=0 if restrained else -1,
            iter_callback=constrain)
    return raw.astype(bool)


def breast_finish(raw, chrom, initial, excluded, balance, maps):
    cell = small_holes(attached(raw, chrom), maps['tissue'] & ~excluded, 10)
    return made(cell, chrom.copy(), excluded, raw=raw, initial=initial, membership=balance)


def run_breast(rgb, method='Geodésico con interior estable', perturb=0):
    maps = breast_maps(rgb)
    chrom, initial, excluded, balance = breast_initial(maps, legacy_second(rgb))
    raw = (initial if method == 'Inicial geodésica' else
           breast_evolution(maps, chrom, initial, excluded, method, perturb))
    return breast_finish(raw, chrom, initial, excluded, balance, maps)


def preparar_reconstruccion_mama(maps, old, mode, step=0):
    chrom = old['chromatin'].copy()
    s = maps['response']
    if mode == 'Multi-Otsu y reconstrucción':
        vector = .57 * s['E'] + .43 * s['Lab-L inverso']
        _, levels, classes = threshold_result(vector, 'Multi-Otsu')
        if classes is None:
            classes = np.zeros(chrom.shape, np.int32)
        posterior = np.eye(3, dtype=np.float32)[classes]
    else:
        algo = 'Fuzzy C-means' if mode == 'Fuzzy y reconstrucción' else 'GMM EM'
        classes, posterior = cluster_options(maps, algo, 4, 'E+Lab')
    score = .61 * s['H'] + .24 * s['Blue Ratio'] + .15 * s['Contraste local']
    parts = split_components(score > np.quantile(score, .84), score, minimum=20)
    opponents = reject_union([part for part in parts if not (part & expand(chrom, 2)).any()], chrom.shape)
    background = expand(FRAME, 2) | shrink(~maps['tissue'], 1) | shrink(opponents)
    edge = .58 * maps['edge4'] + .42 * scale(grad(maps['e'], 1.75))
    allowed, balance = constrained_support(maps, chrom, background, edge)
    shoulder = expand(chrom, 12) & ~expand(chrom, 2) & ~background & maps['tissue']
    nearby = posterior[shoulder].sum(axis=0) + 4
    distant = posterior[background].sum(axis=0) + 4
    contrast = (nearby / nearby.sum()) / (distant / distant.sum())
    class_support = posterior @ (contrast / (1 + contrast))
    return chrom, classes, opponents, allowed, balance, class_support, edge


def reconstruir_mama(maps, prepared, mode, step=0):
    chrom, classes, opponents, allowed, balance, support, edge = prepared
    possible = allowed & maps['tissue'] & ~opponents & ~FRAME
    # Chromatin is foreground during reconstruction. Its shape is fixed across the experiment.
    strong = possible & ((support > .57 + .02 * step) | (balance > .27))
    weak = possible & ((support > .43 + .02 * step) | (balance > .16)) & ((edge <= .58) | (balance >= .19))
    markers = strong & expand(chrom, 4)
    stable = attached((weak | markers | chrom) & (possible | chrom), chrom)
    if mode == 'GMM y contorno geodésico' and stable.any():
        interior = shrink(stable, 3)

        def constrain(level):
            level[~possible] = 0
            level[chrom | interior] = 1

        stop = 1 / (1 + 6 * edge**1.4)
        stable = segmentation.morphological_geodesic_active_contour(
            stop, 14 + 2 * step, stable, smoothing=1, balloon=0, iter_callback=constrain).astype(bool)
    cell = small_holes(stable, possible | chrom, 9)
    return made(cell, chrom, opponents, candidate=strong, seeds=(markers.astype(np.int32) + 2 * chrom),
                territory=classes, domain=weak, membership=support)


def contorno_regional(maps, state, support, anchor, rivals):
    # A scalar likelihood posterior carries the correlated color cues. Pixels far from the
    # posterior/boundary corridor keep their initial assignment during 8 iterations.
    probability = state['probability'].astype(np.float32)
    keep = shrink(support, 2) & (probability > .68)
    reject = (~maps['tissue']) | rivals | (distance(support | anchor) > 7)
    region = morphological_chan_vese(probability, 8, init_level_set=support.astype(np.int8),
                                    smoothing=2, lambda1=1., lambda2=1.2).astype(bool)
    return (region | keep) & ~reject


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


def label_body(raw, detection, initial, method, graph=None, model=None):
    # Core contact and hole filling apply to new region estimates. Established bodies are kept intact here.
    has_graph = graph is not None
    result = formar_resultado_regional(raw,detection,initial,graph,model,cleanup=has_graph)
    result['method'] = method
    return result


def segmentar_recorte(rgb, source):
    initial = run_breast(rgb,'Geodésico con interior estable')
    maps = mapas_tincion_multiescala(rgb)
    mode = 'Multi-Otsu y reconstrucción'
    first = reconstruir_mama(maps,preparar_reconstruccion_mama(maps,initial,mode),mode)
    if first.get('reference_retained',False):
        first = initial
    first = full_chromatin(first)
    maps = mapas_apariencia(rgb)
    ctx = new_context(maps,first)
    detection = ctx['base']
    vectors = np.dstack((maps['lab6'],.86*maps['e'],.4*maps['std']))
    raw,graph,model = corte_en_banda(ctx,detection,vectors,15)
    lattice = grafo_multicanal(ctx,vectors,n=400,compactness=.105)
    likelihood = modelos_de_apariencia(ctx,lattice,vectors,order=3)
    initial_cut = corte_espacial(lattice,likelihood,smoothing=1.35,offset=.03)
    cont = contorno_regional(maps,likelihood,initial_cut,ctx['anchor'],detection[3])
    body = [dict(first,method='Multi-Otsu y reconstrucción'),
            label_body(raw,detection,first,'corte de banda multimodal',graph,model),
            label_body(cont,detection,first,'contorno activo sobre pertenencia',lattice,likelihood)]
    result = select_body(source,body)
    return result, f"Cromatina por reconstrucción de tinción. Cuerpo por {result['method']}"
