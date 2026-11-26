"""Pipeline RGB consolidado para sarcoma."""


from ._procesamiento import (
    CENTER,
    FRAME,
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
    region_contacts,
    region_means,
    regions,
    robust_scale,
    scale,
    segment_image,
    segmentation,
    shrink,
    signature,
    small_holes,
    sparse,
    corte_espacial,
    spsolve,
    u8,
    mapas_region_tincion,
    formar_resultado_regional,
)
from .seleccion import select_body


CONFIG = {'map': 'he_blue', 'threshold': 'hysteresis', 'sigma': 0.45, 'clahe': False, 'offset': 0.1, 'high_min': 0.56, 'high_max': 0.82, 'hysteresis_width': 0.13, 'min_area': 5, 'split_area': 850, 'anchor_radius': 30, 'group_gap': 12, 'group_radius': 30, 'cell_method': 'geodesic', 'color_weight': 0.22, 'edge_weight': 16, 'seed_expand': 1, 'compactness': 0.0005, 'geodesic_bias': 1.0, 'contour_iterations': 12}


SETTINGS = {'stain_mix': (0.62, 0.28), 'edge_mix': (0.46, 0.26, 0.28), 'sigma': 0.5, 'context_sigma': 8, 'local_boost': 0.35, 'high_range': (0.56, 0.86), 'offset': 0.11, 'hysteresis': 0.11, 'fragment_area': 5, 'split_area': 850, 'anchor_radius': 29, 'gap': 13, 'group_radius': 31, 'pair_center': 13, 'barrier': 0.6, 'stain_drop': 0.22, 'bridge_drop': 0.27, 'ring': 8, 'ring_similarity': 0.48, 'seed_edge': 0.45, 'grabcut_iterations': 3}


PROPOSED_CELL = 'grabcut'


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


def harmonic_membership(labels, channels, edge, foreground, background, prior, coupling=.20):
    means = region_means(labels, channels)
    (left, right), boundary, length = region_contacts(labels, edge)
    contrast = np.linalg.norm(means[left] - means[right], axis=1)
    weight = np.sqrt(length) * np.exp(-4 * boundary - 5 * contrast)
    adjacency = sparse.coo_matrix((np.r_[weight, weight], (np.r_[left, right], np.r_[right, left])),
                                  shape=(len(means), len(means))).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    unary = region_means(labels, [prior]).ravel()
    fg = np.unique(labels[foreground])
    bg = np.setdiff1d(np.unique(labels[background]), fg)
    unary[fg] = 1
    unary[bg] = 0
    confidence = np.full(len(means), coupling)
    confidence[np.r_[fg, bg]] = 18
    solution = spsolve(laplacian + sparse.diags(confidence), confidence * unary)
    return solution[labels].astype(np.float32)


def sarcoma_maps(rgb):
    maps = base_maps(rgb, (9, 24, 5))
    image = .65 * (1 - maps['l']) + .35 * maps['e']
    energies = []
    for theta in np.arange(4) * np.pi / 4:
        quadrature = []
        for phase in (0., np.pi / 2):
            kernel = cv2.getGaborKernel((17, 17), 3.3, theta, 7., .55, phase, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            kernel /= np.sqrt(np.sum(kernel**2))
            quadrature.append(cv2.filter2D(image, cv2.CV_32F, kernel))
        energy = cv2.GaussianBlur(np.hypot(*quadrature), (0, 0), 1.8)
        energies.append(scale(energy))
    stack = np.stack(energies, axis=-1)
    maps['gabor'] = stack
    maps['orientation'] = np.argmax(stack, axis=-1)
    maps['texture_edge'] = scale(np.sqrt(sum(grad(energy, 1.)**2 for energy in energies)), 5, 99)
    return maps


def sarcoma_seeds(maps, ref):
    chrom = ref['chromatin']
    rejected = (ref['labels'] > 0) & ~expand(chrom, 1) & ~ref['ambiguous']
    bg = ~maps['tissue'] | expand(FRAME, 2) | shrink(rejected, 1)
    base = np.dstack((maps['l'], 1.25 * maps['a'], 1.25 * maps['b'], .48 * maps['e']))
    neighborhood = expand(chrom, 7) & ~expand(chrom, 2) & ~bg
    prior = .53 + .21 * maps['e'] - .0065 * RADIAL
    if chrom.any() and rejected.any():
        relative = distance(rejected) - distance(chrom)
        prior -= .15 * (-relative / 14).clip(0, 1)
    prior = prior.clip(.03, .68)
    seed = neighborhood & (maps['edge'] < .22)
    return chrom, rejected, bg, base, prior, seed


def sarcoma_partition(maps, chrom, rejected, bg, base, prior, seed, method='Grafo Lab + Gabor', perturb=0):
    sp = segmentation.slic(base, n_segments=380, compactness=.11, start_label=0,
                            convert2lab=False, max_num_iter=7)
    if not chrom.any():
        return chrom.copy(), sp, np.zeros(chrom.shape)
    if method == 'Contorno geodésico':
        transition = .62 * maps['edge'] + .38 * maps['texture_edge']
        stop = 1 / (1 + 5.5 * transition)
        allowed = ~bg & ~rejected
        initial = attached((seed | chrom | (expand(chrom, 5) & (maps['edge'] <= .37))) & allowed, chrom)

        def constrain(level):
            level[~allowed] = 0
            level[chrom] = 1

        raw = segmentation.morphological_geodesic_active_contour(
            stop, 18 + 2 * perturb, initial, smoothing=1, threshold=.39, balloon=1,
            iter_callback=constrain).astype(bool)
        return raw, sp, stop
    channels = list(np.moveaxis(base, -1, 0))
    transition = maps['edge']
    if method == 'Grafo Lab + Gabor':
        channels.extend(np.moveaxis(.33 * maps['gabor'], -1, 0))
        transition = .68 * transition + .32 * maps['texture_edge']
    probability = harmonic_membership(sp, channels, transition, chrom | seed, bg, prior, .32)
    return probability >= .56 + .025 * perturb, sp, probability


def sarcoma_finish(raw, sp, probability, maps, chrom, neighbors):
    allowed = maps['tissue'] & ~neighbors & ~FRAME
    cell = small_holes(attached(raw & allowed, chrom), allowed, 12)
    return made(cell, chrom.copy(), neighbors, regions=sp, membership=probability, raw=raw)


def run_sarcoma(rgb, method='Grafo Lab + Gabor', perturb=0):
    maps = sarcoma_maps(rgb)
    chrom, rejected, bg, base, prior, seed = sarcoma_seeds(maps, legacy_second(rgb))
    raw, sp, probability = sarcoma_partition(maps, chrom, rejected, bg, base, prior, seed, method, perturb)
    return sarcoma_finish(raw, sp, probability, maps, chrom, rejected)


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
    fixed = run_sarcoma(rgb, 'Grafo Lab + Gabor')
    fixed = full_chromatin(fixed)
    maps = mapas_apariencia(rgb, with_fibers=True)
    ctx = new_context(maps, fixed)
    detection = ctx['base']
    pigment = np.dstack((maps['lab6'], .6*maps['e']))
    fiber = maps['fibers']
    vectors = np.dstack((pigment, .30*fiber[..., :8], .48*fiber[..., 8:10], .5*fiber[...,10]))
    raw, band, density = corte_en_banda(ctx,detection,vectors,11,maps['fiber_edge'])
    lattice = grafo_multicanal(ctx,vectors,maps['fiber_edge'],365,.118)
    model = modelos_de_apariencia(ctx,lattice,pigment,order=2)
    global_cut = corte_espacial(lattice,model,smoothing=1.15,offset=.2)
    body = [
        dict(fixed, method='grafo LAB y Gabor'),
        label_body(raw,detection,fixed,'banda de transición al estroma',band,density),
        label_body(global_cut,detection,fixed,'corte de apariencia cromática',lattice,model),
    ]
    result = select_body(source,body)
    return result, f"Cromatina fija de LAB y Gabor. Cuerpo por {result['method']}"
