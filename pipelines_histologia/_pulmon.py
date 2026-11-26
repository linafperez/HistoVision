"""Pipeline RGB consolidado para pulmon."""


from ._procesamiento import (
    CENTER,
    FRAME,
    MCP_Geometric,
    RADIAL,
    SEED,
    XX,
    YY,
    all_dark6,
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
    candidate_groups6,
    choose_anchor,
    class_probabilities,
    cluster_options,
    color,
    foco_compacto,
    compatible_groups,
    territorios_por_coste,
    components,
    contour5,
    coverage6,
    cv2,
    candidatos_oscuros,
    dark_signals6,
    distance,
    ell,
    modelos_de_apariencia,
    expand,
    expit,
    energia_orientada,
    field_means,
    filters,
    ajustar_mezcla,
    fixed_mask6,
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
    fusion_adyacente,
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


def stain_basis(rgb):
    """Estimate an H/E basis and robust concentrations in optical density."""
    od = -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1.) / 256.)
    foreground = (np.linalg.norm(od, axis=1) > .32) & (od.min(axis=1) > .052)
    samples = od[foreground]
    if len(samples) < 120:
        return None
    _, sv, vt = np.linalg.svd(samples, full_matrices=False)
    if sv[1] < .045 * sv[0]:
        return None
    plane = vt[:2].T.copy()
    if np.median(samples @ plane[:, 0]) < 0:
        plane[:, 0] *= -1
    xy = samples @ plane
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    ends = np.quantile(angles, [.015, .985])
    basis = plane @ np.vstack([np.cos(ends), np.sin(ends)])
    basis = basis[:, np.argsort(-basis[0])]
    if np.linalg.cond(basis) > 11 or (basis < -.07).any():
        return None
    basis = basis.clip(0)
    basis /= np.linalg.norm(basis, axis=0)
    concentration = np.linalg.lstsq(basis, od.T, rcond=None)[0].clip(0)
    top = np.quantile(concentration[:, foreground], .99, axis=1)
    if min(top) < .055:
        return None
    return basis, concentration, top, foreground


def lab_statistics(rgb):
    rgbf = rgb.astype(np.float32) / 255.
    lab = cv2.cvtColor(rgbf, cv2.COLOR_RGB2Lab)
    foreground = np.linalg.norm(-np.log((rgb.astype(np.float32) + 1) / 256), axis=-1) > .32
    pixels = lab[foreground] if foreground.sum() > 32 else lab.reshape(-1, 3)
    return lab, foreground, pixels.mean(axis=0), pixels.std(axis=0).clip([2., 1., 1.])


def normalize_stain(rgb, method, target_lab, target_stain):
    if method == 'Original':
        return rgb.copy(), 'original'
    if method == 'Reinhard Lab':
        lab, tissue, center, deviation = lab_statistics(rgb)
        _, _, target_center, target_deviation = target_lab
        transformed = lab.copy()
        affine = (lab[tissue] - center) * (target_deviation / deviation).clip(.60, 1.70) + target_center
        transformed[tissue] = .82 * affine + .18 * lab[tissue]
        converted = cv2.cvtColor(transformed, cv2.COLOR_Lab2RGB)
        return np.rint(converted.clip(0, 1) * 255).astype(np.uint8), 'adaptación Lab'
    source = stain_basis(rgb)
    if source is None or target_stain is None:
        return rgb.copy(), 'base H&E no estimable; se conserva el original'
    _, concentrations, scale_in, foreground = source
    target, _, scale_out, _ = target_stain
    normalized = target @ (concentrations * (scale_out / scale_in).clip(.55, 1.85)[:, None])
    rgb_new = (256 * np.exp(-normalized.T) - 1).clip(0, 255)
    source_rgb = rgb.reshape(-1, 3)
    rgb_new[~foreground] = source_rgb[~foreground]
    return np.rint(rgb_new.reshape(rgb.shape)).astype(np.uint8), 'base H&E estimada'


def fixed_flood(rgb_or_channel, seeds, allowed, tolerance):
    frame = np.pad((~allowed).astype(np.uint8), 1, constant_values=1)
    src = np.ascontiguousarray(rgb_or_channel)
    fill = (float(tolerance),) * (src.shape[-1] if src.ndim == 3 else 1)
    for y, x in seeds:
        if frame[y + 1, x + 1]:
            continue
        cv2.floodFill(src, frame, (int(x), int(y)), 0, fill, fill,
                      8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
    return frame[1:-1, 1:-1] == 2


def seed_points(seeds, step=6):
    points = np.argwhere(seeds)
    if not len(points):
        return []
    by_stride = (points[:, 0] % step == 0) & (points[:, 1] % step == 0)
    return points[by_stride].tolist() or [points[len(points) // 2].tolist()]


def preparar_competencia_multicanal(maps, mode, step=0):
    s = maps['response']
    response = .50 * s['Blue Ratio'] + .26 * s['DO'] + .14 * s['H normalizada'] + .10 * s['Contraste local']
    response = cv2.GaussianBlur(response, (0, 0), .68)
    central = (RADIAL <= 31) & maps['tissue']
    samples = response[central] if central.sum() >= 30 else response.ravel()
    # The central zone estimates the threshold. The search and extent use the full field.
    local_hist = u8(samples)
    if mode == 'Triángulo y FloodFill':
        t, _ = cv2.threshold(local_hist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        high = float(np.clip(t / 255, np.quantile(samples, .76), np.quantile(samples, .93)))
        low = high - .082
        support = fixed_flood(response, seed_points((response >= high) & maps['tissue'], 5),
                              (response >= low) & maps['tissue'], .12)
    else:
        upper = (filters.threshold_multiotsu(local_hist, classes=3)[-1] / 255
                 if np.unique(local_hist).size >= 3 else np.quantile(samples, .86))
        high = max(float(upper), float(np.quantile(samples, .78))) + .013 * step
        low = high - .105
        domain = (response >= low) & maps['tissue']
        core = (response >= high) & domain
        if mode in ('GMM y reconstrucción', 'Fuzzy y reconstrucción'):
            algo = 'GMM EM' if mode.startswith('GMM') else 'Fuzzy C-means'
            labs, posterior = cluster_options(maps, algo, 4, 'BR+H+OD')
            support_color = posterior[..., :2].sum(axis=-1)
            domain &= support_color >= .24
            core &= support_color >= .47
        support = attached(domain, core)
    parts = split_components(support, response, minimum=12)
    anchor = choose_anchor(parts, response, 33)
    return response, support, parts, anchor


def territorios_multicanal(maps, prepared, mode, step=0):
    response, support, parts, primary = prepared
    background = expand(FRAME, 2) | shrink(~maps['tissue'], 1)
    markers = marker_image(parts, background)
    broad_context = mode == 'Multi-Otsu y expansión de marcadores'
    limited_context = mode == 'Multi-Otsu y competencia regional'
    if (broad_context or limited_context) and primary is not None:
        nucleus = parts[primary]
        dark_floor = np.quantile(response[nucleus], .78) - .14
        min_support = max(19, np.median([part.sum() for part in parts]) * .80)
        for index, part in enumerate(parts):
            small = part.sum() < min_support
            faint = np.quantile(response[part], .8) < dark_floor
            weak_competitor = (small and faint) if limited_context else (small or faint)
            if index != primary and weak_competitor:
                markers[markers == index + 2] = 0
        rival_seed = markers > 1
        neighborhood = expand(nucleus, 7 if limited_context else 10) & ~expand(nucleus, 2) & ~expand(rival_seed & ~nucleus, 2)
        if neighborhood.any():
            level = np.quantile(maps['e'][neighborhood], .50)
            transition = np.quantile(maps['edge4'][neighborhood], .35)
            companions = neighborhood & maps['tissue'] & (maps['e'] >= level) & (maps['edge4'] <= transition)
            markers[companions & (markers == 0)] = primary + 2
    parcels = v4_parcels(maps['color_vector'], n=530)
    owner, prob = random_walker_regions(parcels, maps['color_vector'], maps['edge4'], markers, 5.8 + .4 * step)
    if primary is None:
        return owner, markers, np.zeros_like(background), np.zeros_like(response), parcels
    selected_id = primary + 2
    # The probabilistic envelope includes areas open to a compatible second group.
    local = prob[..., selected_id]
    competitors = prob.copy()
    competitors[..., selected_id] = 0
    if limited_context:
        other = competitors.max(axis=-1)
        raw = ((owner == selected_id) & (local >= .28)) | ((local >= .31) & (local + .065 >= other))
    elif broad_context:
        raw = (owner == selected_id) | ((local >= .20) & (local + .18 >= competitors.max(axis=-1)))
    else:
        raw = (owner == selected_id) | ((local >= .25) & (local + .13 >= competitors.max(axis=-1)))
    return owner, markers, raw, local, parcels


def segmentar_competencia_multicanal(maps, prepared, territory):
    response, candidate, parts, primary = prepared
    owner, markers, raw, probability, parcels = territory
    empty = np.zeros(raw.shape, bool)
    if primary is None:
        return made(empty, empty, omitted=candidate, candidate=candidate, seeds=markers, territory=owner, domain=raw, membership=probability)
    anchor = parts[primary]
    acceptable, unsure, rivals = compatible_groups(parts, primary, response, maps['edge4'], maps['tissue'], raw, 14)
    rim_context = raw & maps['tissue'] & ~FRAME
    outside = reject_union([parts[i] for i in rivals], raw.shape)
    cell = attached_clean(rim_context, anchor, maps['tissue'] & ~outside, 10)
    chrom = reject_union([parts[i] for i in acceptable], raw.shape)
    uncertain = reject_union([parts[i] for i in unsure], raw.shape) | chrom & ~cell
    return made(cell, chrom & cell, outside, uncertain, candidate=candidate, seeds=markers, territory=owner, domain=raw, membership=probability, preconstraint=chrom.copy())


def kuwahara_lab(lab, radius=2):
    """Choose the local quadrant with the smallest multichannel variance."""
    vector = lab / np.array([100, 51, 51], np.float32)
    offsets = [(0, 0), (radius, 0), (0, radius), (radius, radius)]
    ksize = (radius + 1, radius + 1)
    means, variances = [], []
    for offset in offsets:
        average = cv2.boxFilter(vector, -1, ksize, anchor=offset)
        square = cv2.boxFilter(vector**2, -1, ksize, anchor=offset)
        means.append(average)
        variances.append(np.maximum(0, square - average**2).sum(axis=-1))
    best = np.argmin(variances, axis=0)
    return np.take_along_axis(np.stack(means), best[None, ..., None], axis=0)[0] * np.array([100, 51, 51], np.float32)


def anisotropic_lab(lab, iterations=9):
    vector = lab / np.array([100, 65, 65], np.float32)
    kappa = .062
    for _ in range(iterations):
        padded = np.pad(vector, ((1, 1), (1, 1), (0, 0)), mode='edge')
        neighbors = (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
        change = sum(delta * np.exp(-(delta**2).sum(axis=-1, keepdims=True) / kappa**2)
                     for neighbor in neighbors for delta in [neighbor - vector])
        vector = vector + .19 * change
    return vector * np.array([100, 65, 65], np.float32)


def filter_color(rgb, method):
    if method == 'Bilateral':
        return cv2.bilateralFilter(rgb, 7, 22, 4)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255., cv2.COLOR_RGB2Lab)
    if method == 'Kuwahara':
        lab = kuwahara_lab(lab)
    elif method == 'Difusión anisotrópica':
        lab = anisotropic_lab(lab)
    elif method == 'Multiescala':
        fine = cv2.GaussianBlur(lab, (0, 0), .68)
        coarse = cv2.GaussianBlur(lab, (0, 0), 1.7)
        strength = scale(grad(lab[..., 0], .7))
        lab = (1 - strength[..., None]) * (.50 * fine + .50 * coarse) + strength[..., None] * fine
    elif method == 'Reconstrucción':
        light = lab[..., 0]
        opening = morphology.reconstruction(cv2.erode(light, ell(2)), light, method='dilation')
        closing = morphology.reconstruction(cv2.dilate(light, ell(2)), light, method='erosion')
        lab[..., 0] = .50 * light + .25 * opening + .25 * closing
    else:
        raise ValueError(method)
    floating = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.rint(floating.clip(0, 1) * 255).astype(np.uint8)


def preparar_territorios_de_tincion(rgb, normalization='Original', filtering='Bilateral', targets=None):
    target_lab, target_stain = targets if targets is not None else (None, None)
    normalized, status = normalize_stain(rgb, normalization, target_lab, target_stain)
    maps = mapas_region_tincion(rgb, filter_color(normalized, filtering), use_texture=True)
    maps['normalized'] = normalized
    maps['normalization_status'] = status
    maps['score'] = .40 * maps['ratio'] + .23 * maps['od'] + .23 * maps['h'] + .14 * maps['local']
    return maps


def jerarquia_pulmon(maps, mode, step=0):
    if mode == 'Textura y Felzenszwalb':
        field = jerarquia_regional(maps, 'Felzenszwalb', use_texture=True, limit=.42, step=step, use_outer=True)
    elif mode in ('H-maxima y fusión jerárquica', 'Territorios por coste cromático'):
        field = jerarquia_regional(maps, 'H-maxima de espacio libre', limit=.40, step=step, use_outer=True)
    else:
        field = jerarquia_regional(maps, 'H-minima', limit=.42, step=step, use_outer=True)
    field['initial_id'] = anchor_region(field['labels'], maps['score'], maps['tissue'])
    return field


def segmentar_territorios_de_tincion(maps, field, mode, step=0):
    score = maps['score']
    labels = field['labels']
    groups, candidates, strong = candidatos_oscuros(score, maps, .859, .096, 780, step)
    primary = foco_central(groups, score, labels, field['initial_id'], 37, maps['l'])
    blank = np.zeros(score.shape, bool)
    if primary is None:
        return audited_result(blank, blank, uncertain=candidates, field=field, candidate=candidates,
                              initial=blank, strong=strong)
    accept, unsure = associate5(maps, score, groups, primary, field, 14, .55, minimum=.14)
    chrom = reject_union([groups[i] for i in accept], score.shape)
    competing, rival = boundary_rivals(labels, groups, accept + unsure)
    primary_span = np.bincount(labels[groups[primary]])
    primary_span[0] = 0
    anchor = primary_span.argmax() if primary_span.any() else field['initial_id']
    candidates_ids = [anchor]
    if field['initial_id'] and field['initial_id'] != anchor:
        independently_selected = labels == field['initial_id']
        if (independently_selected & groups[primary]).sum() >= .25 * groups[primary].sum():
            pair_contact = expand(labels == anchor, 1) & independently_selected
            if pair_contact.any() and np.median(field['edge'][pair_contact]) < .44:
                candidates_ids.append(field['initial_id'])
    raw = np.isin(labels, [i for i in candidates_ids if i])
    domain = body_from_regions(field, maps, raw, rival, .385 + .012 * step, 1)
    domain &= ~competing | raw
    if mode == 'Territorios por coste cromático':
        ownership, markers, costs = territorios_por_coste(maps, field, groups, accept, raw, step)
        competing = ownership > 2
        cell = (ownership == 2) & maps['tissue']
        raw = cell.copy()
        field = {**field, 'competitive': ownership}
    elif mode == 'Contorno vectorial externo':
        allowed = maps['tissue'] & ~FRAME & (~competing | raw)
        cell = contour5(maps, domain, allowed, shrink(raw, 2), field['edge'], field['vector'], 14 + 2 * step)
    else:
        cell = domain
    connected = attached(cell, raw)
    cell = small_holes(connected, maps['tissue'] & (~competing | raw), 10)
    ambiguous = reject_union([groups[i] for i in unsure], score.shape)
    return audited_result(cell, chrom, competing, ambiguous, field=field, candidate=candidates,
                          initial=raw, strong=strong)


def asociacion_pulmon_multicanal(maps, previous, graph, foreground):
    vote, signals, dark = all_dark6(maps, previous)
    score = (signals['LAB'] + signals['H&E'] + signals['Blue Ratio']) / 3
    anchor = foco_compacto(previous)
    old = previous['chromatin']
    if not anchor.any():
        groups = components(dark & (RADIAL < 32), 12)
        if groups:
            anchor = max(groups, key=lambda item: item[2] / (np.linalg.norm(item[3] - CENTER) + 7))[1]
        else:
            return old, dark, previous['omitted'], previous['excluded'], ()
    supported = attached(dark & (distance(old | anchor) <= 5.5)
                         & (score >= np.quantile(score[anchor], .17) - .13), old | anchor)
    selected = old | anchor | supported
    uncertain, rivals = previous['omitted'].copy(), previous['excluded'].copy()
    signatures = []
    for candidate in candidate_groups6(dark, score):
        pending = candidate & ~selected
        if pending.sum() < 12:
            continue
        gap, bridge = interval(anchor, pending)
        radius = RADIAL[pending].min()
        barrier = float(maps['edge6'][bridge].mean()) if len(bridge[0]) else 0.
        stain = float(np.mean(score[pending]) - np.mean(score[anchor]))
        target_group = np.unique(foreground[anchor])
        compartment = np.mean(np.isin(foreground[pending], target_group[target_group >= 0]))
        if gap < 12 and radius < 26 and barrier < .30 and compartment > .53 and stain > -.14:
            selected |= pending
        elif gap < 18 and radius < 35 and stain > -.19:
            uncertain |= pending
        elif (candidate & FRAME).any() or gap >= 15:
            rivals |= pending
        else:
            uncertain |= pending
        if gap < 25:
            signatures.append((int(pending.sum()), round(gap, 1), round(barrier, 3),
                               round(compartment, 2), round(stain, 2)))
    return selected, dark | old, uncertain & ~selected, rivals & ~selected, signatures


def reconstruccion_relieve(maps, previous):
    vote, signals, candidate = all_dark6(maps, previous)
    old = previous['chromatin']
    if not old.any():
        return informacion_cromatinica(previous)
    score = .5*signals['LAB'] + .3*signals['Blue Ratio'] + .2*signals['H&E']
    local_floor = cv2.GaussianBlur(score, (0,0), 4.8)
    relief = score - local_floor
    support = ((score >= np.quantile(score[old], .13) - .10) &
               (relief >= np.quantile(relief[old], .17) - .07))
    near_core = distance(old)
    raw = (support & (near_core <= 7.5) & maps['tissue']) | old
    recovered = attached(raw, old) & ~previous['excluded']
    # Component boundaries and their weaker local slopes stay visible in the audit.
    added = recovered & ~old
    weak = attached((near_core < 10) & (score >= np.quantile(score[old], .10)-.15), old)
    return (old | added, candidate | weak | old, (previous['omitted'] | (weak & ~recovered)) & ~old,
            previous['excluded'] & ~(old | added), ())


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
    wide_maps = mapas_tincion_multiescala(rgb)
    mode = 'Multi-Otsu y competencia regional'
    initial = preparar_competencia_multicanal(wide_maps, mode)
    wide = segmentar_competencia_multicanal(wide_maps, initial, territorios_multicanal(wide_maps, initial, mode))
    wide = full_chromatin(wide)
    terrain = preparar_territorios_de_tincion(rgb)
    mode = 'Territorios por coste cromático'
    compact = segmentar_territorios_de_tincion(terrain, jerarquia_pulmon(terrain, mode), mode)
    ctx = new_context(mapas_apariencia(rgb), compact)
    maps, anchor = ctx['maps'], ctx['anchor']
    shared = np.dstack((maps['lab6'], .65*maps['e'], .37*maps['std']))
    lattice = grafo_multicanal(ctx, shared, n=390, compactness=.087)
    fusion = fusion_adyacente(lattice, shared, .79)
    grouped = asociacion_pulmon_multicanal(maps, compact, lattice, fusion[lattice['labels']])
    relief = reconstruccion_relieve(maps, compact)
    chrom_routes = [
        with_detection(compact, ctx['base'], 'territorio cromático'),
        with_detection(compact, grouped, 'asociación multicanal'),
        with_detection(compact, relief, 'reconstrucción por relieve local'),
        with_detection(compact, informacion_cromatinica(wide), 'Multi-Otsu y agrupación regional'),
    ]
    focus = select_chromatin(source, chrom_routes, required_gain=.023)
    detection = informacion_cromatinica(focus)
    band, graph, density = corte_en_banda(ctx, detection, shared)
    lab_model = modelos_de_apariencia(ctx, lattice, maps['lab6'], detection)
    likely = (lab_model['probability'] > .52) & ~lab_model['other'][lattice['labels']]
    likely = attached(likely | lab_model['target'][lattice['labels']], focus['chromatin'])
    spatial = corte_espacial(lattice, lab_model, smoothing=1.25, offset=.13)
    body = [
        label_body(compact['cell'], detection, compact, 'territorio cromático'),
        label_body(wide['cell'], detection, wide, 'competencia regional Multi-Otsu'),
        label_body(band, detection, compact, 'corte de banda local', graph, density),
        label_body(likely, detection, compact, 'apariencia regional LAB', lattice, lab_model),
        label_body(spatial, detection, compact, 'corte de grafos LAB', lattice, lab_model),
    ]
    result = select_body(source, body)
    return result, f"Cromatina por {focus['method']}. Cuerpo por {result['method']}"
