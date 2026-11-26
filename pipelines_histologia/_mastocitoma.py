"""Pipeline RGB consolidado para mastocitoma."""


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
    anchor_region,
    associate_components,
    associate_v2,
    attached,
    audited_result,
    base_maps,
    border_support,
    build_markers,
    cell_raw,
    chromatin_candidates,
    chromatin_response,
    color,
    territorios_por_coste,
    components,
    contour5,
    cv2,
    candidatos_oscuros,
    dilate,
    disk,
    distance,
    distance_to,
    ell,
    erode,
    estimate_cell,
    expand,
    feature,
    field_means,
    filters,
    grad,
    jerarquia_regional,
    historical_holes,
    historical_objects,
    image_features,
    interval,
    landscape_markers,
    made,
    measure,
    merge_hierarchy,
    morphology,
    ndi,
    np,
    pair_evidence,
    pd,
    preprocess,
    foco_central,
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
    threshold_result,
    u8,
    upper_contacts,
    mapas_tincion_multiescala,
    v4_parcels,
    mapas_region_tincion,
)


CONFIG = {'map': 'he_blend', 'threshold': 'hysteresis', 'sigma': 0.55, 'clahe': False, 'offset': 0.1, 'high_min': 0.56, 'high_max': 0.82, 'hysteresis_width': 0.13, 'min_area': 5, 'split_area': 850, 'anchor_radius': 30, 'group_gap': 10, 'group_radius': 29, 'cell_method': 'geodesic', 'color_weight': 0.16, 'edge_weight': 10.0, 'seed_expand': 1, 'compactness': 0.0004, 'geodesic_bias': 0.95, 'contour_iterations': 12}


SETTINGS = {'stain_mix': (0.56, 0.28), 'edge_mix': (0.43, 0.37, 0.2), 'sigma': 0.6, 'context_sigma': 7, 'local_boost': 0.32, 'high_range': (0.54, 0.86), 'offset': 0.11, 'hysteresis': 0.12, 'fragment_area': 5, 'split_area': 940, 'anchor_radius': 29, 'gap': 12, 'group_radius': 29, 'pair_center': 12, 'barrier': 0.58, 'stain_drop': 0.24, 'bridge_drop': 0.27, 'ring': 8, 'ring_similarity': 0.49, 'seed_edge': 0.45, 'grabcut_iterations': 3}


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


def mast_maps(rgb):
    reference = legacy_second(rgb)
    maps = base_maps(rgb, (5, 19, 3))
    fine = scale(np.hypot(grad(maps['a']), grad(maps['b'])), 4, 99)
    maps['edge'] = .60 * maps['edge'] + .40 * fine
    return maps, reference


def resolve_mast_band(maps, ref, width=5, mode='GrabCut local'):
    cell, chrom = ref['cell'], ref['chromatin']
    neck_opened = cv2.morphologyEx(cell.astype(np.uint8), cv2.MORPH_OPEN, ell(2)).astype(bool)
    stem = attached(neck_opened, expand(chrom, 2))
    edge = maps['edge']
    if not cell.any() or not stem.any():
        return cell.copy(), np.zeros_like(cell), np.zeros_like(cell)
    candidates = ref['labels'] > 0
    remote = (candidates & ~chrom & ~ref['ambiguous'] & ~expand(cell, 2))
    domain = (expand(cell, width) & maps['tissue'] & ~remote & ~FRAME)
    core = (shrink(stem, 2) | (chrom & expand(stem, 3))) & domain
    if mode == 'Reconstrucción de prolongaciones':
        support = expand(stem, 2) | (cell & (cv2.dilate(edge, ell(1)) >= .40))
        return attached(cell & support, stem), domain, core
    gc = np.full(cell.shape, cv2.GC_BGD, np.uint8)
    gc[domain] = cv2.GC_PR_BGD
    gc[cell & domain] = cv2.GC_PR_FGD
    gc[core] = cv2.GC_FGD
    if min(np.count_nonzero(gc > 0), np.count_nonzero(gc == 1)) < 5:
        return cell.copy(), domain, core
    color_input = cv2.cvtColor(maps['smooth'], cv2.COLOR_RGB2LAB)
    cv2.setRNGSeed(SEED)
    cv2.grabCut(color_input, gc, None, np.zeros((1, 65)), np.zeros((1, 65)),
                2, cv2.GC_INIT_WITH_MASK)
    proposal = (gc & 1) != 0
    unsupported_neck = cell & ~expand(stem, 2) & (cv2.dilate(edge, ell(1)) < .42)
    proposal &= ~(unsupported_neck & ~core)
    proposal = attached(proposal, core)
    return proposal, domain, core


def mast_groups(raw, domain, core, maps, ref, recover=True):
    cell = small_holes(raw, domain, 7)
    chrom = ref['chromatin'].copy()
    neighbors = np.zeros_like(cell)
    omitted = np.zeros_like(cell)
    labels = ref['labels']
    review_groups = [labels == idx for idx in np.unique(labels[labels > 0])]
    if chrom.any() and cell.any():
        # Search within observable cell support for small dark groups missed by the historical threshold.
        darkness = .69 * maps['h'] + .31 * (1 - maps['l'])
        strong = max(.54, float(np.quantile(darkness[chrom], .40)) - .065)
        available = domain & ~expand(chrom, 1) & (labels == 0)
        high = available & (darkness >= strong)
        faint = available & (darkness >= strong - .05)
        review_groups.extend(mask for _, mask, area, _ in components(attached(faint, high), 6)
                             if area <= 95 and (mask & high).sum() >= 4)
    for component in review_groups:
        if (component & chrom).any():
            continue
        if not chrom.any():
            neighbors |= component
            continue
        gap, path = interval(chrom, component)
        overlap = (cell & component).sum() / component.sum()
        barrier = np.quantile(maps['edge'][path], .75) if path[0].size else 1.
        midpoint = (np.argwhere(component).mean(axis=0) + np.argwhere(chrom).mean(axis=0)) / 2
        same_stain = np.quantile(maps['h'][component], .8) >= np.quantile(maps['h'][chrom], .8) - .17
        eligible = gap <= 12 and np.linalg.norm(midpoint - CENTER) <= 18 and same_stain and barrier < .51
        if recover and eligible and overlap >= .90 and not (component & FRAME).any():
            chrom |= component & cell
        elif eligible or (gap <= 22 and np.linalg.norm(midpoint - CENTER) <= 13
                          and same_stain and np.quantile(maps['h'][component], .8) >= .62):
            omitted |= component
        else:
            neighbors |= component
    return made(cell, chrom, neighbors, omitted, domain=domain, core=core, raw=raw)


def run_mast(rgb, method='GrabCut local', perturb=0):
    maps, ref = mast_maps(rgb)
    raw, domain, core = resolve_mast_band(maps, ref, 5 + perturb, method)
    return mast_groups(raw, domain, core, maps, ref, method == 'GrabCut local')


def preparar_reconstruccion_mastocitoma(maps, old, mode, step=0):
    s = maps['response']
    response = s['BR · H · contraste'] if mode == 'Reconstrucción BR/H' else s['H normalizada']
    support = old['cell'] & maps['tissue']
    local = response[support]
    if len(local) < 12:
        empty = np.zeros(old['cell'].shape, bool)
        return response, empty, empty, empty
    global_levels = threshold_result(response, 'Otsu')[1]
    level = (np.quantile(local, .86) if mode == 'Reconstrucción BR/H' or not global_levels
             else global_levels[0])
    high = (response >= level + .018 * step) & support
    low = (response >= level - (.105 if mode == 'Reconstrucción BR/H' else .018) + .018 * step) & support
    weak = low & ((s['LoG'] > .12) | (response > level - .045))
    return response, high, weak, attached(weak, high)


def reconstruir_mastocitoma(maps, old, prepared, mode):
    score, high, weak, reconstructed = prepared
    existing = old['chromatin'].copy()
    cell = old['cell'].copy()
    additions, doubtful = np.zeros_like(cell), np.zeros_like(cell)
    for _, part, area, center in components(reconstructed & ~existing, 7):
        if not existing.any():
            doubtful |= part
            continue
        gap, path = interval(existing, part)
        edge = float(np.quantile(maps['edge4'][path], .8)) if path[0].size else .0
        overlap = (part & shrink(cell)).sum() / area
        strong_share = (part & high).sum() / area
        supported = area >= 10 and gap <= 9 and overlap >= .85 and strong_share >= .22 and edge < .48
        if supported and (part & ~old['excluded']).sum() == area:
            additions |= part
        else:
            doubtful |= part
    return made(cell, existing | additions, old['excluded'], old['omitted'] | doubtful,
                candidate=reconstructed, seeds=high.astype(np.int32), domain=weak, added=additions)


def preparar_rescate_mastocitoma(maps, previous, mode, step=0):
    score = .42 * maps['ratio'] + .33 * maps['h'] + .18 * maps['od'] + .07 * maps['log']
    groups, loose, high = candidatos_oscuros(score, maps, .828, .120, 900, step)
    field = jerarquia_regional(maps, 'H-maxima de espacio libre', limit=.405, step=step)
    return score, groups, loose, high, field


def rescatar_mastocitoma(maps, previous, prepared, mode, step=0):
    score, groups, candidate, high, field = prepared
    chrom = previous['chromatin'].copy()
    cell = previous['cell'].copy()
    excluded = previous['excluded'].copy()
    pending = previous['omitted'].copy()
    labels = field['labels']
    initial = cell.copy()
    if not chrom.any() and mode == 'BR/H con rescate acotado':
        primary = foco_central(groups, score, labels, anchor_region(labels, score, maps['tissue']), 33, maps['l'])
        if primary is not None:
            nucleus = groups[primary]
            votes = np.bincount(labels[nucleus])
            votes[0] = 0
            initial = labels == votes.argmax() if votes.any() else initial
            ownership, markers, costs = territorios_por_coste(maps, field, groups, [primary], initial)
            rival_territory = ownership > 2
            proposed = (ownership == 2) & maps['tissue']
            near_groups = [part for i, part in enumerate(groups) if i != primary and RADIAL[part].min() <= 33]
            runner_up = max((np.quantile(score[part], .82) for part in near_groups), default=0.)
            dominant = np.quantile(score[nucleus], .83) >= runner_up + .025
            border_floor = max(.19, .94 * float(np.median(field['edge'][maps['tissue']])))
            if ((proposed & nucleus).sum() >= .75 * nucleus.sum() and (high & nucleus).sum() >= .26 * nucleus.sum()
                    and dominant and np.quantile(1 - maps['l'][nucleus], .72) > .65
                    and not (nucleus & FRAME).any()
                    and border_support(proposed, nucleus, field['edge']) >= border_floor):
                return audited_result(proposed, nucleus, rival_territory, pending, field=field,
                                      candidate=candidate, initial=initial, strong=high, rescued=True)
            pending |= nucleus
    if chrom.any() and cell.any():
        anchor = max(components(chrom), key=lambda item: item[2])[1]
        anchor_labels = np.bincount(labels[anchor])
        anchor_labels[0] = 0
        for part in groups:
            if (part & chrom).sum() >= .23 * part.sum():
                continue
            gap, location, barrier, valley, exterior = pair_evidence(maps, score, anchor, part, field['edge'])
            same_territory = bool(anchor_labels.any() and (labels[part] == anchor_labels.argmax()).mean() > .58)
            stain = np.quantile(score[part], .82)
            similar = stain >= np.quantile(score[anchor], .82) - .11
            if (location < 19 and gap <= 16 and barrier < .48 and valley > .38 and similar
                    and exterior < .13 and part.sum() >= 14 and (high & part).sum() >= .25 * part.sum()
                    and ((part & cell).mean() > 0 or same_territory)
                    and ((part & cell).sum() >= .63 * part.sum() or same_territory)):
                chrom |= part
            elif location <= 22 and gap <= 22 and similar and part.sum() >= 12:
                pending |= part
        if mode == 'BR/H con rescate acotado':
            allowed = expand(cell, 3) & maps['tissue'] & ~excluded & ~FRAME
            revised = contour5(maps, cell, allowed, shrink(cell, 2), field['edge'], field['vector'], 6 + step)
            if ((cell ^ revised).sum() <= .10 * cell.sum()
                    and ((chrom & ~revised).sum() <= (chrom & ~cell).sum())
                    and border_support(revised, chrom, field['edge']) >= border_support(cell, chrom, field['edge']) + .015):
                cell = revised
    return audited_result(cell, chrom, excluded, pending & ~chrom, field=field, candidate=candidate,
                          initial=initial, strong=high)


def segmentar_recorte(rgb, source):
    compact_mode = 'Reconstrucción BR/H'
    local = mapas_tincion_multiescala(rgb)
    seeds = run_mast(rgb, 'GrabCut local')
    initial = reconstruir_mastocitoma(local, seeds, preparar_reconstruccion_mastocitoma(local, seeds, compact_mode), compact_mode)
    maps = mapas_region_tincion(rgb)
    mode = 'BR/H con rescate acotado'
    result = rescatar_mastocitoma(maps, initial, preparar_rescate_mastocitoma(maps, initial, mode), mode)
    return result, 'Reconstrucción BR/H, GrabCut local y rescate acotado del foco oscuro'
