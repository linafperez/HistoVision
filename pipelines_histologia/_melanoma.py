"""Pipeline RGB consolidado para melanoma."""


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
    associate5,
    associate_components,
    associate_v2,
    attached,
    audited_result,
    base_maps,
    informacion_cromatinica,
    body_from_regions,
    border_support,
    corte_en_banda,
    boundary_rivals,
    breadth_first_order,
    build_markers,
    cell_raw,
    chromatin_candidates,
    chromatin_response,
    class_probabilities,
    cluster_options,
    color,
    foco_compacto,
    territorios_por_coste,
    components,
    coverage6,
    cv2,
    candidatos_oscuros,
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
    maximum_flow,
    mean_shift_centers,
    measure,
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
    u8,
    upper_contacts,
    v4_contacts,
    mapas_tincion_multiescala,
    v4_means,
    v4_parcels,
    mapas_region_tincion,
    formar_resultado_regional,
)


CONFIG = {'map': 'blue_dark', 'threshold': 'hysteresis', 'sigma': 0.45, 'clahe': False, 'offset': 0.11, 'high_min': 0.57, 'high_max': 0.82, 'hysteresis_width': 0.13, 'min_area': 5, 'split_area': 850, 'anchor_radius': 30, 'group_gap': 14, 'group_radius': 32, 'cell_method': 'geodesic', 'color_weight': 0.15, 'edge_weight': 12, 'seed_expand': 1, 'compactness': 0.0005, 'geodesic_bias': 0.95, 'contour_iterations': 12}


SETTINGS = {'stain_mix': (0.18, 0.57), 'edge_mix': (0.49, 0.23, 0.28), 'sigma': 0.55, 'context_sigma': 9, 'local_boost': 0.48, 'high_range': (0.59, 0.86), 'offset': 0.12, 'hysteresis': 0.1, 'fragment_area': 5, 'split_area': 825, 'anchor_radius': 29, 'gap': 13, 'group_radius': 31, 'pair_center': 13, 'barrier': 0.55, 'stain_drop': 0.2, 'bridge_drop': 0.27, 'ring': 9, 'ring_similarity': 0.51, 'seed_edge': 0.45, 'grabcut_iterations': 3}


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


def melanoma_maps(rgb):
    maps = base_maps(rgb, (5, 20, 3))
    dark = scale(grad(maps['h'], .9), 6, 99)
    maps['stop'] = .67 * maps['edge'] + .33 * dark
    return maps


def melanoma_markers(maps, reference, width=4, review_groups=True):
    cell, chrom = reference['cell'], reference['chromatin']
    core = shrink(cell, 2) | (chrom & expand(cell))
    omitted = np.zeros_like(cell)
    excluded = np.zeros_like(cell)
    kept = chrom.copy()
    primary = max(components(chrom), key=lambda item: item[2])[1] if chrom.any() else chrom
    for idx in np.unique(reference['labels'][reference['labels'] > 0]):
        part = reference['labels'] == idx
        if not primary.any() or (part & primary).any():
            continue
        gap, path = interval(primary, part)
        barrier = np.quantile(maps['stop'][path], .8) if path[0].size else 1
        fraction = (part & cell).sum() / part.sum()
        if float(np.min(RADIAL[part])) <= 26 and (part & chrom).any() and review_groups:
            # Reassign a fringe group only when multiple independent signs agree.
            if (fraction < .18 and gap > 5 and barrier > .52
                    and not (part & shrink(cell)).any()):
                kept &= ~part
                excluded |= part
            elif fraction < .65:
                omitted |= part
        elif not (part & chrom).any() and gap <= 20:
            if fraction < .30 or barrier > .53:
                excluded |= part
            else:
                omitted |= part
    allowed = expand(cell, width) & maps['tissue'] & ~excluded & ~FRAME
    return kept, (core & allowed & ~excluded), allowed, excluded, omitted


def melanoma_contour(maps, reference, core, allowed, variant='Contorno local'):
    cell = reference['cell']
    if not core.any():
        return cell.copy()
    if variant == 'Reconstrucción local':
        opening = cv2.morphologyEx(cell.astype(np.uint8), cv2.MORPH_OPEN, ell(2)).astype(bool)
        edge = cv2.dilate(maps['stop'], ell(1))
        candidate = allowed & (opening | core | (cell & (edge > .38)))
        return attached(candidate, core)
    stop = 1 / (1 + 6.5 * maps['stop'])

    def constrain(level):
        level[~allowed] = 0
        level[core] = 1

    return segmentation.morphological_geodesic_active_contour(
        stop, 12, cell & allowed, smoothing=1, threshold=.40, balloon=0,
        iter_callback=constrain).astype(bool)


def melanoma_finish(raw, maps, reference, kept, core, allowed, excluded, omitted, conservative):
    proposal = small_holes(attached(raw, core), allowed, 7) if core.any() else raw
    cell = reference['cell']
    border = cv2.dilate(maps['stop'], ell(1))
    new_rim, old_rim = proposal & ~shrink(proposal), cell & ~shrink(cell)
    old_support = border[old_rim].mean() if old_rim.any() else 0.
    new_support = border[new_rim].mean() if new_rim.any() else 0.
    change = (cell ^ proposal).sum()
    accept = (proposal.any() and change <= .22 * max(cell.sum(), 1)
              and (new_support >= old_support + .012 or (cell & excluded).any())
              and (kept & ~proposal).sum() <= (reference['chromatin'] & ~cell).sum())
    if conservative and not accept:
        return made(cell.copy(), reference['chromatin'].copy(), excluded, omitted,
                    raw=raw, domain=allowed, accepted=False)
    return made(proposal, kept, excluded, omitted, raw=raw, domain=allowed, accepted=True)


def run_melanoma(rgb, method='Cambio local con control', perturb=0):
    maps = melanoma_maps(rgb)
    ref = legacy_second(rgb)
    kept, core, allowed, excluded, omitted = melanoma_markers(maps, ref, 4 + perturb)
    raw = melanoma_contour(maps, ref, core, allowed, method)
    return melanoma_finish(raw, maps, ref, kept, core, allowed, excluded, omitted,
                           method == 'Cambio local con control')


def preparar_reconstruccion_melanoma(maps, old, mode, step=0):
    s = maps['response']
    previous = old['chromatin']
    cell = old['cell']
    channel = .58 * s['Blue Ratio'] + .23 * s['Contraste local'] + .19 * s['H']
    sample = channel[cell]
    if sample.size < 12:
        blank = np.zeros_like(cell)
        return channel, blank, blank, blank
    level = np.quantile(sample, .86) + .016 * step
    strong = (channel >= level) & cell
    weak = (channel >= level - .092) & cell
    if mode == 'Fuzzy cromático':
        _, probs = cluster_options(maps, 'Fuzzy C-means', 4, 'BR+H+OD')
        strong &= probs[..., 0] >= .5
        weak &= probs[..., :2].sum(axis=-1) >= .56
    recon = attached(weak, strong)
    return channel, strong, weak, recon


def reconstruccion_melanoma(maps, old, prepared, mode):
    channel, strong, domain, recon = prepared
    cell = old['cell'].copy()
    previous = old['chromatin']
    result = previous.copy()
    pending = old['omitted'].copy()
    for _, fragment, area, _ in components(recon & ~previous, 8):
        if not previous.any():
            pending |= fragment
            continue
        gap, path = interval(previous, fragment)
        barrier = np.quantile(maps['edge4'][path], .75) if path[0].size else 0.
        if ((fragment & shrink(cell, 2)).sum() >= .88 * area
                and (strong & fragment).sum() >= .28 * area and gap <= 8 and barrier < .42):
            result |= fragment
        else:
            pending |= fragment
    # Preserve peripheral groups unless they have little support in the cell AND a strong barrier.
    if previous.any():
        primary = max(components(previous), key=lambda item: item[2])[1]
        for _, part, area, _ in components(previous & ~primary, 5):
            gap, path = interval(primary, part)
            barrier = np.quantile(maps['edge4'][path], .80) if path[0].size else 0.
            coverage = (part & cell).sum() / area
            if coverage < .28 and gap > 6 and barrier > .52 and not (part & strong).any():
                result &= ~part
                pending |= part
    return made(cell, result, old['excluded'], pending, candidate=recon, seeds=strong.astype(np.int32),
                domain=domain, added=result & ~previous)


def preparar_focos_densos(maps, mode, step=0):
    if mode == 'Foco OD/BR de campo completo':
        score = .39 * maps['od'] + .32 * maps['ratio'] + .17 * maps['h'] + .12 * maps['local']
    else:
        score = .30 * maps['ratio'] + .25 * maps['od'] + .22 * maps['h'] + .15 * maps['local'] + .08 * maps['log']
    field = jerarquia_regional(maps, 'H-maxima de espacio libre', use_texture=True, limit=.42, step=step)
    groups, candidate, strong = candidatos_oscuros(score, maps, .855, .102, 820, step)
    if mode in ('Focos por máximos de densidad óptica', 'Revisión de focos externos dominantes'):
        score = .47 * maps['od'] + .26 * maps['ratio'] + .20 * maps['local'] + .07 * maps['log']
        groups, candidate, strong = candidatos_oscuros(score, maps, .864, .079, 700, step)
        # Isolate high-density cores in a merged plateau before choosing an event.
        compact = []
        for part in groups:
            threshold = np.quantile(maps['od'][part], .47 if mode == 'Revisión de focos externos dominantes' else .54)
            core = part & (maps['od'] >= threshold)
            compact.extend(mask for _, mask, area, _ in components(core, 18))
        groups = compact or groups
    primary = foco_central(groups, score, reach=40, light=maps['l'], offset_penalty=.36)
    return score, field, groups, candidate, strong, primary


def seleccionar_foco_melanoma(maps, prepared, previous, mode, step=0):
    score, field, parts, candidate, strong, primary = prepared
    old_chrom = previous['chromatin'].copy()
    cell = previous['cell'].copy()
    chrom = old_chrom.copy()
    initial = cell.copy()
    pending = previous['omitted'].copy()
    excluded = previous['excluded'].copy()
    test_external = mode == 'Revisión de focos externos dominantes'
    if test_external:
        outside_parts = [
            idx for idx, part in enumerate(parts)
            if RADIAL[part].min() <= 34 and (part & cell).sum() <= .55 * part.sum()
            and not (part & FRAME).any() and part.sum() >= 28
        ]
        if outside_parts:
            variants = [parts[i] for i in outside_parts]
            best = foco_central(variants, score, reach=35, light=maps['l'], offset_penalty=.30)
            if best is not None:
                primary = outside_parts[best]
    if primary is None:
        return audited_result(cell, chrom, excluded, pending, candidate=candidate, initial=initial,
                              field=field, strong=strong)
    chosen = parts[primary]
    old_focus = max(components(old_chrom), key=lambda item: item[2])[1] if old_chrom.any() else old_chrom
    chosen_score = np.quantile(score[chosen], .76)
    old_score = np.quantile(score[old_focus], .76) if old_focus.any() else 0.
    new_darkness = np.quantile(1 - maps['l'][chosen], .65)
    old_darkness = np.quantile(1 - maps['l'][old_focus], .65) if old_focus.any() else 0.
    distinct = (chosen & expand(old_chrom, 2)).sum() < .30 * chosen.sum()
    strong_fraction = (strong & chosen).sum() / chosen.sum()
    release_plateau = mode == 'Focos por máximos de densidad óptica'
    stain_gain = chosen_score > old_score + .066
    if release_plateau:
        stain_gain = chosen_score >= old_score - .030
    recovery = (test_external and distinct and chosen_score >= old_score - .048
                and new_darkness >= old_darkness - .015 and strong_fraction >= .42
                and chosen.sum() >= max(32, 1.24 * old_focus.sum())
                and RADIAL[chosen].min() <= 33 and (chosen & cell).sum() <= .50 * chosen.sum())
    if recovery or (distinct and stain_gain and new_darkness > old_darkness + (.019 if release_plateau else .05)
            and strong_fraction > .38 and chosen.sum() >= max(24, .50 * old_focus.sum())
            and RADIAL[chosen].min() <= 33 and not (chosen & FRAME).any()):
        accepted, ambiguity = associate5(maps, score, parts, primary, field, 13, .50, .15, strict=True)
        chrom = reject_union([parts[idx] for idx in accepted], score.shape)
        pending |= reject_union([parts[idx] for idx in ambiguity], score.shape) | old_chrom
        labels = field['labels']
        votes = np.bincount(labels[chosen])
        votes[0] = 0
        if votes.any():
            initial = labels == votes.argmax()
            competing, rivals = boundary_rivals(labels, parts, accepted + ambiguity)
            proposal = body_from_regions(field, maps, initial, rivals, .385, 1) & (~competing | initial)
            if (proposal & chosen).sum() >= .75 * chosen.sum() and border_support(proposal, chosen, maps['edge']) >= .39:
                cell, excluded = proposal, competing
            elif release_plateau or test_external:
                ownership, _, _ = territorios_por_coste(maps, field, parts, accepted, initial, step)
                proposed_cell = (ownership == 2) & maps['tissue']
                if (proposed_cell & chosen).sum() >= .85 * chosen.sum() and border_support(
                        proposed_cell, chosen, maps['edge']) >= .30:
                    cell, excluded = proposed_cell, ownership > 2
    elif distinct and strong_fraction >= .28 and chosen_score >= old_score - (.075 if test_external else .02):
        pending |= chosen
    return audited_result(cell, chrom, excluded, pending & ~chrom, field=field, candidate=candidate,
                          initial=initial, strong=strong, changed=not np.array_equal(chrom, old_chrom))


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


def segmentar_recorte(rgb, source):
    initial = run_melanoma(rgb,'Cambio local con control')
    color_maps = mapas_tincion_multiescala(rgb)
    base = reconstruccion_melanoma(color_maps,initial,preparar_reconstruccion_melanoma(color_maps,initial,'Doble umbral BR'),'Doble umbral BR')
    stain = mapas_region_tincion(rgb,use_texture=True)
    mode = 'Revisión de focos externos dominantes'
    result = seleccionar_foco_melanoma(stain,preparar_focos_densos(stain,mode),base,mode)
    description = 'Foco dominante de densidad óptica y BR con control de competidores'
    chrom = result['chromatin']
    if result.get('changed') and result['cell'].sum() < 1.6*max(1,chrom.sum()):
        maps = mapas_apariencia(rgb)
        ctx = new_context(maps,result)
        vectors = np.dstack((maps['lab6'],maps['e'],.5*maps['std']))
        raw,graph,model = corte_en_banda(ctx,ctx['base'],vectors,16,maps['outer'])
        proposed = formar_resultado_regional(raw,ctx['base'],result,graph,model)
        body = proposed['cell']
        added = body & ~result['cell'] & ~chrom
        envelope = body & ~shrink(body) & ~expand(chrom,2)
        support = cv2.dilate(maps['edge6'],ell(1))
        if (added.any() and envelope.any() and (chrom & body).sum() > (chrom & result['cell']).sum()
                and not np.any(body & ~result['cell'] & FRAME) and np.count_nonzero(added & result['excluded']) <= 5
                and np.mean((~maps['tissue'])[added]) <= .07
                and np.median(model['probability'][added]) >= .65 and np.mean(support[envelope]) >= .28):
            return proposed, description + '. Cuerpo ampliado por banda exterior y GMM'
    return result, description
