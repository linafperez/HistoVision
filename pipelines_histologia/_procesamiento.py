"""Operaciones de segmentación compartidas por los pipelines RGB."""


from inspect import signature
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi, sparse
from scipy.sparse.linalg import spsolve
from skimage import color, filters, morphology, segmentation, measure, feature
from skimage.graph import MCP_Geometric
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from scipy.special import logsumexp, expit


SEED = 42


CENTER = np.array([63.5, 63.5])


YY, XX = np.indices((128, 128))


HAS_INCLUSIVE_MAX = 'max_size' in signature(morphology.remove_small_objects).parameters


GLOBAL_FILTERS = {'Yen': filters.threshold_yen, 'Li': filters.threshold_li,
                  'IsoData': filters.threshold_isodata, 'Minimum': filters.threshold_minimum}


FRAME = (YY == 0) | (XX == 0) | (YY == 127) | (XX == 127)


RADIAL = np.hypot(YY - CENTER[0], XX - CENTER[1])


def scale(values, lo=2, hi=98):
    low, high = np.percentile(values, [lo, hi])
    return np.clip((values - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def ell(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def expand(mask, radius=1):
    return cv2.dilate(mask.astype(np.uint8), ell(radius)).astype(bool)


def shrink(mask, radius=1):
    return cv2.erode(mask.astype(np.uint8), ell(radius), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0).astype(bool)


def distance(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def components(mask, minimum=1):
    count, ids, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(i, ids == i, int(stats[i, cv2.CC_STAT_AREA]), centers[i][::-1])
            for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]


def attached(mask, seeds):
    _, ids = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.unique(ids[seeds])
    return np.isin(ids, keep[keep > 0])


def small_holes(mask, allowed, max_area=9):
    for _, hole, area, _ in components(~mask):
        if area <= max_area and not np.any(hole & (FRAME | ~allowed)):
            mask = mask | hole
    return mask


def grad(channel, sigma=0):
    src = cv2.GaussianBlur(channel, (0, 0), sigma) if sigma else channel
    dx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    return cv2.magnitude(dx, dy)


def base_maps(rgb, bilateral=(7, 23, 4)):
    smooth = cv2.bilateralFilter(rgb, *bilateral)
    rgbf = smooth.astype(np.float32) / 255
    lab = cv2.cvtColor(rgbf, cv2.COLOR_RGB2Lab)
    hed = color.rgb2hed(rgbf)
    h, e = scale(hed[..., 0]), scale(hed[..., 1])
    luminance = lab[..., 0] / 100
    a, b = lab[..., 1] / 100, lab[..., 2] / 100
    color_edge = scale(np.sqrt(grad(a)**2 + grad(b)**2), 4, 99)
    edge = .45 * color_edge + .3 * scale(grad(luminance), 4, 99) + .25 * scale(grad(e), 4, 99)
    tissue = ~((luminance > .92) & (np.hypot(a, b) < .075) & (h < .28))
    return dict(rgb=rgb, smooth=smooth, rgbf=rgbf, lab=lab, h=h, e=e, l=luminance,
                a=a, b=b, edge=edge, tissue=tissue)


def interval(first, second):
    delta, nearest = ndi.distance_transform_edt(~first, return_indices=True)
    other = np.argwhere(second)
    endpoint = other[np.argmin(delta[second])]
    start = nearest[:, endpoint[0], endpoint[1]]
    coords = np.rint(np.linspace(start, endpoint, max(3, int(delta[tuple(endpoint)]) + 2))).astype(int)
    yy, xx = coords.T
    middle = ~(first | second)[yy, xx]
    return float(delta[tuple(endpoint)]), (yy[middle], xx[middle])


def made(cell, chromatin, excluded=None, omitted=None, **details):
    empty = np.zeros(cell.shape, bool)
    return dict(cell=cell, chromatin=chromatin, excluded=empty if excluded is None else excluded,
                omitted=empty if omitted is None else omitted, **details)


def historical_objects(mask, *, max_size):
    if HAS_INCLUSIVE_MAX:
        return morphology.remove_small_objects(mask, max_size=max_size)
    return morphology.remove_small_objects(mask, min_size=max_size + 1)


def historical_holes(mask, *, max_size):
    if HAS_INCLUSIVE_MAX:
        return morphology.remove_small_holes(mask, max_size=max_size)
    return morphology.remove_small_holes(mask, area_threshold=max_size + 1)


def robust_scale(values, lower=2, upper=99):
    """Rescale a working map by percentiles; never alter the source RGB array."""
    lo, hi = np.percentile(values, [lower, upper])
    return np.clip((values - lo) / max(float(hi - lo), 1e-8), 0, 1)


def u8(values):
    """Convert a map in [0, 1] to the OpenCV representation."""
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), disk(radius)).astype(bool)


def erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), disk(radius)).astype(bool)


def distance_to(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def regions(mask, minimum=1):
    """Label eight-connected components and discard objects below minimum."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= minimum
    keep[0] = False
    return np.where(keep[labels], labels, 0), stats, centers


def mapas_tincion_multiescala(rgb):
    """Describe staining, local contrast and transitions on the original field."""
    maps = base_maps(rgb, (5, 20, 3))
    r, g, b = np.moveaxis(maps['smooth'].astype(np.float32), -1, 0)
    br = (100 * b / (1 + r + g)) * (256 / (1 + r + g + b))
    lab = cv2.cvtColor(maps['rgbf'], cv2.COLOR_RGB2Lab)
    hsv = cv2.cvtColor(maps['rgbf'], cv2.COLOR_RGB2HSV)
    yc = cv2.cvtColor(maps['rgbf'], cv2.COLOR_RGB2YCrCb)
    density = -np.log((maps['smooth'].astype(np.float32) + 1) / 256)
    hraw = color.rgb2hed(maps['rgbf'])[..., 0]
    h_norm = scale(hraw, 8, 96)
    local = scale(cv2.GaussianBlur(maps['l'], (0, 0), 4) - maps['l'], 12, 99)
    darkness = scale(1 - maps['l'])
    normalized_l = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(u8(maps['l']))
    normalized_rgb = np.stack([scale(channel, 4, 97) for channel in (r, g, b)], axis=-1)
    br = scale(br)
    log_response = np.maximum.reduce([
        np.maximum(0., -sigma**2 * cv2.Laplacian(cv2.GaussianBlur(br, (0, 0), sigma), cv2.CV_32F))
        for sigma in (1.1, 2.4, 4.)])
    gabor = []
    for theta in np.arange(4) * np.pi / 4:
        kernel = cv2.getGaborKernel((13, 13), 2.65, theta, 6., .6, 0., ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        energy = np.abs(cv2.filter2D(darkness, cv2.CV_32F, kernel))
        gabor.append(scale(cv2.GaussianBlur(energy, (0, 0), 1.7)))
    gabor = np.stack(gabor, axis=-1)
    tx_edge = scale(np.sqrt(sum(grad(gabor[..., i], 1.)**2 for i in range(gabor.shape[-1]))))
    color_edge = scale(np.hypot(grad(lab[..., 1], .8), grad(lab[..., 2], .8)), 4, 99)
    sobel = scale(np.hypot(cv2.Sobel(maps['e'], cv2.CV_32F, 1, 0),
                           cv2.Sobel(maps['e'], cv2.CV_32F, 0, 1)))
    canny = cv2.Canny(u8(darkness), 45, 110) > 0
    morph_edge = scale(cv2.morphologyEx(maps['e'], cv2.MORPH_GRADIENT, ell(1)))
    edge = .55 * maps['edge'] + .26 * color_edge + .19 * sobel
    response = {
        'RGB-R': scale(255 - r), 'RGB-G': scale(255 - g), 'RGB-B': scale(255 - b),
        'HSV-S': scale(hsv[..., 1]), 'HSV-V inverso': scale(1 - hsv[..., 2]),
        'Lab-L inverso': darkness, 'Lab-b inverso': scale(-lab[..., 2]),
        'YCbCr-Cb': scale(yc[..., 2]), 'YCbCr-Y inverso': scale(1 - yc[..., 0]),
        'H': maps['h'], 'E': maps['e'], 'DO': scale(density.mean(axis=-1)), 'Blue Ratio': br,
        'H normalizada': h_norm, 'Contraste local': local, 'LoG': scale(log_response),
        'Lab-L CLAHE inverso': 1 - normalized_l.astype(np.float32) / 255,
        'RGB normalizado': 1 - cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2GRAY),
        'BR · H · contraste': .42 * br + .30 * h_norm + .28 * local,
        'E · Lab': .62 * maps['e'] + .38 * scale(lab[..., 1]),
        'Contraste · DO': .55 * local + .45 * scale(density.mean(axis=-1)),
    }
    cart = np.dstack((maps['l'], 1.6 * maps['a'], 1.6 * maps['b'], .65 * maps['e'], .7 * maps['h']))
    xy = np.dstack((.22 * (YY - 63.5) / 64, .22 * (XX - 63.5) / 64))
    return {**maps, 'response': response, 'edge4': edge, 'color_edge': color_edge, 'sobel': sobel,
            'canny': canny, 'morph_edge': morph_edge, 'gabor': gabor, 'tx_edge': tx_edge,
            'color_vector': cart, 'xy': xy}


def threshold_result(src, method):
    encoded = u8(src)
    if np.ptp(encoded) == 0:
        return None, (), None
    if method in ('Otsu', 'Triángulo'):
        flag = cv2.THRESH_OTSU if method == 'Otsu' else cv2.THRESH_TRIANGLE
        cutoff, mask = cv2.threshold(encoded, 0, 255, cv2.THRESH_BINARY | flag)
        return mask > 0, (cutoff / 255,), None
    if method in GLOBAL_FILTERS:
        try:
            level = float(GLOBAL_FILTERS[method](encoded)) / 255
        except (RuntimeError, ValueError, IndexError):
            return None, (), None
        return src > level, (level,), None
    if method == 'Multi-Otsu':
        if np.unique(encoded).size < 3:
            return None, (), None
        levels = filters.threshold_multiotsu(encoded, classes=3) / 255
        classes = np.digitize(src, levels)
        return classes == 2, levels, classes
    if method == 'Sauvola':
        threshold = filters.threshold_sauvola(1 - src, window_size=23, k=.13)
        return 1 - src < threshold, (), 1 - threshold
    if method == 'Adaptativo Gaussiano':
        mask = cv2.adaptiveThreshold(encoded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, -5)
        return mask > 0, (), None
    high = float(np.quantile(src, .90))
    low = max(0, high - .10)
    return attached(src >= low, src >= high), (low, high), None


def class_probabilities(values, centers):
    squares = ((values[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    return squares.clip(1e-8)


def mean_shift_centers(sample, bandwidth=.15):
    bin_id, population = np.unique(np.floor(sample / bandwidth).astype(np.int32), return_counts=True, axis=0)
    starts = (bin_id[np.argsort(population)[-36:]] + .5) * bandwidth
    modes = []
    for point in starts:
        for _ in range(18):
            members = np.linalg.norm(sample - point, axis=1) <= bandwidth
            if not members.any():
                break
            updated = sample[members].mean(axis=0)
            shift = np.linalg.norm(point - updated)
            point = updated
            if shift < .002:
                break
        if members.any():
            modes.append((int(members.sum()), point))
    retained = []
    for _, mode in sorted(modes, key=lambda item: -item[0]):
        if all(np.linalg.norm(mode - other) >= .66 * bandwidth for other in retained):
            retained.append(mode)
    return np.array(retained or [sample.mean(axis=0)])


def cluster_options(maps, algorithm, k, channels='LabHE'):
    features = maps['color_vector']
    if channels == 'LabHE+xy':
        features = np.dstack((features, maps['xy']))
    if channels == 'LabHE+Gabor':
        features = np.dstack((features, .29 * maps['gabor']))
    if channels == 'BR+H+OD':
        d = maps['response']
        features = np.dstack((d['Blue Ratio'], d['H normalizada'], d['DO'], .6 * d['Contraste local']))
    if channels == 'E+Lab':
        features = np.dstack((maps['l'], 1.5 * maps['a'], maps['e'], .38 * maps['h']))
    x = np.ascontiguousarray(features.reshape(-1, features.shape[-1]), dtype=np.float32)
    sample = np.ascontiguousarray(features[1::3, 1::3].reshape(-1, x.shape[1]), dtype=np.float32)
    cv2.setRNGSeed(SEED)
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .001)
    if algorithm == 'Mean Shift':
        centers = mean_shift_centers(sample, bandwidth=k)
        scores = -class_probabilities(x, centers) / (.5 * k**2)
    elif algorithm == 'GMM EM':
        model = cv2.ml.EM_create()
        model.setClustersNumber(k)
        model.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
        model.setTermCriteria(term)
        ok, _, _, _ = model.trainEM(sample)
        if not ok:
            raise RuntimeError('EM no produjo componentes')
        centers = model.getMeans()
        var = np.asarray([np.diag(matrix) for matrix in model.getCovs()]).clip(2e-4)
        log_weight = np.log(model.getWeights().ravel().clip(1e-8))
        scores = log_weight - .5 * (np.log(var).sum(axis=1)
                           + (((x[:, None, :] - centers)**2) / var).sum(axis=2))
    else:
        _, _, centers = cv2.kmeans(sample, k, None, term, 2, cv2.KMEANS_PP_CENTERS)
        if algorithm == 'Fuzzy C-means':
            for _ in range(24):
                inverse = 1 / class_probabilities(sample, centers)
                weights = (inverse / inverse.sum(axis=1, keepdims=True)) ** 2
                updated = (weights.T @ sample) / weights.sum(axis=0)[:, None]
                if np.max(np.abs(centers - updated)) < .001:
                    centers = updated
                    break
                centers = updated
            scores = -np.log(class_probabilities(x, centers))
        else:
            # K-means assignments are nearest centers; this soft display is a distance score.
            scores = -class_probabilities(x, centers) / .012
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p = p.reshape(maps['l'].shape + (-1,))
    intensity = np.einsum('yxc,yx->c', p, maps['l']) / p.sum(axis=(0, 1)).clip(1e-6)
    p = p[..., np.argsort(intensity)]
    return p.argmax(axis=-1), p.astype(np.float32)


def split_components(mask, score, minimum=9, split=True):
    """Use distance peaks only to test separation of merged dark candidates."""
    groups = []
    for _, part, area, _ in components(mask, minimum):
        if not split or area < 430:
            groups.append(part)
            continue
        clearance = cv2.distanceTransform(part.astype(np.uint8), cv2.DIST_L2, 5)
        peaks = (clearance >= cv2.dilate(clearance, ell(5))) & (clearance >= 3.2)
        markers = np.zeros(mask.shape, np.int32)
        for idx, (_, peak, _, _) in enumerate(components(peaks), 1):
            markers[peak] = idx
        if markers.max() < 2:
            groups.append(part)
            continue
        split_map = segmentation.watershed(.75 * (1 - score) - .04 * clearance, markers, mask=part)
        groups.extend(piece for idx in range(1, markers.max() + 1)
                      if (piece := split_map == idx).sum() >= minimum)
    return groups


def marker_image(parts, background):
    markers = np.zeros(background.shape, np.int32)
    markers[background] = 1
    for idx, part in enumerate(parts, 2):
        inner = shrink(part)
        if not inner.any():
            inner = part
        markers[inner] = idx
    return markers


def reject_union(parts, shape):
    return np.logical_or.reduce(parts) if parts else np.zeros(shape, bool)


def choose_anchor(parts, score, radius=32):
    ranked = []
    for idx, part in enumerate(parts):
        points = np.argwhere(part)
        center = points.mean(axis=0)
        nearest = RADIAL[part].min()
        if nearest > radius:
            continue
        mass = np.log1p(part.sum())
        stain = np.quantile(score[part], .76)
        on_frame = (part & FRAME).sum() / max(part.sum(), 1)
        rank = .54 * np.linalg.norm(center - CENTER) + .46 * nearest - 4.0 * mass - 13 * stain + 36 * on_frame
        ranked.append((rank, idx))
    return min(ranked)[1] if ranked else None


def compatible_groups(parts, anchor, score, edge, tissue, current_cell, max_gap=13):
    """Evaluate continuity, stain and membership in a provisional region."""
    primary = parts[anchor]
    centers = [np.argwhere(part).mean(axis=0) for part in parts]
    stain = np.quantile(score[primary], .82)
    accept, ambiguous, rival = [anchor], [], []
    for i, part in enumerate(parts):
        if i == anchor:
            continue
        gap, path = interval(primary, part)
        midpoint = np.linalg.norm((centers[i] + centers[anchor]) / 2 - CENTER)
        contrast = float(np.quantile(edge[path], .8)) if path[0].size else .55
        paper = float((~tissue[path]).mean()) if path[0].size else 0.
        similar = np.quantile(score[part], .82) >= stain - .18
        inside = (part & current_cell).sum() / part.sum()
        if (gap <= max_gap and midpoint <= 19 and similar and inside >= .68
                and contrast <= .56 and paper <= .22 and not (part & FRAME).any()):
            accept.append(i)
        elif (gap <= max_gap + 4 and midpoint < 23 and similar) or inside >= .42:
            ambiguous.append(i)
        else:
            rival.append(i)
    return accept, ambiguous, rival


def v4_parcels(vectors, method='SLIC', n=440):
    if method == 'Felzenszwalb':
        parts = segmentation.felzenszwalb(vectors, scale=45, sigma=.65, min_size=18,
                                          channel_axis=-1)
    else:
        parts = segmentation.slic(vectors, n_segments=n, compactness=.13, convert2lab=False,
                                   start_label=0, max_num_iter=8)
    return np.unique(parts, return_inverse=True)[1].reshape(parts.shape).astype(np.int32)


def v4_means(parcels, vectors):
    area = np.bincount(parcels.ravel())
    return np.column_stack([np.bincount(parcels.ravel(), weights=q.ravel()) / area
                            for q in np.moveaxis(vectors, -1, 0)])


def v4_contacts(parcels, edge):
    u = np.r_[parcels[:, :-1].ravel(), parcels[:-1].ravel()]
    v = np.r_[parcels[:, 1:].ravel(), parcels[1:].ravel()]
    transition = np.r_[np.maximum(edge[:, :-1], edge[:, 1:]).ravel(),
                       np.maximum(edge[:-1], edge[1:]).ravel()]
    different = u != v
    raw = np.sort(np.column_stack((u[different], v[different])), axis=1)
    pairs, inv = np.unique(raw, axis=0, return_inverse=True)
    count = np.bincount(inv)
    strength = np.bincount(inv, weights=transition[different]) / count
    return pairs.T, strength, count


def random_walker_regions(parcels, vectors, edge, markers, barrier_weight=5.2):
    """Solve absorbing-node hitting probabilities on an adjacency graph."""
    means = v4_means(parcels, vectors)
    (a, b), transition, contact = v4_contacts(parcels, edge)
    weight = np.sqrt(contact) * np.exp(-barrier_weight * transition - 4 * np.linalg.norm(means[a] - means[b], axis=1))
    network = sparse.csr_matrix((np.r_[weight, weight], (np.r_[a, b], np.r_[b, a])),
                                shape=(len(means), len(means)))
    lap = sparse.diags(np.asarray(network.sum(axis=1)).ravel()) - network
    votes = np.zeros((len(means), int(markers.max()) + 1), dtype=np.int32)
    np.add.at(votes, (parcels[markers > 0], markers[markers > 0]), 1)
    fixed = votes.sum(axis=1) > 0
    states = votes.argmax(axis=1)
    ids = np.unique(states[fixed])
    if ids.size == 0:
        return np.zeros(parcels.shape, np.int32), np.zeros(parcels.shape + (1,), np.float32)
    probability = (states[:, None] == ids).astype(np.float64)
    if (~fixed).any():
        reduced = lap.tocsr()[~fixed, :][:, ~fixed]
        forcing = network[~fixed, :][:, fixed] @ probability[fixed]
        solution = spsolve(reduced.tocsc(), forcing)
        probability[~fixed] = np.asarray(solution).reshape(np.count_nonzero(~fixed), len(ids))
    p = np.zeros((len(means), int(markers.max()) + 1), np.float32)
    p[:, ids] = probability.clip(0, 1)
    return ids[probability.argmax(axis=1)][parcels], p[parcels]


def attached_clean(mask, chrom, allowed, holes=9):
    mask = attached(mask & allowed, chrom)
    return small_holes(mask, allowed, holes)


def image_features(rgb):
    """Compute stain, absorption, luminance and gradient evidence at native size."""
    floating = rgb.astype(np.float64) / 255
    od = -np.log(np.clip((rgb.astype(float) + 1) / 256, 1 / 256, 1))
    hed = color.rgb2hed(floating)
    lab = color.rgb2lab(floating)
    hsv = color.rgb2hsv(floating)
    gray = color.rgb2gray(floating)
    tissue = (od.mean(axis=2) > .09) & ((hsv[..., 1] > .07) | (gray < .82))
    tissue = historical_objects(tissue, max_size=19)
    tissue = historical_holes(tissue, max_size=19)
    smoothed_lab = np.stack([ndi.gaussian_filter(lab[..., c], .8) for c in range(3)], axis=-1)
    grad = np.sqrt(sum(filters.scharr(smoothed_lab[..., c] / s) ** 2
                       for c, s in enumerate((100, 128, 128))))
    gradient = robust_scale(grad, 5, 97)
    return dict(od=od, hed=hed, lab=lab, hsv=hsv, gray=gray, tissue=tissue,
                gradient=gradient, H=robust_scale(hed[..., 0]),
                blue_od=robust_scale(od[..., 2]), dark=robust_scale(od.mean(axis=2)))


def chromatin_response(features, config):
    """Construct a stain response using the explicitly chosen tumor configuration."""
    h, blue, dark = features['H'], features['blue_od'], features['dark']
    maps = {'hed': h, 'blue_dark': .65 * blue + .35 * dark,
            'he_blend': .55 * h + .30 * blue + .15 * dark,
            'od_blue': .75 * blue + .25 * dark,
            'he_blue': .70 * h + .30 * blue}
    score = maps[config['map']]
    if config.get('clahe', False):
        enhanced = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(4, 4)).apply(
            np.uint8(np.clip(score, 0, 1) * 255)) / 255
        score = .75 * score + .25 * enhanced
    return ndi.gaussian_filter(score, config['sigma'])


def chromatin_candidates(features, score, config):
    """Hysteresis retains weak chromatin only when connected to a strong seed."""
    values = score[features['tissue']]
    if values.size < 10 or np.ptp(values) < 1e-6:
        return np.zeros(score.shape, np.int32), dict(high=np.nan, low=np.nan)
    if config['threshold'] == 'multiotsu':
        try:
            high = filters.threshold_multiotsu(values, classes=3)[-1]
        except ValueError:
            high = filters.threshold_otsu(values)
    else:
        high = filters.threshold_otsu(values) + config['offset']
    high = float(np.clip(high, config['high_min'], config['high_max']))
    low = max(.15, high - config['hysteresis_width'])
    weak = (score >= low) & features['tissue']
    if config['threshold'] == 'local':
        local = filters.threshold_local(score, block_size=25, method='gaussian', offset=-.03)
        weak &= score >= local
    strong = (score >= high) & weak
    raw = ndi.binary_propagation(strong, mask=weak)
    raw = historical_objects(raw, max_size=config['min_area'] - 1)
    raw = historical_holes(raw, max_size=4)
    labels = measure.label(raw, connectivity=2)
    # Split only unusually large fused objects, not every disconnected mitotic group.
    next_label = int(labels.max()) + 1
    for prop in measure.regionprops(labels):
        if prop.area <= config['split_area']:
            continue
        component = labels == prop.label
        distance = ndi.distance_transform_edt(component)
        peaks = feature.peak_local_max(distance, min_distance=7, labels=component,
                                       exclude_border=False)
        if len(peaks) < 2:
            continue
        seeds = np.zeros_like(labels)
        for k, position in enumerate(peaks, 1):
            seeds[tuple(position)] = k
        split = segmentation.watershed(-distance, seeds, mask=component)
        labels[component] = 0
        for k in range(1, len(peaks) + 1):
            piece = split == k
            if piece.sum() >= config['min_area']:
                labels[piece] = next_label
                next_label += 1
    return labels, dict(high=high, low=low)


def associate_components(labels, score, features, config):
    """Rank central candidates and associate nearby, dense, stain-supported fragments.

    No central pixel is required. Association is directly to the anchor (no chains).
    The returned group is a hypothesis, not an annotated mitotic identity.
    """
    center = (np.array(labels.shape) - 1) / 2
    properties = measure.regionprops(labels, intensity_image=score)
    records = []
    for prop in properties:
        coords = prop.coords
        centroid_distance = float(np.linalg.norm(np.array(prop.centroid) - center))
        nearest_distance = float(np.linalg.norm(coords - center, axis=1).min())
        border = bool(np.any((coords == 0) | (coords == np.array(labels.shape) - 1)))
        small_penalty = 14 * max(0, 30 - prop.area) / 30
        rank = (.55 * nearest_distance + .45 * centroid_distance
                - 10 * prop.intensity_mean - 2 * min(prop.area / 100, 1)
                + 12 * border + small_penalty)
        records.append(dict(label=prop.label, area=int(prop.area),
                            centroid_y=prop.centroid[0], centroid_x=prop.centroid[1],
                            central_distance=centroid_distance, nearest_distance=nearest_distance,
                            mean_score=prop.intensity_mean, solidity=prop.solidity,
                            border=border, rank=rank))
    table = pd.DataFrame(records)
    selected = np.zeros(labels.shape, bool)
    if table.empty:
        return selected, table, None
    eligible = table[table['nearest_distance'] <= config['anchor_radius']]
    if eligible.empty:
        table['selected'] = False
        return selected, table, None
    anchor = eligible.sort_values(['rank', 'label']).iloc[0]
    anchor_mask = labels == int(anchor['label'])
    distance_from_anchor = ndi.distance_transform_edt(~anchor_mask)
    chosen = [int(anchor['label'])]
    for record in records:
        if record['label'] in chosen:
            continue
        candidate = labels == record['label']
        gap = float(distance_from_anchor[candidate].min())
        midpoint = np.array([record['centroid_y'], record['centroid_x']])
        anchor_center = np.array([anchor['centroid_y'], anchor['centroid_x']])
        line = np.linspace(midpoint, anchor_center, max(2, int(np.linalg.norm(midpoint-anchor_center))))
        line = np.rint(line).astype(int)
        background_between = np.mean(~features['tissue'][line[:, 0], line[:, 1]])
        if (gap <= config['group_gap'] and record['central_distance'] <= config['group_radius']
                and record['mean_score'] >= max(.48, .82 * anchor['mean_score'])
                and record['area'] <= max(180, 2.5 * anchor['area'])
                and not record['border'] and background_between < .25):
            chosen.append(record['label'])
    selected = np.isin(labels, chosen)
    table['selected'] = table['label'].isin(chosen)
    return selected, table, int(anchor['label'])


def estimate_cell(features, score, labels, chromatin, config):
    """Estimate a cell using multichannel edges, tissue and competing nuclear markers.

    Dilations initialize markers only. The final boundary comes from watershed,
    competing geodesic paths or an edge contour; it is not a dilated chromatin mask.
    """
    if not chromatin.any():
        return np.zeros_like(chromatin), np.zeros_like(labels), features['gradient']
    tissue = features['tissue']
    others = (labels > 0) & ~chromatin
    nearby = morphology.dilation(chromatin, morphology.disk(3), mode='ignore') & tissue & ~others
    ring = nearby & ~chromatin
    reference = np.median(features['lab'][ring if ring.sum() >= 8 else chromatin], axis=0)
    color_distance = np.linalg.norm((features['lab'] - reference) / np.array([35, 25, 25]), axis=2)
    color_distance = np.clip(color_distance / 3, 0, 1)
    topography = (features['gradient'] + config['color_weight'] * color_distance
                  + .25 * (~tissue))
    background = morphology.erosion(others, morphology.disk(1), mode='ignore') | ~tissue
    background[0, :] = background[-1, :] = True
    background[:, 0] = background[:, -1] = True
    foreground = morphology.dilation(chromatin, morphology.disk(config['seed_expand']), mode='ignore')
    foreground &= tissue & ~others
    foreground |= chromatin
    background &= ~foreground
    markers = np.zeros_like(labels, np.int32)
    markers[background] = 1
    markers[foreground] = 2
    if config['cell_method'] == 'geodesic':
        # Costs have no circular/elliptical constraint or maximum radius cutoff.
        cost = 1 + config['edge_weight'] * topography
        foreground_points = np.argwhere(foreground)[::3]
        background_points = np.argwhere(background)[::3]
        d_foreground, _ = MCP_Geometric(cost, fully_connected=True).find_costs(foreground_points)
        d_background, _ = MCP_Geometric(cost, fully_connected=True).find_costs(background_points)
        cell = (d_foreground < config['geodesic_bias'] * d_background) & tissue
        # Markers stay fixed in the competing-path formulation.
        cell[background] = False
        cell[foreground] = True
    else:
        basins = segmentation.watershed(topography, markers, mask=np.ones_like(tissue),
                                         compactness=config['compactness'])
        cell = (basins == 2) & tissue
        if config['cell_method'] == 'active_contour':
            edge_stop = 1 / (1 + config['edge_weight'] * topography ** 2)
            cell = segmentation.morphological_geodesic_active_contour(
                edge_stop, num_iter=config['contour_iterations'], init_level_set=cell,
                smoothing=1, threshold=.45, balloon=0)
            cell = cell.astype(bool) & tissue & ~others
    # Retain every basin intersecting a selected chromatin group, not just the largest.
    components = measure.label(cell, connectivity=2)
    intersecting = np.unique(components[chromatin])
    cell = np.isin(components, intersecting[intersecting > 0])
    cell = historical_holes(cell, max_size=11) & tissue
    return cell, markers, topography


def segment_image(rgb, config):
    """Run the complete deterministic pipeline, returning auditable intermediate maps."""
    features = image_features(rgb)
    score = chromatin_response(features, config)
    labels, thresholds = chromatin_candidates(features, score, config)
    chromatin, components, anchor = associate_components(labels, score, features, config)
    cell, markers, topography = estimate_cell(features, score, labels, chromatin, config)
    return dict(features=features, score=score, labels=labels, thresholds=thresholds,
                chromatin=chromatin, cell=cell, components=components, anchor=anchor,
                markers=markers, topography=topography)


def preprocess(rgb, cfg):
    """Prepare stains, contrast and boundaries in the original 128 × 128 grid."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lab = np.stack((lab[..., 0] / 100, lab[..., 1] / 100, lab[..., 2] / 100), axis=-1)
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hed = color.rgb2hed(rgb)
    od = -np.log((rgb.astype(np.float32) + 1) / 256)
    h, e, blue = [robust_scale(item).astype(np.float32)
                  for item in (hed[..., 0], hed[..., 1], od[..., 2])]
    paper = (od.mean(axis=2) < .13) & (hsv[..., 1] < .20)
    black = (rgb.max(axis=2) <= 20)
    tissue = ~((regions(paper, 12)[0] > 0) | black)
    h_weight, blue_weight = cfg['stain_mix']
    absorption = h_weight * h + blue_weight * blue + (1 - h_weight - blue_weight) * robust_scale(od.mean(axis=2))
    absorption = cv2.GaussianBlur(absorption.astype(np.float32), (0, 0), cfg['sigma'])
    # Difference from the local background, measured in the stain-response map.
    local_background = cv2.GaussianBlur(absorption, (0, 0), cfg['context_sigma'])
    contrast = np.maximum(0, absorption - local_background)
    response = np.clip(absorption + cfg['local_boost'] * contrast, 0, 1)

    # Keep Lab chroma and E separate from the luminance/H transition that dominates nuclei.
    smooth_lab = cv2.bilateralFilter(lab.astype(np.float32), 5, .09, 2)
    e_smooth = cv2.GaussianBlur(e, (0, 0), 1.2)
    h_smooth = cv2.GaussianBlur(h, (0, 0), 1.0)
    planes = [smooth_lab[..., 0], smooth_lab[..., 1], smooth_lab[..., 2], e_smooth, h_smooth]
    gradients = []
    for plane in planes:
        gradients.append(cv2.magnitude(cv2.Scharr(plane, cv2.CV_32F, 1, 0),
                                       cv2.Scharr(plane, cv2.CV_32F, 0, 1)))
    light, a, b, eosin, hematoxylin = gradients
    chromatic = robust_scale(np.hypot(a, b))
    eosin_edge = robust_scale(eosin)
    structure = robust_scale(.6 * light + .4 * hematoxylin)
    weights = cfg['edge_mix']
    edge = (weights[0] * chromatic + weights[1] * eosin_edge + weights[2] * structure).astype(np.float32)
    # An additional view of thin luminance transitions, used only for inspection.
    canny = cv2.Canny(u8(smooth_lab[..., 0]), 28, 68) > 0
    return dict(rgb=rgb, lab=smooth_lab, h=h, e=e_smooth, score=response, tissue=tissue,
                chromatic=chromatic, eosin_edge=eosin_edge, structure=structure, edge=edge,
                luminance_edges=canny, stain_e_raw=hed[..., 1], stain_h_raw=hed[..., 0])


def propose_components(features, cfg):
    """Grow high-response seeds inside a lower, locally contrasted tissue mask."""
    score = features['score']
    tissue = features['tissue']
    if tissue.sum() < 10:
        return np.zeros(score.shape, np.int32), np.nan
    # The central window estimates thresholds; it never crops the candidate mask.
    local_values = score[30:98, 30:98][tissue[30:98, 30:98]]
    if local_values.size < 10:
        local_values = score[tissue]
    high, _ = cv2.threshold(u8(local_values).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high = float(np.clip(high / 255 + cfg['offset'], *cfg['high_range']))
    weak = (score >= high - cfg['hysteresis']) & tissue
    strong = (score >= high) & weak
    weak_labels, _, _ = regions(weak, cfg['fragment_area'])
    supported = np.unique(weak_labels[strong])
    candidate = np.isin(weak_labels, supported[supported > 0])

    # Split large merged regions by their strong cores only if there are at least two.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    next_label = int(labels.max()) + 1
    for idx in range(1, len(stats)):
        if stats[idx, cv2.CC_STAT_AREA] < cfg['split_area']:
            continue
        component = labels == idx
        core_level = max(high, float(np.percentile(score[component], 75)))
        cores, _, _ = regions(component & (score >= core_level), 14)
        if np.unique(cores[cores > 0]).size < 2:
            continue
        pieces = segmentation.watershed(-score, cores, mask=component, watershed_line=True)
        labels[component] = 0
        for piece in np.unique(pieces[pieces > 0]):
            fragment = pieces == piece
            if fragment.sum() >= cfg['fragment_area']:
                labels[fragment] = next_label
                next_label += 1
    return labels, high


def associate_v2(labels, features, high, cfg, use_bridges=True):
    """Choose a central anchor and evaluate each separate fragment with local evidence."""
    center = (np.array(labels.shape) - 1) / 2
    score = features['score']
    records = []
    for number in np.unique(labels[labels > 0]):
        mask = labels == number
        pixels = np.argwhere(mask)
        centroid = pixels.mean(axis=0)
        distance = np.linalg.norm(centroid - center)
        nearest = np.linalg.norm(pixels - center, axis=1).min()
        peak = np.percentile(score[mask], 85)
        edge_touch = bool((pixels == 0).any() or (pixels == np.array(labels.shape)-1).any())
        penalty = .42 * max(0, 25 - len(pixels)) + 16 * edge_touch
        rank = .38 * nearest + .62 * distance - 13 * peak + penalty
        records.append(dict(label=int(number), area=len(pixels), centroid=centroid, distance=distance,
                            nearest=nearest, peak=peak, mean=float(score[mask].mean()), rank=rank, border=edge_touch))
    eligible = [record for record in records if record['nearest'] <= cfg['anchor_radius']]
    if not eligible:
        return np.zeros_like(labels, bool), pd.DataFrame(), np.zeros_like(labels, bool)
    anchor = min(eligible, key=lambda record: (record['rank'], record['label']))
    anchor_mask = labels == anchor['label']
    distances = distance_to(anchor_mask)
    selected = [anchor['label']]
    competition = []
    uncertain = np.zeros_like(anchor_mask)
    for record in sorted(records, key=lambda item: item['label']):
        label = record['label']
        if label == anchor['label']:
            continue
        region = labels == label
        gap = float(distances[region].min())
        if gap > cfg['gap'] + 4 or record['distance'] > cfg['group_radius'] + 4:
            continue
        endpoint = np.argwhere(region & (distances <= gap + .02))[0]
        points = np.argwhere(anchor_mask)
        start = points[np.argmin(np.sum((points - endpoint) ** 2, axis=1))]
        n_samples = max(3, int(np.linalg.norm(endpoint - start)) + 1)
        yy, xx = np.rint(np.linspace(start, endpoint, n_samples)).astype(int).T
        between = (labels[yy, xx] != label) & (labels[yy, xx] != anchor['label'])
        if not between.any():
            between[:] = True
        paper = float((~features['tissue'][yy[between], xx[between]]).mean())
        halo = float(np.quantile(features['edge'][yy[between], xx[between]], .8))
        between_score = float(np.median(score[yy[between], xx[between]]))
        crossed_other = bool(np.any((labels[yy, xx] > 0) & (labels[yy, xx] != label) &
                                    (labels[yy, xx] != anchor['label'])))
        # Pair midpoint allows the center to lie between two intense groups.
        pair_center = float(np.linalg.norm((record['centroid'] + anchor['centroid']) / 2 - center))
        stain_similar = record['peak'] >= max(high - .04, anchor['peak'] - cfg['stain_drop'])
        size_ok = record['area'] <= max(130, 2.1 * anchor['area'])
        geometric = gap <= cfg['gap'] and record['distance'] <= cfg['group_radius']
        support = (between_score >= high - cfg['bridge_drop'] or pair_center <= cfg['pair_center'])
        admissible = ((geometric or (gap <= cfg['gap'] + 3 and pair_center <= cfg['pair_center']))
                      and support and halo <= cfg['barrier'] and not crossed_other)
        if use_bridges:
            accept = admissible and stain_similar and size_ok and not record['border'] and paper <= .2
        else:
            accept = geometric and stain_similar and size_ok and not record['border'] and paper <= .2
        if accept:
            selected.append(label)
        elif geometric and stain_similar and size_ok and not record['border']:
            uncertain |= region
        competition.append(dict(label=label, gap_px=round(gap, 1), midpoint_px=round(pair_center, 1),
                                stain_p85=round(record['peak'], 2), bridge_intensity=round(between_score, 2),
                                barrier=round(halo, 2), paper=round(paper, 2), crosses_candidate=crossed_other,
                                accepted=accept))
    return np.isin(labels, selected), pd.DataFrame(competition), uncertain


def build_markers(features, chromatin, labels, uncertain, cfg, width_delta=0):
    """Construct competing seeds from tissue, candidate nuclei and perichromatin."""
    rgb, tissue, lab = features['rgb'], features['tissue'], features['lab']
    candidate_neighbors = (labels > 0) & ~chromatin & ~uncertain
    background = erode(candidate_neighbors, 1) | ~tissue
    background[[0, -1], :] = True
    background[:, [0, -1]] = True
    foreground = erode(chromatin, 1)
    # Preserve a seed for each selected group, including very thin ones.
    chrom_ids, _, _ = regions(chromatin)
    for label in np.unique(chrom_ids[chrom_ids > 0]):
        component = chrom_ids == label
        if not (foreground & component).any():
            foreground |= component
    if not chromatin.any():
        return foreground, background, candidate_neighbors, foreground, np.zeros(chromatin.shape)
    distance = distance_to(chromatin)
    width = cfg['ring'] + width_delta
    ring = (distance >= 2) & (distance <= width) & tissue & ~(labels > 0)
    similarity = np.zeros_like(distance)
    if ring.sum() >= 10:
        samples = np.dstack((lab, features['e']))[ring]
        reference = np.median(samples, axis=0)
        scale = np.maximum(np.quantile(np.abs(samples - reference), .7, axis=0), [.045, .027, .028, .09])
        delta = (np.dstack((lab, features['e'])) - reference) / scale
        similarity = np.exp(-.5 * np.mean(delta ** 2, axis=2))
    # Only perichromatin with observable color similarity and modest transition response
    # contributes seeds. Ring width initializes the methods and never clips the result.
    candidate_ring = ring & (similarity >= cfg['ring_similarity']) & (features['edge'] <= cfg['seed_edge'])
    training_ring = cv2.morphologyEx(candidate_ring.astype(np.uint8), cv2.MORPH_OPEN, disk(1)).astype(bool)
    background &= ~foreground
    return foreground, background, candidate_neighbors, training_ring, similarity


def cell_raw(features, chromatin, labels, uncertain, cfg, method, width_delta=0):
    """Compete foreground with adjacent nuclei and background using appearance."""
    fg, bg, neighbors, ring, similarity = build_markers(features, chromatin, labels, uncertain, cfg, width_delta)
    markers = dict(fg=fg, bg=bg, training=ring, similarity=similarity, neighbors=neighbors)
    if not chromatin.any():
        return chromatin.copy(), markers
    lab, tissue, edge = features['lab'], features['tissue'], features['edge']
    # Retain L and both chromatic axes, with E blended into the b-axis proxy.
    channels = np.ascontiguousarray(np.dstack((u8(lab[..., 0]), u8(.5 + 1.4 * lab[..., 1]),
                                              u8(.65 * (.5 + 1.4 * lab[..., 2]) + .35 * features['e']))))
    if method == 'grabcut':
        gc = np.full(chromatin.shape, cv2.GC_PR_BGD, np.uint8)
        distance = distance_to(chromatin)
        possible = (distance <= 3 * (cfg['ring'] + width_delta)) & (similarity > cfg['ring_similarity'] * .7)
        gc[(possible & tissue & ~neighbors) | chromatin] = cv2.GC_PR_FGD
        gc[bg] = cv2.GC_BGD
        gc[fg | (ring & (similarity >= .8))] = cv2.GC_FGD
        cv2.setRNGSeed(SEED)
        if min(np.count_nonzero((gc & 1) == 1), np.count_nonzero((gc & 1) == 0)) < 5:
            return np.zeros_like(chromatin), {**markers, 'degenerate': True}
        cv2.grabCut(channels, gc, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), cfg['grabcut_iterations'], cv2.GC_INIT_WITH_MASK)
        mask = (gc & 1).astype(bool)
    elif method == 'watershed':
        # Encode transitions as features. OpenCV watershed compares neighboring feature
        # values, so this is a watershed on a stack, not a scalar-elevation flood.
        transition_stack = np.ascontiguousarray(channels.copy())
        transition_stack[..., 0] = u8(.57 * lab[..., 0] + .43 * edge)
        marker_ids = np.zeros(chromatin.shape, np.int32)
        # Place distinct competing labels; they may meet at a watershed boundary.
        bg_labels, _, _ = regions(bg)
        marker_ids[bg] = bg_labels[bg] + 1
        marker_ids[fg | ring] = 1
        # OpenCV assigns -1 to the outer frame, so place frame seeds one pixel inside.
        for line in (1, -2):
            row = marker_ids[line, :]
            row[row == 0] = 2
            column = marker_ids[:, line]
            column[column == 0] = 2
        cv2.watershed(transition_stack, marker_ids)
        mask = marker_ids == 1
    else:
        raise ValueError(f'Método celular desconocido: {method}')
    return mask & tissue, markers


def refine_cell(raw, chromatin, features, markers):
    """Remove unanchored regions and fill only small internal holes with tissue evidence."""
    ids, _, _ = regions(raw)
    # A one-pixel adjacency allowance preserves a case where the watershed line is nearby.
    source = np.unique(ids[dilate(chromatin, 1)])
    tracked = np.isin(ids, source[source > 0])
    holes, stats, _ = regions(~tracked)
    small_ids = np.flatnonzero((stats[:, cv2.CC_STAT_AREA] <= 12) & (np.arange(len(stats)) > 0))
    filler = np.isin(holes, small_ids) & features['tissue'] & ~markers['bg']
    return tracked | filler


def mapas_region_tincion(rgb, work=None, use_texture=False):
    """Extract separate evidence for dark material, tissue, and regional boundaries."""
    smoothed = cv2.bilateralFilter(rgb, 5, 17, 3) if work is None else work
    lab = cv2.cvtColor(smoothed.astype(np.float32) / 255., cv2.COLOR_RGB2Lab)
    od = -np.log((smoothed.astype(np.float32) + 1.) / 256.)
    raw = -np.log((rgb.astype(np.float32) + 1.) / 256.)
    hed = color.rgb2hed(smoothed.astype(np.float32) / 255.)
    h = scale(hed[..., 0], 5, 98)
    e = scale(hed[..., 1], 5, 98)
    light, a, b = np.moveaxis(lab, -1, 0)
    light /= 100.
    a /= 75.
    b /= 75.
    opacity = scale(od.mean(axis=-1), 3, 98)
    r, g, blue = np.moveaxis(smoothed.astype(np.float32), -1, 0)
    ratio = scale((100 * blue / (1 + r + g)) * (256 / (1 + r + g + blue)))
    white = (raw.mean(axis=-1) < .17) & (raw.std(axis=-1) < .09)
    # Very bright, weakly stained pixels provide an exterior cue. Their boundary is still image-derived.
    tissue = ~white
    local = scale(np.maximum(0., cv2.GaussianBlur(light, (0, 0), 5.0) - light), 1, 98)
    compact = scale(np.maximum.reduce([
        np.maximum(0, -sigma**2 * cv2.Laplacian(cv2.GaussianBlur(ratio, (0, 0), sigma), cv2.CV_32F))
        for sigma in (1., 2.2, 4.)]), 3, 98)
    chromatic = scale(np.hypot(grad(a, .7), grad(b, .7)), 4, 99)
    broad = scale(np.sqrt(grad(a, 1.9)**2 + grad(b, 1.9)**2 + .35 * grad(e, 1.9)**2), 4, 99)
    edge = .24 * scale(grad(light, .7), 4, 99) + .34 * chromatic + .24 * broad + .18 * scale(grad(e, 1.), 4, 99)
    outer = .22 * edge + .45 * broad + .33 * scale(grad(e, 2.4), 5, 99)
    stack = np.dstack((light, a, b, .56 * e, .18 * h))
    texture, texture_edge = np.zeros((*light.shape, 1), np.float32), np.zeros(light.shape, np.float32)
    if use_texture:
        bands = []
        for wavelength in (4.7, 8.2):
            for theta in np.arange(4) * np.pi / 4:
                response = []
                for phase in (0., np.pi / 2):
                    kernel = cv2.getGaborKernel((19, 19), .43 * wavelength, theta, wavelength, .58, phase, ktype=cv2.CV_32F)
                    kernel -= kernel.mean()
                    kernel /= np.linalg.norm(kernel)
                    response.append(cv2.filter2D(light, cv2.CV_32F, kernel))
                bands.append(scale(cv2.GaussianBlur(np.hypot(*response), (0, 0), 1.65)))
        texture = np.stack(bands, axis=-1)
        texture_edge = scale(np.sqrt(sum(grad(channel, 1.1)**2 for channel in bands)), 5, 99)
    return dict(rgb=rgb, filtered=smoothed, l=light, a=a, b=b, h=h, e=e, ratio=ratio, od=opacity,
                local=local, log=compact, tissue=tissue, edge=edge, broad=broad, outer=outer, vector=stack,
                texture=texture, texture_edge=texture_edge)


def candidatos_oscuros(score, maps, quantile=.83, band=.12, split_area=740, step=0):
    """Find and, where supported by saddles, separate dark components across the field."""
    center = (RADIAL < 39) & maps['tissue']
    sample = score[center] if center.sum() >= 25 else score[maps['tissue']]
    if sample.size < 25:
        blank = np.zeros(score.shape, bool)
        return [], blank, blank
    high_cut = np.quantile(sample, quantile) + .014 * step
    high = (score >= high_cut) & maps['tissue']
    admissible = (score >= high_cut - band) & maps['tissue']
    candidate = attached(admissible, high)
    groups = []
    for _, part, area, _ in components(candidate, 10):
        if area < split_area:
            groups.append(part)
            continue
        clear = cv2.distanceTransform(part.astype(np.uint8), cv2.DIST_L2, 5)
        peaks = morphology.h_maxima(clear, 2.1).astype(bool) & (clear > 3.25) & part
        markers = np.zeros(part.shape, np.int32)
        for idx, (_, peak, _, _) in enumerate(components(peaks), 1):
            markers[peak] = idx
        if markers.max() < 2:
            groups.append(part)
            continue
        slope = (1 - score) - .055 * clear
        territories = segmentation.watershed(slope, markers, mask=part, watershed_line=True)
        groups.extend(piece for label in range(1, markers.max() + 1)
                      if (piece := territories == label).sum() >= 13)
    return groups, candidate, high


def foco_central(parts, signal, labels=None, domain_id=0, reach=36, light=None, offset_penalty=.47):
    ranked = []
    for idx, part in enumerate(parts):
        points = np.argwhere(part)
        center = points.mean(axis=0)
        near = RADIAL[part].min()
        if near > reach:
            continue
        stain = np.quantile(signal[part], .82)
        membership = (labels[part] == domain_id).mean() if labels is not None and domain_id else 0.
        body = np.quantile(1 - light[part], .75) if light is not None else 0.
        price = (.54 * near + offset_penalty * np.linalg.norm(center - CENTER)
                 - 23 * stain - 2.3 * np.log1p(len(points)) - 7.3 * membership - 12.5 * body)
        price += 38 * (part & FRAME).sum() / len(points)
        ranked.append((price, idx))
    return min(ranked)[1] if ranked else None


def pair_evidence(maps, score, first, second, terrain):
    gap, path = interval(first, second)
    centers = np.argwhere(first).mean(axis=0), np.argwhere(second).mean(axis=0)
    midpoint = np.linalg.norm((centers[0] + centers[1]) / 2 - CENTER)
    seam = expand(first, 1) & expand(second, 1)
    if path[0].size:
        barrier = float(np.quantile(terrain[path], .78))
        valley = float(np.quantile(score[path], .37))
        exterior = float((~maps['tissue'][path]).mean())
    elif seam.any():
        barrier = float(np.quantile(terrain[seam], .72))
        valley = float(np.quantile(score[seam], .37))
        exterior = float((~maps['tissue'][seam]).mean())
    else:
        return gap, midpoint, 1., 0., 1.
    return gap, midpoint, barrier, valley, exterior


def associate5(maps, signal, parts, primary, region, reach, edge_limit, minimum=.12, strict=False):
    indexed, unsure = [primary], []
    main = parts[primary]
    labels, terrain = region['labels'], region['edge']
    ids = np.bincount(labels[main])
    ids[0] = 0
    common = ids.argmax() if ids.any() else 0
    staining = float(np.quantile(signal[main], .78))
    base_color = np.median(maps['vector'][main], axis=0)
    for i, part in enumerate(parts):
        if i == primary or RADIAL[part].min() > 38:
            continue
        gap, midpoint, barrier, valley, exterior = pair_evidence(maps, signal, main, part, terrain)
        same_region = bool(common and (labels[part] == common).mean() >= .40)
        delta = np.linalg.norm(np.median(maps['vector'][part], axis=0) - base_color)
        stain = float(np.quantile(signal[part], .78))
        distributed_support = same_region or (barrier < edge_limit - .05 and valley > staining - .31)
        relative = part.sum() / main.sum()
        strong = (gap <= reach and midpoint <= 19 and barrier < edge_limit
                  and exterior <= .23 and stain >= staining - .16
                  and delta <= .31 and relative >= minimum and distributed_support
                  and not (part & FRAME).any())
        if strict:
            strong &= valley > staining - .24 and delta < .23
        if strong:
            indexed.append(i)
        elif (gap <= reach + 3 and midpoint <= 22 and stain >= staining - .15
              and exterior <= .32 and delta <= .30 and .10 <= relative <= 2.1):
            unsure.append(i)
    return indexed, unsure


def audited_result(cell, chromatin, neighbor=None, uncertain=None, **details):
    """Keep selected chromatin independent from the final cellular extent."""
    chromatin = chromatin.astype(bool, copy=True)
    return made(cell.astype(bool), chromatin, neighbor, uncertain,
                preconstraint=chromatin.copy(), excluded_by_cell=chromatin & ~cell, **details)


def border_support(cell, chromatin, edge):
    ring = (cell & ~shrink(cell)) & ~expand(chromatin, 2)
    return float(cv2.dilate(edge, ell(1))[ring].mean()) if ring.any() else 0.


def landscape_markers(surface, tissue, depth, kind):
    accessible = tissue & ~FRAME
    if not accessible.any():
        return np.zeros(tissue.shape, dtype=np.int32)
    if kind == 'H-maxima de espacio libre':
        barrier = (surface >= np.quantile(surface[accessible], .72)) | ~accessible
        clearance = distance(barrier)
        extrema = morphology.h_maxima(clearance, depth) & (clearance >= 2.4) & accessible
        value = -clearance
    else:
        value = np.where(accessible, surface, 1.4).astype(np.float32)
        extrema = morphology.h_minima(value, depth).astype(bool) & accessible
    markers = np.zeros(tissue.shape, np.int32)
    for idx, (_, component, _, _) in enumerate(components(extrema), 1):
        xy = np.argwhere(component)[np.argmin(value[component])]
        markers[tuple(xy)] = idx
    for _, island, _, _ in components(accessible, 4):
        if not markers[island].any():
            xy = np.argwhere(island)[np.argmin(value[island])]
            markers[tuple(xy)] = int(markers.max()) + 1
    return markers


def upper_contacts(labels, edge):
    u = np.r_[labels[:, :-1].ravel(), labels[:-1].ravel()]
    v = np.r_[labels[:, 1:].ravel(), labels[1:].ravel()]
    gradient = np.r_[np.maximum(edge[:, :-1], edge[:, 1:]).ravel(),
                     np.maximum(edge[:-1], edge[1:]).ravel()]
    contact = u != v
    n = int(labels.max()) + 1
    codes = np.minimum(u[contact], v[contact]).astype(np.int64) * n + np.maximum(u[contact], v[contact])
    ranking = np.lexsort((gradient[contact], codes))
    unique, start, sizes = np.unique(codes[ranking], return_index=True, return_counts=True)
    high = gradient[contact][ranking][start + np.floor(.64 * (sizes - 1)).astype(int)]
    return (unique // n, unique % n), high, sizes


def merge_hierarchy(fine, coarse, vector, edge, limit, texture_weight=0.):
    """Contract contacts with weak transitions and consistent appearance."""
    count = int(fine.max()) + 1
    parent = np.arange(count)
    area = np.bincount(fine.ravel(), minlength=count).astype(float)
    channel_sum = [np.bincount(fine.ravel(), channel.ravel(), minlength=count)
                   for channel in np.moveaxis(vector, -1, 0)]
    sums = np.column_stack(channel_sum)
    signatures = np.zeros(count, np.int32)
    packed = fine.astype(np.int64) * (int(coarse.max()) + 1) + coarse
    codes, votes = np.unique(packed, return_counts=True)
    div = int(coarse.max()) + 1
    best = np.zeros(count, np.int32)
    for code, vote in zip(codes, votes):
        region, wider = divmod(code, div)
        if vote > best[region]:
            signatures[region], best[region] = wider, vote

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    (first, second), transitions, width = upper_contacts(fine, edge)
    base_cost = transitions + .035 / np.sqrt(width)
    for index in np.argsort(base_cost, kind='stable'):
        i, j = root(first[index]), root(second[index])
        if i == j or not i or not j or width[index] < 3:
            continue
        delta = (sums[i] / area[i]) - (sums[j] / area[j])
        chromatic = np.linalg.norm(delta[:5])
        texture_change = np.linalg.norm(delta[5:]) if len(delta) > 5 else 0.
        mismatch = .14 if signatures[i] != signatures[j] else 0.
        price = base_cost[index] + .33 * chromatic + texture_weight * texture_change + mismatch
        if price > limit:
            continue
        if area[i] < area[j]:
            i, j = j, i
        parent[j] = i
        sums[i] += sums[j]
        area[i] += area[j]
    lookup = np.array([root(i) for i in range(count)])
    return np.unique(lookup[fine], return_inverse=True)[1].reshape(fine.shape).astype(np.int32)


def jerarquia_regional(maps, approach='H-minima', use_texture=False, limit=.40, step=0, use_outer=False):
    tissue = maps['tissue']
    transition = maps['outer'] if use_outer else maps['edge']
    vector = maps['vector']
    if use_texture:
        transition = .66 * transition + .34 * maps['texture_edge']
        vector = np.dstack((vector, .33 * maps['texture']))
    relief = (.56 * transition + .44 * cv2.GaussianBlur(transition, (0, 0), 1.25)).astype(np.float32)
    coarse_relief = cv2.GaussianBlur(relief, (0, 0), 2.05)
    if approach == 'H-maxima de espacio libre':
        fine_markers = landscape_markers(relief, tissue, 1.05, approach)
        coarse_markers = landscape_markers(coarse_relief, tissue, 2.05, approach)
    else:
        fine_markers = landscape_markers(relief, tissue, .038, 'H-minima')
        coarse_markers = landscape_markers(coarse_relief, tissue, .080, 'H-minima')
    coarse = segmentation.watershed(coarse_relief, coarse_markers, mask=tissue & ~FRAME)
    if approach == 'Felzenszwalb':
        fine = v4_parcels(vector, 'Felzenszwalb') + 1
        fine[~tissue | FRAME] = 0
    else:
        fine = segmentation.watershed(relief, fine_markers, mask=tissue & ~FRAME)
    merged = merge_hierarchy(fine, coarse, vector, transition, limit - .037 + .018 * step, .09 if use_texture else 0.)
    return dict(fine=fine, labels=merged, coarse=coarse, seeds=fine_markers, edge=transition,
                vector=vector, surface=relief)


def anchor_region(labels, score, tissue):
    """Score whole-field regions with a spatial prior and color evidence."""
    candidates = []
    for idx in np.unique(labels):
        if idx == 0:
            continue
        mask = (labels == idx) & tissue
        if not mask.any() or RADIAL[mask].min() > 31:
            continue
        locality = np.exp(-.5 * (RADIAL[mask] / 19.)**2)
        strength = .21 + .79 * score[mask]**2
        rank = float(np.sum(locality * strength) / max(mask.sum(), 1)**.56)
        candidates.append((rank, int(idx)))
    return max(candidates)[1] if candidates else 0


def body_from_regions(region, maps, target, rival, limit=.44, max_hops=2):
    """Admit adjacent tissue according to image contact evidence and regional appearance."""
    labels, vector, edge = region['labels'], region['vector'], region['edge']
    if not target.any():
        return target.copy()
    area = np.bincount(labels.ravel())
    targets = np.bincount(labels[target], minlength=len(area)) / np.maximum(area, 1)
    opponents = np.bincount(labels[rival], minlength=len(area)) / np.maximum(area, 1)
    main = np.flatnonzero(targets > .20)
    if not len(main):
        main = np.unique(labels[target])
    parts = {int(i): [] for i in range(len(area))}
    (left, right), strength, widths = v4_contacts(labels, edge)
    means = v4_means(labels, vector)
    for i, j, barrier, width in zip(left, right, strength, widths):
        if not i or not j or width < 3:
            continue
        color_change = np.linalg.norm(means[i] - means[j])
        price = .74 * barrier + .26 * color_change
        parts[i].append((j, price))
        parts[j].append((i, price))
    accepted = set(main[main > 0].tolist())
    frontier = {i: 0. for i in accepted}
    for _ in range(max_hops):
        following = {}
        for i, prev in frontier.items():
            for j, price in parts[i]:
                if j in accepted or opponents[j] > .18 or price > limit:
                    continue
                value = prev + price
                if value <= limit + .22:
                    following[j] = min(value, following.get(j, np.inf))
        accepted.update(following)
        frontier = following
    return np.isin(labels, list(accepted)) & maps['tissue']


def boundary_rivals(labels, nuclei, excepted):
    """Label territories bearing unrelated dark cores."""
    candidates = [part for i, part in enumerate(nuclei) if i not in excepted]
    stain_rivals = reject_union(candidates, labels.shape)
    area = np.bincount(labels.ravel())
    burden = np.bincount(labels[stain_rivals], minlength=len(area))
    competing = (burden >= 11) & (burden >= .20 * area)
    if len(competing):
        competing[0] = False
    return competing[labels], stain_rivals


def territorios_por_coste(maps, field, parts, selected, priority, step=0):
    """Assign tissue through multichannel travel costs from foreground, rival and exterior markers."""
    exterior = expand(FRAME, 2) | shrink(~maps['tissue'])
    marker = np.zeros(exterior.shape, np.int32)
    marker[exterior] = 1
    anchor_mask = reject_union([parts[i] for i in selected], exterior.shape)
    for idx, group in enumerate(parts, 2):
        core = shrink(group, 1)
        marker[core if core.any() else group] = 2 if idx - 2 in selected else idx + 2
    labels = field['labels']
    # Stable regime membership contributes weak additional anchors only inside the primary territory.
    neighbors = (marker > 2) & ~anchor_mask
    support = priority & maps['tissue'] & ~expand(neighbors, 2)
    if support.any():
        local = support & expand(anchor_mask, 8) & (field['edge'] < .31)
        marker[local & (marker == 0)] = 2
    cost = 1. + (9.2 + .65 * step) * maps['outer']**1.75
    cost[~maps['tissue']] *= 3.2
    traces = MCP_Geometric(cost)
    arrivals, parent = traces.find_costs([tuple(point) for point in np.argwhere(marker > 0)])
    steps = np.asarray(traces.offsets, dtype=np.intp)
    differences = steps[:, 0] * marker.shape[1] + steps[:, 1]
    owner = marker.copy().ravel()
    parents = parent.ravel()
    visit = np.argsort(arrivals.ravel(), kind='stable')
    for pixel in visit[parents[visit] >= 0]:
        owner[pixel] = owner[pixel - differences[parents[pixel]]]
    return owner.reshape(marker.shape), marker, arrivals.astype(np.float32)


def field_means(vectors, mask, prototype_count):
    values = vectors[mask]
    if len(values) < 8:
        return None
    if prototype_count == 1:
        return [np.median(values, axis=0)]
    threshold = np.median(values[:, 0])
    pieces = [values[values[:, 0] <= threshold], values[values[:, 0] > threshold]]
    return [np.median(piece, axis=0) for piece in pieces if len(piece) >= 4]


def contour5(maps, initial, domain, fixed, edge, channels, iterations=13, prototype_count=2):
    """Discrete narrow-band level set with vector region fitting and barrier weights."""
    phase = (initial & domain).copy()
    if not phase.any():
        return phase
    for _ in range(iterations):
        inside = field_means(channels, phase, prototype_count)
        outside = field_means(channels, expand(phase, 8) & ~phase, prototype_count)
        if not inside or not outside:
            break
        fit_in = np.minimum.reduce([np.mean((channels - center)**2, axis=-1) for center in inside])
        fit_out = np.minimum.reduce([np.mean((channels - center)**2, axis=-1) for center in outside])
        phi = distance(~phase) - distance(phase)
        band = np.abs(phi) <= 1.55
        regularity = cv2.GaussianBlur(phase.astype(np.float32), (0, 0), 1.15) - .5
        drive = ((fit_out - fit_in) / (fit_in + fit_out + .007)) + .30 * regularity
        barrier = .073 + .49 * edge**2
        added = band & ~phase & (drive >= barrier) & domain
        lost = band & phase & (drive < -barrier) & ~fixed
        updated = ((phase | added) & ~lost) & domain
        if np.array_equal(updated, phase):
            break
        phase = updated
    return phase


def energia_orientada(light):
    """Oriented energy at two scales and the axial structure tensor."""
    bands = []
    for wavelength in (4., 8.):
        for angle in np.arange(4) * np.pi / 4:
            response = []
            for phase in (0., np.pi / 2):
                kernel = cv2.getGaborKernel((19, 19), .47 * wavelength, angle,
                                            wavelength, .62, phase, ktype=cv2.CV_32F)
                kernel -= kernel.mean()
                response.append(cv2.filter2D(light, -1, kernel / np.linalg.norm(kernel)))
            bands.append(scale(cv2.GaussianBlur(np.hypot(*response), (0, 0), 1.7)))
    smooth = cv2.GaussianBlur(light, (0, 0), 1.1)
    gx, gy = cv2.Scharr(smooth, -1, 1, 0), cv2.Scharr(smooth, -1, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), 3.) for v in (gx*gx, gy*gy, gx*gy))
    power = jxx + jyy + .01
    contrast = scale(cv2.GaussianBlur(light**2, (0, 0), 3.) -
                     cv2.GaussianBlur(light, (0, 0), 3.)**2)
    return np.dstack((*bands, (jxx-jyy)/power, 2*jxy/power, contrast))


def mapas_apariencia(rgb, with_fibers=False):
    """Color, staining, local texture, and image gradients remain separately inspectable."""
    maps = mapas_region_tincion(rgb)
    local_mean = cv2.GaussianBlur(maps['l'], (0, 0), 3.)
    local_variance = cv2.GaussianBlur(maps['l']**2, (0, 0), 3.) - local_mean**2
    maps['std'] = scale(np.sqrt(np.maximum(0, local_variance)))
    maps['lab6'] = np.dstack((scale(maps['l']), scale(maps['a']), scale(maps['b'])))
    maps['he6'] = np.dstack((maps['h'], maps['e'], maps['od']))
    maps['br6'] = np.dstack((maps['ratio'], maps['l'], maps['e']))
    maps['edge6'] = .42 * maps['outer'] + .38 * maps['edge'] + .20 * maps['broad']
    if with_fibers:
        maps['fibers'] = energia_orientada(maps['l'])
        amplitudes = maps['fibers'][..., :8].mean(axis=-1)
        directional = np.sqrt(grad(maps['fibers'][..., 8], 1.5)**2 +
                              grad(maps['fibers'][..., 9], 1.5)**2)
        maps['fiber_edge'] = .52 * maps['edge6'] + .28 * scale(grad(amplitudes, 1.8)) + .20 * scale(directional)
    return maps


def fixed_mask6(signal, method, tissue):
    src = u8(cv2.GaussianBlur(signal.astype(np.float32), (0, 0), .65))
    if method == 'Adaptativo':
        level = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 23, -7)
        return (level > 0) & tissue
    if method == 'Multi-Otsu':
        values = src[tissue] if tissue.sum() >= 30 else src.ravel()
        if len(np.unique(values)) >= 3:
            level = filters.threshold_multiotsu(values, classes=3)[-1]
            return (src > level) & tissue
    method_flag = cv2.THRESH_TRIANGLE if method == 'Triángulo' else cv2.THRESH_OTSU
    values = src[tissue] if tissue.sum() >= 30 else src.ravel()
    level, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | method_flag)
    if method == 'Histéresis':
        return attached((src > max(0, level - 15)) & tissue, (src > level + 11) & tissue)
    return (src > level) & tissue


def parcel_graph(vectors, edges, segments=350, tightness=.09):
    labels = segmentation.slic(vectors, n_segments=segments, compactness=tightness,
                               start_label=0, convert2lab=False, enforce_connectivity=True)
    count = int(labels.max()) + 1
    nodes = labels.ravel()
    area = np.bincount(nodes, minlength=count)
    coords = np.column_stack([np.bincount(nodes, weights=c.ravel()) / area for c in (YY, XX)])
    ends = np.concatenate([
        np.column_stack((labels[:, :-1].ravel(), labels[:, 1:].ravel())),
        np.column_stack((labels[:-1, :].ravel(), labels[1:, :].ravel()))])
    values = np.concatenate([(.5 * (edges[:, :-1] + edges[:, 1:])).ravel(),
                             (.5 * (edges[:-1, :] + edges[1:, :])).ravel()])
    valid = ends[:, 0] != ends[:, 1]
    ends = np.sort(ends[valid], axis=1)
    encoded, indices, length = np.unique(ends[:, 0] * count + ends[:, 1],
                                         return_inverse=True, return_counts=True)
    first, second = encoded // count, encoded % count
    barrier = np.bincount(indices, weights=values[valid]) / length
    return dict(labels=labels, area=area, coords=coords, first=first, second=second,
                barrier=barrier, length=length, edge=edges)


def parcel_means6(graph, channels):
    ids, area = graph['labels'].ravel(), graph['area']
    return np.column_stack([np.bincount(ids, weights=channel.ravel()) / area
                            for channel in np.moveaxis(channels, -1, 0)])


def coverage6(graph, mask):
    return np.bincount(graph['labels'].ravel(), weights=mask.ravel()) / graph['area']


def fusion_adyacente(graph, vectors, limit=.82):
    """Agglomerate neighbors with recomputed Ward color/texture loss and contact barriers."""
    means = parcel_means6(graph, vectors)
    area = graph['area'].astype(float).copy()
    member = np.arange(len(means))
    normalizer = np.maximum(vectors.reshape(-1, vectors.shape[-1]).var(axis=0), .018)
    for _ in range(len(means) - 1):
        first, second = member[graph['first']], member[graph['second']]
        valid = first != second
        if not valid.any():
            break
        first, second = first[valid], second[valid]
        contacts = graph['length'][valid]
        low, high = np.minimum(first, second), np.maximum(first, second)
        codes, inverse = np.unique(low * len(means) + high, return_inverse=True)
        a, b = codes // len(means), codes % len(means)
        barrier = np.bincount(inverse, weights=contacts * graph['barrier'][valid]) / np.bincount(inverse, weights=contacts)
        ward = np.sqrt(np.mean((means[a] - means[b])**2 / normalizer, axis=1))
        balance = 2 * np.sqrt(area[a] * area[b]) / (area[a] + area[b])
        cost = ward * (.75 + .25 * balance) + 1.65 * barrier
        best = cost.argmin()
        if cost[best] >= limit:
            break
        left, right = a[best], b[best]
        combined = area[left] + area[right]
        means[left] = (area[left] * means[left] + area[right] * means[right]) / combined
        area[left] = combined
        member[member == right] = left
    return member


def ajustar_mezcla(vectors, mask, order):
    data = vectors[mask].astype(np.float64)
    if len(data) > 900:
        data = data[np.linspace(0, len(data) - 1, 900, dtype=int)]
    if len(data) < 30 or np.unique(data, axis=0).shape[0] < order:
        order = 1
    if not data.size:
        return None
    if order == 1:
        return data.mean(axis=0)[None], np.maximum(data.var(axis=0), .004)[None], np.ones(1)
    cv2.setRNGSeed(SEED)
    model = cv2.ml.EM_create()
    model.setClustersNumber(order)
    model.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 45, .001))
    ok, _, _, _ = model.trainEM(np.ascontiguousarray(data))
    if not ok:
        return ajustar_mezcla(vectors, mask, 1)
    variance = np.maximum(np.array([np.diag(cov) for cov in model.getCovs()]), .004)
    return model.getMeans(), variance, model.getWeights().ravel()


def log_densidad_mezcla(query, model):
    centers, variances, weights = model
    loss = ((query[:, None] - centers)**2 / variances + np.log(2*np.pi*variances)).sum(axis=-1)
    return logsumexp(-.5 * loss + np.log(np.maximum(weights, 1e-8)), axis=1)


def appearance6(graph, vectors, maps, reference, anchor, rivals, order=2, local_exterior=False):
    """Fit target cytoplasm, its dark material, each rival, and regional exterior separately."""
    ids = graph['labels']
    means = parcel_means6(graph, vectors)
    samples = distance(anchor)
    foreground = (reference & maps['tissue'] & (samples > 1.) &
                  ~expand(rivals, 2) & (maps['edge6'] < .43) & (samples <= 26))
    if foreground.sum() < 38:
        foreground = (samples > 1.) & (samples < 10.) & maps['tissue'] & ~rivals
    source = ajustar_mezcla(vectors, foreground, order)
    dark = ajustar_mezcla(vectors, anchor, 1)
    if source is None:
        source = dark
    distance_from_body = distance(reference | anchor)
    exterior = ((distance_from_body >= 8) & (distance_from_body <= 27) & ~expand(rivals, 2)
                & ~expand(anchor, 8))
    if local_exterior:
        exterior &= (samples < 39) & ((maps['edge6'] < .39) | ~maps['tissue'])
    sectors = (np.floor((np.arctan2(YY - YY[anchor].mean(), XX - XX[anchor].mean()) + np.pi)
                        * (2 / np.pi))).astype(int)
    rivals_models = []
    for _, group, _, _ in components(rivals, 22):
        support = expand(group, 5) & ~expand(anchor, 4) & maps['tissue'] & ~foreground
        if support.sum() >= 24:
            rivals_models.append(ajustar_mezcla(vectors, support, min(order, 2)))
    outside_models = [ajustar_mezcla(vectors, exterior & (sectors == sector), order)
                      for sector in range(4) if np.count_nonzero(exterior & (sectors == sector)) >= 24]
    models = rivals_models + outside_models
    if not models:
        models = [ajustar_mezcla(vectors, (samples >= 22) & ~anchor, 1) or source]
    # Evaluate per-pixel likelihood ratios, then aggregate the evidence in each parcel.
    pixel_query = vectors.reshape(-1, vectors.shape[-1])
    cell_log = log_densidad_mezcla(pixel_query, source)
    dark_log = log_densidad_mezcla(pixel_query, dark)
    # A dark submodel contributes locally. It cannot absorb a remote matching nucleus.
    target_log = np.logaddexp(cell_log, dark_log - 1.4 - (samples.ravel()/9)**2)
    opposition = np.max([log_densidad_mezcla(pixel_query, model) for model in models], axis=0)
    odds_px = (target_log - opposition).reshape(ids.shape) / np.sqrt(means.shape[1])
    radial_prior = np.maximum(0, samples - 11) / 15
    log_odds = parcel_means6(graph, np.clip(odds_px - radial_prior, -7, 7)[..., None])[:, 0]
    hard_target = coverage6(graph, anchor) > .48
    if not hard_target.any():
        hard_target[coverage6(graph, anchor).argmax()] = True
    hard_other = (coverage6(graph, rivals) > .32) | (coverage6(graph, expand(FRAME, 1)) > .44)
    hard_other |= coverage6(graph, ~maps['tissue']) > .83
    hard_other &= ~hard_target
    return dict(odds=log_odds, probability=expit(log_odds)[ids], source=foreground,
                exterior=exterior, target=hard_target, other=hard_other,
                mixtures=1 + len(models), components=len(source[2]), rivals=len(rivals_models))


def corte_espacial(graph, model, smoothing=1., offset=0.):
    n = len(model['odds'])
    nodes = np.arange(n)
    odds = model['odds'] - offset
    area_factor = np.sqrt(graph['area'] / np.median(graph['area']))
    source_link = np.rint(700 * area_factor * np.logaddexp(0, odds)).astype(np.int64)
    sink_link = np.rint(700 * area_factor * np.logaddexp(0, -odds)).astype(np.int64)
    boundary_link = np.rint(900 * smoothing * np.sqrt(graph['length'] / 5)
                            * np.exp(-4.5 * graph['barrier'])).astype(np.int64)
    hard = 1 + int(source_link.sum() + sink_link.sum() + boundary_link.sum()*2)
    source_link[model['target']] = hard
    sink_link[model['other']] = hard
    a, b = graph['first'], graph['second']
    capacity = sparse.csr_matrix(
        (np.r_[source_link, sink_link, boundary_link, boundary_link],
         (np.r_[np.full(n, n), nodes, a, b], np.r_[nodes, np.full(n, n+1), b, a])),
        shape=(n+2, n+2), dtype=np.int64)
    residual = capacity - maximum_flow(capacity, n, n+1).flow
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    return np.isin(graph['labels'], reachable[reachable < n])


def dark_signals6(maps):
    return {
        'LAB': .66 * (1 - maps['lab6'][..., 0]) + .22 * (1 - maps['lab6'][..., 2]) + .12 * maps['local'],
        'H&E': .57 * maps['h'] + .29 * maps['od'] + .14 * maps['local'],
        'Blue Ratio': .74 * maps['ratio'] + .16 * maps['od'] + .10 * maps['local'],
    }


def all_dark6(maps, previous):
    signals = dark_signals6(maps)
    vote = np.zeros_like(maps['l'], dtype=np.uint8)
    for score in signals.values():
        vote += fixed_mask6(score, 'Histéresis', maps['tissue'])
    old_candidate = previous.get('candidate', previous['chromatin'])
    dark = (vote >= 2) | ((vote >= 1) & old_candidate)
    return vote, signals, dark


def foco_compacto(previous):
    old = previous['chromatin']
    pieces = components(old)
    if not pieces:
        return old.copy()
    best = max(pieces, key=lambda item: item[2] / (1 + .035 * np.linalg.norm(item[3] - CENTER))**2)
    return best[1]


def candidate_groups6(dark, signal):
    return split_components(dark, signal, minimum=13, split=True)


def informacion_cromatinica(previous):
    chrom = previous['chromatin']
    return chrom, previous.get('candidate', chrom), previous['omitted'], previous['excluded'], ()


def formar_resultado_regional(raw, detection, original, graph=None, model=None, marker=None, cleanup=True):
    chrom, candidate, unsure, rivals, group_evidence = detection
    if cleanup and raw.any():
        contact = raw & expand(chrom, 1)
        if contact.any():
            raw = attached(raw, contact)
        else:
            raw = raw.copy()
        allowed = ~rivals
        cell = small_holes(raw, allowed, 12)
    else:
        cell = raw
    return audited_result(cell, chrom, rivals, unsure, raw_cell=raw,
                          candidate=candidate | chrom, group_evidence=group_evidence,
                          graph=graph, model=model, marker=marker,
                          old_anchor=original['chromatin'])


def grafo_multicanal(ctx, vectors, edge=None, n=350, compactness=.10):
    edge = ctx['maps']['edge6'] if edge is None else edge
    return parcel_graph(vectors, edge, n + ctx['perturb'] * 29, compactness)


def modelos_de_apariencia(ctx, graph, vectors, detection=None, order=2, outer=False):
    detection = ctx['base'] if detection is None else detection
    chrom, _, _, rivals, _ = detection
    anchor = ctx['anchor'] if not chrom.any() else chrom
    return appearance6(graph, vectors, ctx['maps'], ctx['previous']['cell'], anchor, rivals, order, outer)


def corte_en_banda(ctx, detection, vectors, exterior_width=13, edge=None):
    """Refine a provisional boundary with local cytoplasm and exterior mixtures."""
    maps, previous = ctx['maps'], ctx['previous']
    body, chrom, rivals = previous['cell'], detection[0], detection[3]
    graph = grafo_multicanal(ctx, vectors, edge, 560, .14)
    if not body.any():
        model = modelos_de_apariencia(ctx, graph, vectors, detection)
        return corte_espacial(graph, model), graph, model
    from_dark = distance(chrom if chrom.any() else ctx['anchor'])
    internal = distance(~body)
    outside = distance(body)
    src_mask = body & (internal >= 2) & (from_dark > 2) & ~expand(rivals, 1)
    # Very compact I5 bodies contain few representative cytoplasmic pixels.
    if src_mask.sum() < 110:
        src_mask = ((src_mask | ((from_dark >= 2) & (from_dark <= 12) & (outside <= 10)))
                    & (maps['edge6'] < .37) & maps['tissue'] & ~expand(rivals, 2))
    inside_model = ajustar_mezcla(vectors, src_mask, 3)
    if inside_model is None:
        inside_model = ajustar_mezcla(vectors, body, 1)
    shell = ((outside >= exterior_width*.68) & (outside <= exterior_width + 12)
             & ~expand(rivals, 1) & (from_dark > 12))
    outside_model = ajustar_mezcla(vectors, shell, 3)
    if outside_model is None:
        outside_model = ajustar_mezcla(vectors, ~body, 1)
    query = vectors.reshape(-1, vectors.shape[-1])
    raw_odds = (log_densidad_mezcla(query, inside_model)-log_densidad_mezcla(query, outside_model)).reshape(body.shape)
    per_pixel = .61*np.clip(raw_odds / np.sqrt(vectors.shape[-1]), -3.5, 3.5)
    per_pixel += .42*np.minimum(internal, 7) / 7 - .38*outside/exterior_width
    odds = parcel_means6(graph, per_pixel[...,None])[:,0]
    core = (body & (internal >= 7) & ~rivals) | (shrink(chrom) & (from_dark < 1))
    fg = (coverage6(graph, core) > .47) | (coverage6(graph, chrom) > .66)
    if not fg.any():
        fg[coverage6(graph, chrom | body).argmax()] = True
    bg = coverage6(graph, (outside > exterior_width) | (rivals & ~body) | ~maps['tissue'] |
                   expand(FRAME, 1)) > .61
    bg &= ~fg
    model = dict(odds=odds, probability=expit(odds)[graph['labels']], source=src_mask,
                 exterior=shell, target=fg, other=bg, mixtures=2,
                 components=len(inside_model[2]), rivals=0)
    return corte_espacial(graph, model, smoothing=1.55, offset=-.04), graph, model


def region_means(labels, channels):
    count = np.bincount(labels.ravel())
    return np.column_stack([np.bincount(labels.ravel(), weights=value.ravel(), minlength=len(count))
                            for value in channels]) / np.maximum(count[:, None], 1)


def region_contacts(labels, edge):
    first = np.r_[labels[:, :-1].ravel(), labels[:-1].ravel()]
    second = np.r_[labels[:, 1:].ravel(), labels[1:].ravel()]
    strength = np.r_[np.maximum(edge[:, :-1], edge[:, 1:]).ravel(),
                     np.maximum(edge[:-1], edge[1:]).ravel()]
    distinct = first != second
    pairs = np.sort(np.column_stack((first[distinct], second[distinct])), axis=1)
    unique, reverse = np.unique(pairs, axis=0, return_inverse=True)
    count = np.bincount(reverse)
    return unique.T, np.bincount(reverse, weights=strength[distinct]) / count, count
