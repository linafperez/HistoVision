"""Diagramas explicativos derivados de las expresiones del código, sin ejecutarlas."""
import ast
import inspect
import textwrap


# Verbs describe the operation; the adjacent expression retains its exact operands.
VERBS = {
    'bilateralFilter': 'Suavizar conservando transiciones de color',
    'cvtColor': 'Transformar el espacio de color', 'threshold': 'Separar por umbral',
    'threshold_multiotsu': 'Calcular cortes entre varias clases',
    'threshold_otsu': 'Maximizar la separación entre dos clases',
    'watershed': 'Propagar marcadores sobre el relieve',
    'grabCut': 'Ajustar apariencia y minimizar el corte',
    'distance': 'Medir distancia al conjunto', 'distance_to': 'Medir distancia al conjunto',
    'expand': 'Dilatar la región', 'shrink': 'Erosionar la región',
    'attached': 'Conservar componentes conectados al foco',
    'attached_clean': 'Restringir, conectar y rellenar la región',
    'small_holes': 'Rellenar huecos admisibles',
    'historical_holes': 'Rellenar huecos según su área',
    'components': 'Etiquetar y filtrar componentes por área',
    'regions': 'Separar componentes conectados',
    'grad': 'Calcular la magnitud del gradiente',
    'Scharr': 'Derivar la intensidad en el eje indicado',
    'magnitude': 'Combinar las derivadas por norma euclídea',
    'GaussianBlur': 'Suavizar con la escala gaussiana indicada',
    'gaussian_filter': 'Suavizar la respuesta gaussiana',
    'percentile': 'Calcular el percentil de la respuesta',
    'quantile': 'Calcular el cuantil de la respuesta',
    'mapas_tincion_multiescala': 'Combinar respuestas cromáticas multiescala',
    'base_maps': 'Extraer luminancia, cromaticidad y fronteras',
    'mapas_region_tincion': 'Separar H&E, densidad óptica y textura',
    'mapas_apariencia': 'Construir los vectores de apariencia',
    'energia_orientada': 'Integrar respuestas de textura orientada',
    'ajustar_mezcla': 'Ajustar distribuciones a las muestras de la máscara',
    'appearance6': 'Contrastar modelos celulares, rivales y exteriores',
    'modelos_de_apariencia': 'Ajustar la apariencia regional con los focos detectados',
    'log_densidad_mezcla': 'Evaluar la log-verosimilitud de la mezcla',
    'corte_en_banda': 'Delimitar una banda con modelos interiores y exteriores',
    'corte_espacial': 'Resolver el corte con restricciones espaciales',
    'maximum_flow': 'Resolver el flujo máximo de capacidades',
    'breadth_first_order': 'Recuperar nodos alcanzables en el grafo residual',
    'spsolve': 'Resolver el sistema lineal regional',
    'parcel_graph': 'Construir parcelas, contactos y barreras',
    'grafo_multicanal': 'Construir la adyacencia multicanal',
    'fusion_adyacente': 'Fusionar contactos según apariencia y barrera',
    'parcel_means6': 'Agregar la evidencia por parcela',
    'coverage6': 'Medir cobertura de la máscara en cada parcela',
    'select_body': 'Ordenar propuestas por consistencia y apoyo del borde',
    'select_chromatin': 'Exigir ganancia de evidencia cromatínica',
    'inspect_record': 'Medir cromatina, cuerpo y condiciones de advertencia',
    'body_evidence': 'Medir frontera, intrusión y cuerpo extranuclear',
    'chrom_evidence': 'Medir tinción, foco y delimitación cromatínica',
    'support': 'Valorar el soporte conjunto de la propuesta',
    'formar_resultado_regional': 'Integrar cuerpo, cromatina y exclusiones',
    'with_detection': 'Sustituir la detección conservando el registro regional',
    'full_chromatin': 'Consolidar cromatina conservada y omitida',
    'min': 'Seleccionar el mínimo', 'max': 'Seleccionar el máximo',
    'argmax': 'Localizar el máximo', 'sum': 'Agregar valores',
    'clip': 'Acotar la respuesta', 'mean': 'Calcular la media',
    'isin': 'Recuperar píxeles de las etiquetas seleccionadas',
    'unique': 'Identificar valores distintos', 'expit': 'Transformar log-odds en pertenencia',
    'breast_maps': 'Combinar fronteras cromáticas y eosina a dos escalas',
    'breast_initial': 'Competir por coste geodésico entre cromatina y exterior',
    'breast_evolution': 'Evolucionar el contorno con interior y cromatina fijados',
    'breast_finish': 'Conectar al foco y rellenar huecos dentro del tejido admisible',
    'run_breast': 'Estimar cuerpo por competencia geodésica y evolución restringida',
    'contorno_regional': 'Evolucionar Chan–Vese sobre la pertenencia, fijando exclusiones',
    'sarcoma_maps': 'Combinar color con energía Gabor en cuadratura',
    'sarcoma_seeds': 'Separar foco, rivales, fondo y vecindario citoplasmático',
    'sarcoma_partition': 'Elegir partición armónica Lab/Gabor o contorno geodésico',
    'harmonic_membership': 'Resolver pertenencia armónica entre parcelas',
    'sarcoma_finish': 'Restringir al tejido sin vecinos y conectar con la cromatina',
    'run_sarcoma': 'Integrar textura orientada, semillas y partición regional',
    'melanoma_maps': 'Combinar frontera de color y gradiente de hematoxilina',
    'melanoma_markers': 'Separar interior fijo, dominio, excluidos y grupos ambiguos',
    'melanoma_contour': 'Reconstruir o evolucionar el contorno dentro del dominio',
    'melanoma_finish': 'Contrastar apoyo del borde y condiciones de cambio local',
    'run_melanoma': 'Refinar el contorno con marcadores y control del borde',
    'lymph_maps': 'Combinar color, relieve y respuesta cromatínica del linfoma',
    'lymph_parcels': 'Crear cuencas con marcadores y fronteras multicanal',
    'lymph_associate': 'Asociar grupos por contacto, barrera y apoyo regional',
    'lymph_finish': 'Conectar el territorio asociado y conservar material ambiguo',
    'lymph_review_collapsed': 'Revisar colapso del cuerpo frente al material oscuro',
    'run_lymph': 'Construir cuencas, asociar grupos y revisar la región celular',
    'mast_groups': 'Revisar grupos oscuros respecto al dominio y núcleo fijo',
    'build_markers': 'Fijar primer plano, fondo y etiquetas probables',
    'marker_image': 'Etiquetar focos por separado y añadir semillas de fondo',
    'resolve_mast_band': 'Resolver la banda mediante el método y marcadores elegidos',
    'refine_cell': 'Conectar la propuesta al foco y limitar el relleno',
    'preparar_reconstruccion_mastocitoma': 'Preparar soporte fuerte y débil de Blue Ratio y H',
    'reconstruir_mastocitoma': 'Reconstruir añadidos conectados y separar material dudoso',
    'rescatar_mastocitoma': 'Evaluar dominancia, tinción y frontera antes del rescate',
    'preparar_reconstruccion_mama': 'Contrastar clases próximas y exteriores con soporte geodésico',
    'reconstruir_mama': 'Reconstruir tejido desde cromatina con soporte fuerte y débil',
    'preparar_reconstruccion_melanoma': 'Obtener focos y marcadores de tinción según la alternativa',
    'reconstruccion_melanoma': 'Competir desde marcadores y reconstruir los focos admisibles',
    'preparar_focos_densos': 'Obtener grupos de respuesta oscura y relieve local',
    'seleccionar_foco_melanoma': 'Elegir el foco por dominancia, posición y barreras',
    'preparar_competencia_multicanal': 'Preparar clases de tinción y semillas competidoras',
    'territorios_multicanal': 'Resolver pertenencia territorial según canales y marcadores',
    'segmentar_competencia_multicanal': 'Reunir focos aceptados y excluir territorios rivales',
    'segmentar_territorios_de_tincion': 'Asociar focos y estimar tejido sobre la jerarquía regional',
    'territorios_por_coste': 'Competir entre focos mediante distancias de coste acumulado',
    'asociacion_pulmon_multicanal': 'Revisar contactos de focos en el grafo pulmonar',
    'reconstruccion_relieve': 'Recuperar grupos oscuros mediante el relieve de tinción',
    'segmentar_cuencas_multicanal': 'Seleccionar cuencas y asociar sus componentes cromatínicos',
    'preparar_cuencas_jerarquicas': 'Preparar jerarquía y contraste multicanal de cuencas',
    'segmentar_cuencas_jerarquicas': 'Asociar focos y delimitar cuerpo sobre cuencas fusionadas',
    'asociacion_por_contacto': 'Revisar unión de componentes por contacto territorial',
    'preparar_random_walker': 'Preparar focos, candidatos y marcadores enteros',
    'segmentar_random_walker': 'Propagar marcadores, elegir foco y evaluar vecinos',
    'preparar_territorios_neuro': 'Preparar jerarquía regional y respuesta oscura ponderada',
    'segmentar_territorios_neuro': 'Asociar focos y competir por territorios de tejido',
    'nucleo_por_intensidad': 'Revisar bordes cromatínicos por Otsu intracomponente',
    'nucleo_por_densidad': 'Reconstruir cromatina por densidad y contraste local',
    'associate5': 'Evaluar asociación por distancia, tinción y barrera intermedia',
    'associate_components': 'Asociar componentes por criterios cromáticos y espaciales',
    'associate_v2': 'Revisar foco, puentes y componentes vecinos',
    'contour5': 'Actualizar pertenencia por prototipos dentro del dominio permitido',
    'new_context': 'Preparar foco fijo, rivales y contexto regional',
    'legacy_first': 'Obtener la propuesta inicial de tinción y componentes',
    'legacy_second': 'Obtener la propuesta inicial con asociación y marcadores',
    'made': 'Reunir cuerpo, cromatina, excluidos y estados intermedios',
    'audited_result': 'Reunir máscaras independientes y material incierto',
    'morphological_chan_vese': 'Evolucionar regiones por contraste interior y exterior',
    'morphological_geodesic_active_contour': 'Evolucionar el contorno sobre la función de parada',
}


def _names(node, context=ast.Load):
    return sorted({part.id for part in ast.walk(node)
                   if isinstance(part, ast.Name) and isinstance(part.ctx, context)})


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ''


def _writes(statement):
    names = set(_names(statement, ast.Store))
    targets = (statement.targets if isinstance(statement, ast.Assign) else
               [statement.target] if isinstance(statement, (ast.AnnAssign, ast.AugAssign)) else [])
    for target in targets:
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        if isinstance(target, ast.Name):
            names.add(target.id)
    for part in ast.walk(statement):
        if (isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                and part.func.attr in ('append', 'extend', 'update', 'eliminate_zeros')):
            names.update(_names(part.func.value))
    return sorted(names)


def _caption(statement):
    value = getattr(statement, 'value', statement)
    calls = [_call_name(part.func) for part in ast.walk(value) if isinstance(part, ast.Call)]
    if isinstance(statement, ast.Return):
        return ('Devolver: '+VERBS[calls[0]].lower() if calls and calls[0] in VERBS
                else 'Devolver la salida de esta ruta')
    if isinstance(statement, ast.AugAssign):
        if isinstance(statement.op, ast.BitAnd):
            return 'Restringir la máscara por intersección'
        if isinstance(statement.op, ast.BitOr):
            return 'Añadir regiones por unión de máscaras'
        return 'Actualizar la respuesta acumulada'
    if isinstance(value, ast.BinOp):
        if isinstance(value.op, ast.BitAnd):
            return 'Intersectar máscaras y condiciones'
        if isinstance(value.op, ast.BitOr):
            return 'Unir máscaras candidatas'
        if isinstance(value.op, ast.Sub):
            return 'Calcular la diferencia de respuestas'
        if isinstance(value.op, ast.Div):
            return 'Normalizar la respuesta'
        return 'Combinar las respuestas ponderadas'
    if isinstance(value, ast.Compare):
        return 'Obtener la máscara que cumple el criterio'
    if isinstance(value, ast.IfExp):
        return 'Elegir la alternativa según la condición'
    if isinstance(value, (ast.ListComp, ast.DictComp, ast.GeneratorExp)):
        return 'Evaluar cada elemento y sus condiciones'
    if calls and calls[0] in VERBS:
        return VERBS[calls[0]]
    if isinstance(value, ast.Subscript):
        return 'Extraer los datos indicados por la máscara o índice'
    if isinstance(value, (ast.Dict, ast.Tuple, ast.List)) or 'dict' in calls:
        return 'Reunir las entradas en una estructura'
    if calls and calls[0] in ('append', 'extend'):
        return 'Incorporar la propuesta a las alternativas'
    if calls:
        words = calls[0].replace('_', ' ')
        return 'Obtener ' + words
    return 'Asignar los datos de trabajo'


class _Flow:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.producers = {}
        self.number = 0

    def node(self, title, expression='', kind='operation', reads=(), writes=()):
        ident = 'n' + str(self.number)
        self.number += 1
        self.nodes.append(dict(id=ident, titulo=title, expresion=expression, tipo=kind))
        for name in reads:
            for producer in self.producers.get(name, ()):
                if producer != ident:
                    self.edge(producer, ident, name, 'data')
        for name in writes:
            self.producers[name] = [ident]
        return ident

    def edge(self, source, target, label='', kind='sequence'):
        item = dict(desde=source, hasta=target, etiqueta=label, tipo=kind)
        if item not in self.edges:
            self.edges.append(item)

    def chain(self, previous, target):
        for source, label in previous:
            self.edge(source, target, label)

    def block(self, statements, previous, loop=None):
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(statement, ast.FunctionDef):
                ident=self.node('Definir la transformación local y sus restricciones',
                                ast.unparse(statement),'callback',_names(statement),[statement.name])
                self.chain(previous,ident)
                previous=[(ident,'')]
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue  # docstrings
            if isinstance(statement, (ast.Assign, ast.Return)) and isinstance(statement.value, ast.IfExp):
                import copy
                yes, no = copy.copy(statement), copy.copy(statement)
                yes.value, no.value = statement.value.body, statement.value.orelse
                statement = ast.If(test=statement.value.test, body=[yes], orelse=[no])
            if isinstance(statement, ast.If):
                ident = self.node('Evaluar la condición y bifurcar', ast.unparse(statement.test),
                                  'condition', _names(statement.test))
                self.chain(previous, ident)
                before = dict(self.producers)
                yes = self.block(statement.body, [(ident, 'Sí')], loop)
                yes_producers = dict(self.producers)
                self.producers = dict(before)
                no = self.block(statement.orelse, [(ident, 'No')], loop)
                for key, values in yes_producers.items():
                    self.producers[key] = sorted(set(values + self.producers.get(key, [])))
                previous = yes + no
            elif isinstance(statement, (ast.For, ast.While)):
                expression = (ast.unparse(statement.target) + ' ∈ ' + ast.unparse(statement.iter)
                              if isinstance(statement, ast.For) else ast.unparse(statement.test))
                condition = statement.iter if isinstance(statement, ast.For) else statement.test
                ident = self.node('Repetir mientras existan elementos o se cumpla el criterio',
                                  expression, 'loop', _names(condition),
                                  _names(statement.target, ast.Store) if isinstance(statement, ast.For) else [])
                self.chain(previous, ident)
                exit_node = self.node('Continuar al terminar el ciclo', '', 'join')
                exit_object = self.nodes.pop()
                end = self.block(statement.body, [(ident, 'Iterar')], (ident, exit_node))
                for source, label in end:
                    self.edge(source, ident, 'Repetir')
                self.edge(ident, exit_node, 'Fin')
                self.nodes.append(exit_object)
                previous = self.block(statement.orelse, [(exit_node, '')], loop)
            elif isinstance(statement, ast.Try):
                ident = self.node('Intentar el cálculo; usar rescate si falla', '', 'condition')
                self.chain(previous, ident)
                normal = self.block(statement.body, [(ident, 'Cálculo')], loop)
                normal = self.block(statement.orelse, normal, loop)
                alternatives = []
                for handler in statement.handlers:
                    alternatives += self.block(handler.body, [(ident, 'Excepción: ' +
                        (ast.unparse(handler.type) if handler.type else 'cualquiera'))], loop)
                previous = self.block(statement.finalbody, normal + alternatives, loop)
            elif isinstance(statement, (ast.Break, ast.Continue)):
                if loop:
                    self.chain(previous, loop[1] if isinstance(statement, ast.Break) else loop[0])
                previous = []
            else:
                ident = self.node(_caption(statement), ast.unparse(statement),
                    'output' if isinstance(statement, ast.Return) else 'operation',
                    _names(statement), _writes(statement))
                self.chain(previous, ident)
                previous = [] if isinstance(statement, (ast.Return, ast.Raise)) else [(ident, '')]
        return previous


# These native OpenCV operations have no inspectable Python body. Their distinct
# schemas document their public algorithm and use the effective recorded arguments.
NATIVE = {
 'bilateralFilter': [
    ('RGB de entrada', 'src'), ('Vecindarios y pesos espaciales', 'd; Gσs(‖p−q‖), σs = sigmaSpace'),
    ('Pesos de similitud cromática', 'Gσr(‖I(p)−I(q)‖), σr = sigmaColor'),
    ('Promedio ponderado y normalización', 'If(p) = Σq ws(p,q) wr(p,q) I(q) / Σq ws(p,q) wr(p,q)'),
    ('Imagen suavizada', 'dst')],
 'cvtColor': [('Imagen y conversión', 'src; code; dstCn'),
    ('Transformar canales según el código de color', 'dst = cv2.cvtColor(src, code, dstCn=dstCn)'),
    ('Canales transformados', 'dst')],
 'threshold': [('Canal de entrada', 'src'),
    ('Elegir el criterio indicado por type', 'type & THRESH_OTSU; type & THRESH_TRIANGLE; umbral fijo'),
    ('Calcular el corte de intensidad', 'ret = umbral automático o thresh'),
    ('Comparar cada píxel; aplicar inversión si corresponde', 'THRESH_BINARY: src > ret; THRESH_BINARY_INV: src ≤ ret'),
    ('Umbral obtenido y máscara', 'ret, dst; valor positivo = maxval')],
 'grabCut': [('Imagen de entrada', 'img'),
    ('Marcadores de primer plano y fondo', 'mask; rect'),
    ('Estado de los modelos de apariencia', 'bgdModel; fgdModel'),
    ('Inicializar según mode y fijar etiquetas duras', 'GC_BGD=0; GC_FGD=1; GC_PR_BGD=2; GC_PR_FGD=3'),
    ('Ajustar mezclas de apariencia y costes de asignación', 'Modelos de primer plano y fondo sobre img'),
    ('Minimizar energía con términos vecinos y actualizar modelos', 'E(L)=Σp Dp(Lp)+Σpq Vpq[Lp≠Lq]; iterCount iteraciones'),
    ('Marcadores y modelos actualizados', 'mask; bgdModel; fgdModel')],
 'watershed': [('Imagen de entrada', 'image'), ('Marcadores enteros de entrada', 'markers'),
    ('Propagar etiquetas por transiciones locales', 'Semillas: markers > 0; sin etiqueta: markers = 0'),
    ('Resolver contactos entre cuencas', 'Separadores: markers = −1'),
    ('Etiquetas de cuencas y separadores', 'markers, actualizado in situ')],
 'getGaborKernel': [('Escala, orientación y frecuencia', 'ksize; sigma; theta; lambd; gamma; psi; ktype'),
    ('Rotar las coordenadas', 'x′=x cos θ+y sin θ; y′=−x sin θ+y cos θ'),
    ('Multiplicar envolvente gaussiana y oscilación', 'g(x,y)=exp(−(x′²+γ²y′²)/(2σ²)) cos(2πx′/λ+ψ)'),
    ('Núcleo orientado', 'Matriz de tamaño ksize')],
 'filter2D': [('Canal de entrada', 'src'), ('Núcleo de entrada', 'kernel; anchor; ddepth; delta; borderType'),
    ('Correlacionar cada vecindario con el núcleo', 'dst(x,y)=Σu,v kernel(u,v) src(x+u−ax,y+v−ay)+delta'),
    ('Respuesta de textura', 'dst con profundidad ddepth')],
}


def describir_flujo(function, cv_name=None):
    """Return a reusable schema; never infer edges from display order."""
    name = cv_name or function.__name__
    flow = _Flow()
    if cv_name:
        if name not in NATIVE:
            raise ValueError('Operación nativa sin documentación: ' + name)
        input_count = 3 if name=='grabCut' else 2 if name in ('filter2D','watershed') else 1
        previous = []
        for index, (title, expression) in enumerate(NATIVE[name]):
            node = flow.node(title, expression, 'input' if index < input_count else
                             'output' if index == len(NATIVE[name])-1 else 'operation')
            flow.chain(previous, node)
            previous = [(node, '')]
        # Distinct input streams of the bilateral filter converge in its average.
        if name == 'bilateralFilter':
            flow.edges = []
            for first, last, label in ((0,1,'posición'),(0,2,'color'),(1,3,'peso espacial'),
                                       (2,3,'peso cromático'),(0,3,'intensidad'),(3,4,'')):
                flow.edge('n'+str(first), 'n'+str(last), label,
                          'sequence' if first == 3 else 'data')
        if input_count > 1:
            flow.edges=[]
            for index in range(input_count):
                flow.edge('n'+str(index),'n'+str(input_count),NATIVE[name][index][1],'data')
            for index in range(input_count,len(NATIVE[name])-1):
                flow.edge('n'+str(index),'n'+str(index+1))
        source = 'OpenCV · ' + name
    else:
        original = inspect.unwrap(function)
        source = textwrap.dedent(inspect.getsource(original))
        definition = ast.parse(source).body[0]
        positional = definition.args.posonlyargs + definition.args.args
        required = len(positional)-len(definition.args.defaults)
        args = [arg.arg for index,arg in enumerate(positional)
                if index < required or (isinstance(definition.args.defaults[index-required],ast.Constant)
                                        and definition.args.defaults[index-required].value is None)]
        args += [arg.arg for arg,default in zip(definition.args.kwonlyargs,definition.args.kw_defaults)
                 if default is None]
        if definition.args.vararg:
            args.append('*'+definition.args.vararg.arg)
        if definition.args.kwarg:
            args.append('**'+definition.args.kwarg.arg)
        for arg in args:
            flow.node('Entrada · '+arg.lstrip('*'),arg,'input',writes=[arg.lstrip('*')])
        flow.block(definition.body,[])
        source = original.__module__ + '.' + name
    return dict(fuente=source, nodos=flow.nodes, conexiones=flow.edges,
                alcance='Estructura del código: las bifurcaciones muestran rutas posibles; '
                        'los parámetros y las imágenes inferiores pertenecen a la llamada registrada.')
