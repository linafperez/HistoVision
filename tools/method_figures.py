"""Compone las láminas metodológicas; no importa ni ejecuta la segmentación.

Generación: python tools/method_figures.py (requiere matplotlib).
Las recetas resumen operaciones del código consolidado, no sus instrucciones Python.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'methods'
PROFILES = {}


def profile(key, title, blocks, explanation, formula=None, rows=None, edges=None, notes=()):
    rows = rows or [[i] for i in range(len(blocks))]
    edges = edges or [(a,b) for first,last in zip(rows,rows[1:]) for a in first for b in last]
    PROFILES[key] = dict(title=title, blocks=blocks, rows=rows, edges=edges,
                         explanation=explanation, formula=formula, notes=list(notes))


profile('bilateral','Filtrado bilateral',
 ['Imagen RGB','Filtrado bilateral','Imagen suavizada'],
 'El filtro bilateral pondera la proximidad espacial y la semejanza cromática de los píxeles. Suaviza la tinción y conserva las transiciones de color utilizadas en las etapas posteriores.',
 'bilateral',notes=[(1,'Proximidad espacial'),(1,'Semejanza cromática')])
profile('multiscale','Tinción a varias escalas',
 ['Imagen RGB','CIELAB y separación H&E','Blue Ratio y densidad óptica','Contraste local a varias escalas','Mapas de respuesta de tinción'],
 'Se combinan canales de color, deconvolución H&E y respuestas de contraste a distintas escalas. Estas señales complementarias permiten localizar material teñido con diferente intensidad y extensión.',
 'density',rows=[[0],[1,2],[3],[4]])
profile('stain','Separación cromática y de tinción',
 ['Imagen suavizada','Conversión a CIELAB','H&E, Blue Ratio y densidad óptica','Mapas de color y tinción'],
 'CIELAB separa luminancia y cromaticidad, mientras que H&E, Blue Ratio y densidad óptica describen aspectos complementarios de la tinción. Las respuestas se normalizan para construir candidatos y fronteras regionales.',
 'density',rows=[[0],[1,2],[3]])
profile('appearance_maps','Color y textura local',
 ['Imagen RGB','CIELAB y tinción H&E','Variación local y textura','Vectores de apariencia regional'],
 'Los canales de color y tinción se complementan con la variación de intensidad del vecindario. Su combinación describe la apariencia del citoplasma y del tejido exterior sin reducirlos a un único umbral.',
 rows=[[0],[1,2],[3]])
profile('markers','Marcadores cromatínicos',
 ['Mapas de tinción','Candidatos de material oscuro','Componentes conectados','Asignación de foco y fondo','Marcadores para segmentación'],
 'Los candidatos oscuros se separan en componentes conectados y se revisan respecto al foco celular. Su posición y tinción permiten asignar marcadores de primer plano, fondo y regiones inciertas.')
profile('grabcut','Segmentación con GrabCut',
 ['Imagen RGB','Marcadores de foco y fondo','Modelos de apariencia','Minimización del corte de grafos','Máscara celular candidata'],
 'GrabCut ajusta modelos de apariencia de primer plano y fondo a partir de los marcadores. Un corte de grafos combina esos modelos con la continuidad entre píxeles para obtener la región celular candidata.',
 'energy',rows=[[0,1],[2],[3],[4]],notes=[(2,'Modelos de primer plano y fondo'),(3,'Marcadores fijos y probables')])
profile('reconstruct_mast','Reconstrucción BR/H',
 ['Respuesta Blue Ratio y hematoxilina','Umbral alto: semillas','Umbral bajo: región admisible','Reconstrucción por conectividad','Revisión de agrupaciones añadidas','Máscara cromatínica'],
 'La respuesta BR/H define semillas intensamente teñidas y una región débil que puede recuperarse por conectividad. Los componentes añadidos se aceptan según distancia, apoyo de tinción y barrera de borde; los dudosos quedan separados.',
 'reconstruction',rows=[[0],[1,2],[3],[4],[5]],notes=[(0,'Percentil local: 86 %'),(4,'Distancia y apoyo del borde')])
profile('association','Asociación de agrupaciones',
 ['Componentes cromatínicos','Foco de referencia y candidatos','Contacto, distancia y tinción','Revisión de barreras entre grupos','Agrupaciones aceptadas y ambiguas'],
 'La asociación evalúa contacto, separación y continuidad de tinción entre componentes. Las barreras de la imagen ayudan a distinguir agrupaciones del foco de estructuras vecinas o material ambiguo.')
profile('rescue_mast','Rescate del foco oscuro',
 ['Mapas Blue Ratio, H&E y densidad','Identificación de focos densos','Dominancia y posición del foco','Comprobación de tinción y frontera','Máscaras revisadas'],
 'El rescate revisa focos oscuros con apoyo de varias señales de tinción. Solo actualiza la propuesta cuando la dominancia, la posición y el soporte de las fronteras cumplen los criterios establecidos; en otro caso conserva la estimación disponible.')
profile('local_contour','Refinamiento local del contorno',
 ['Máscara candidata y cromatina','Banda de transición admisible','Refinamiento restringido del contorno','Conectividad con el foco','Máscara celular refinada'],
 'El contorno se revisa dentro de una banda admisible definida por el tejido y los marcadores. La cromatina y el contacto con el foco restringen la modificación para conservar una región celular coherente.')
profile('holes','Relleno de huecos pequeños',
 ['Máscara y dominio admisible','Componentes del complemento','Selección de huecos por área','Unión con la máscara','Máscara sin huecos pequeños'],
 'Se identifican regiones del complemento y se rellenan los huecos que cumplen el límite de área y las restricciones del dominio. Este ajuste elimina discontinuidades internas pequeñas sin incorporar regiones exteriores.',
 notes=[(2,'Límite de área del pipeline')])
profile('independent','Conservación de cromatina',
 ['Máscara cromatínica seleccionada','Conservación independiente del cuerpo','Máscaras celular y cromatínica'],
 'La cromatina seleccionada se conserva de forma independiente de la estimación celular. La posible discordancia entre ambas máscaras se mantiene visible y se utiliza en el control de calidad.')
profile('multi_competition','Multi-Otsu y competencia regional',
 ['Mapas de tinción multicanal','Umbralización Multi-Otsu','Marcadores de focos y exterior','Competencia entre territorios','Máscaras regionales candidatas'],
 'Multi-Otsu separa clases de tinción y facilita la construcción de marcadores. La competencia regional delimita territorios de los focos y del exterior, a partir de los cuales se forman las propuestas celular y cromatínica.',
 'multiotsu')
profile('color_territory','Territorios por coste cromático',
 ['Mapas de tinción y fronteras','Jerarquía regional y focos','Costes de pertenencia cromática','Competencia y asociación de regiones','Territorio celular y cromatina'],
 'El color, la tinción y las fronteras definen costes de acceso a los territorios de los focos. La competencia por coste y la asociación de regiones permiten estimar el cuerpo sin imponer una envolvente geométrica fija.',
 'cost')
profile('graph_merge','Grafo multicanal y fusión',
 ['Vectores de color y textura','Parcelas de la imagen','Grafo de regiones adyacentes','Fusión por apariencia y barrera','Regiones agrupadas'],
 'La imagen se divide en parcelas y se conectan las que comparten frontera. La fusión combina semejanza de apariencia y barrera de contacto, actualizando las propiedades de las regiones agregadas.',
 notes=[(1,'Parcelas conectadas'),(3,'Apariencia y frontera de contacto')])
profile('relief','Relieve local y focos',
 ['Tinción y densidad óptica','Relieve local del material oscuro','Reconstrucción de candidatos','Asociación con el foco','Propuesta cromatínica'],
 'El relieve de tinción permite recuperar material oscuro que una separación global puede omitir. Los candidatos se reconstruyen y se revisan por contacto y apoyo multicanal antes de asociarlos al foco.')
profile('select_chrom','Selección cromatínica por evidencia',
 ['Propuestas cromatínicas','Evaluación de tinción y delimitación','Comparación de la mejora de evidencia','Selección de la propuesta cromatínica'],
 'Las propuestas cromatínicas se comparan con indicadores de tinción y delimitación obtenidos de la imagen. Una alternativa sustituye a la mejor disponible cuando supera la ganancia mínima establecida; el criterio no expresa una probabilidad de exactitud anatómica.',
 'gain',notes=[(2,'Ganancia mínima de evidencia')])
profile('appearance_cut','Apariencia regional y cortes',
 ['Color, cromatina y regiones rivales','Muestras de citoplasma y exterior','Ajuste de mezclas gaussianas','Costes regionales y corte espacial','Propuesta celular por apariencia'],
 'Las muestras citoplasmáticas y exteriores definen modelos de apariencia de la imagen actual. Sus diferencias de verosimilitud se combinan con las fronteras y las restricciones espaciales para generar una propuesta celular.',
 'gmm')
profile('select_body','Selección de la región celular',
 ['Propuestas celulares','Consistencia de máscaras y vecindad','Apoyo de la frontera celular','Ordenación por consistencia y evidencia','Máscara celular seleccionada'],
 'Las estimaciones celulares se revisan por concordancia con la cromatina, posible intrusión de vecinos y apoyo del borde. La selección prioriza las propuestas sin incidencias y utiliza después la puntuación de evidencia celular.')
profile('connect','Conectividad con el foco',
 ['Máscara candidata y foco cromatínico','Restricción al dominio admisible','Componentes conectados al foco','Máscara celular conectada'],
 'La máscara candidata se restringe al tejido admisible y se conservan los componentes que contactan con el foco. Así se eliminan fragmentos aislados y regiones excluidas sin alterar la cromatina seleccionada.',
 'connectivity',rows=[[0],[1],[2],[3]])
profile('gradient','Gradientes multicanal',
 ['Canales de color o tinción','Suavizado a la escala utilizada','Derivadas horizontales y verticales','Magnitud del gradiente','Mapa de fronteras'],
 'Los gradientes miden cambios locales de luminancia, cromaticidad o tinción después del suavizado previsto. La magnitud de las derivadas refuerza las transiciones que actúan como barreras entre regiones.',
 'gradient',notes=[(1,'Escala de suavizado registrada')])
profile('lymph_basins','Cuencas multicanal y locales',
 ['Tinción y fronteras multicanal','Marcadores del espacio libre','Propagación regional de etiquetas','Cobertura y asociación cromatínica','Cuencas celulares candidatas'],
 'Los marcadores del espacio libre y las fronteras multicanal organizan los territorios del linfoma. La propagación regional y la revisión de cobertura asocian material cromatínico y prolongaciones con apoyo en la imagen.')
profile('basin_hierarchy','Jerarquía regional de cuencas',
 ['Relieve de tinción y fronteras','Marcadores jerárquicos','Formación y fusión de cuencas','Asociación de focos a regiones','Territorios celulares asociados'],
 'La jerarquía agrupa cuencas según su apariencia y las barreras que las separan. Los focos cromatínicos seleccionan regiones compatibles y orientan la extensión del territorio celular.')
profile('contact_lymph','Asociación por contacto y tinción',
 ['Grupos cromatínicos y regiones fusionadas','Contactos entre agrupaciones','Continuidad H&E y Blue Ratio','Control de separación y barreras','Cromatina asociada'],
 'El contacto territorial se contrasta con la continuidad de H&E y Blue Ratio. La distancia y las barreras entre grupos limitan asociaciones que podrían incluir material de un foco vecino.')
profile('band','Corte de una banda de transición',
 ['Cromatina y cuerpo provisional','Muestras interiores','Muestras exteriores','Contraste de apariencia y corte espacial','Contorno celular de transición'],
 'Una banda alrededor del cuerpo provisional aporta muestras del citoplasma y del exterior. Sus modelos de apariencia producen un contraste regional que se combina con las fronteras para revisar la transición celular.',
 'energy',rows=[[0],[1,2],[3],[4]],notes=[(3,'Interior citoplasmático'),(3,'Banda exterior local')])
profile('joint','Selección conjunta de máscaras',
 ['Parejas de máscaras candidatas','Comprobación de consistencia','Evidencia celular y cromatínica','Selección de la pareja coherente'],
 'Se comparan parejas de máscaras para mantener la correspondencia entre cuerpo y foco cromatínico. Entre las propuestas consistentes, la puntuación combina evidencia celular y cromatínica con pesos de 0,74 y 0,26.',
 'joint',notes=[(1,'Prioridad: consistencia'),(2,'Pesos celular / cromatina: 0,74 / 0,26')])
profile('gabor','Textura y orientación de fibras',
 ['Luminancia de la imagen','Banco de filtros de Gabor','Energía de textura orientada','Tensor de estructura local','Mapas de fibras y orientación'],
 'Los filtros de Gabor describen textura en varias orientaciones y escalas. Sus energías se complementan con el tensor de estructura para representar fibras y transiciones que no se distinguen únicamente por color.',
 'gabor',notes=[(1,'Orientaciones y escalas del banco'),(3,'Dirección y contraste locales')])
profile('sarcoma_graph','Partición LAB y Gabor',
 ['Color CIELAB y textura Gabor','Parcelas de la imagen','Foco cromatínico y fondo','Pertenencia armónica en el grafo','Máscara celular candidata'],
 'Las parcelas combinan color CIELAB y textura Gabor para representar el tejido del sarcoma. La pertenencia armónica combina las afinidades entre regiones, las semillas y el mapa inicial de pertenencia para obtener una propuesta celular.',
 'harmonic')
profile('fixed_chrom','Cromatina de LAB y Gabor',
 ['Detección cromatínica LAB/Gabor','Recuperación de la cromatina completa','Foco cromatínico conservado'],
 'Se conserva la detección cromatínica de la ruta LAB/Gabor como foco común. Las alternativas posteriores estiman el cuerpo celular manteniendo esta hipótesis cromatínica.')
profile('samples','Modelos de apariencia regional',
 ['Cromatina, rivales y canales de tinción','Muestras citoplasmáticas','Muestras exteriores y competidoras','Ajuste de mezclas gaussianas','Evidencia de pertenencia regional'],
 'Las muestras se obtienen del citoplasma, del exterior y de las regiones competidoras de la imagen. Las mezclas gaussianas representan su variación cromática y permiten medir la evidencia de pertenencia de cada región.',
 'gmm',rows=[[0],[1,2],[3],[4]],notes=[(3,'Componentes según la configuración')])
profile('spatial','Corte espacial por apariencia',
 ['Grafo regional y modelos de apariencia','Costes de pertenencia celular','Penalización de discontinuidades','Corte de grafos con regiones fijas','Máscara celular candidata'],
 'El corte espacial combina costes de apariencia con una penalización por separar regiones vecinas. Las regiones de foco y exterior fijan restricciones, y el flujo máximo determina la partición celular.',
 'energy',notes=[(2,'Continuidad entre regiones'),(3,'Foco y exterior restringidos')])
profile('breast_geodesic','Contorno con interior estable',
 ['Cromatina y fronteras cromáticas','Competencia por coste geodésico','Región inicial e interior estable','Evolución geodésica restringida','Máscara celular inicial'],
 'La competencia geodésica entre el foco y el exterior proporciona una región inicial. El contorno evoluciona con la cromatina y el interior estable fijados, evitando que la revisión de la frontera vacíe esas zonas.',
 'cost',notes=[(3,'Cromatina e interior fijados')])
profile('breast_reconstruct','Multi-Otsu y reconstrucción',
 ['Eosina y luminancia inversa','Clasificación Multi-Otsu','Soporte de clases próximas y exteriores','Reconstrucción desde la cromatina','Máscara celular reconstruida'],
 'Multi-Otsu clasifica la combinación de eosina y luminancia inversa. El contraste entre clases próximas y exteriores define soporte fuerte y débil para reconstruir tejido conectado con la cromatina.',
 'multiotsu',notes=[(0,'Pesos eosina / luminancia: 0,57 / 0,43')])
profile('regional_contour','Contorno activo regional',
 ['Mapa de pertenencia celular','Máscara inicial y regiones protegidas','Evolución regional Chan–Vese','Exclusión de rivales y tejido exterior','Máscara celular refinada'],
 'Chan–Vese refina la máscara sobre el mapa de pertenencia, comparando regiones interiores y exteriores. Las zonas protegidas se conservan y se excluyen rivales y regiones sin apoyo de tejido.',
 notes=[(2,'8 iteraciones; suavizado 2')])
profile('blue_density','Blue Ratio y densidad óptica',
 ['Imagen suavizada','Cálculo de Blue Ratio','Densidad óptica y hematoxilina','Respuesta de material oscuro'],
 'Blue Ratio refuerza la contribución del canal azul respecto a los otros canales. La densidad óptica y la hematoxilina complementan esa respuesta para identificar material oscuro con diferente apariencia cromática.',
 'blueratio',rows=[[0],[1,2],[3]])
profile('mel_local','Segmentación local con control',
 ['Máscara inicial y foco cromatínico','Interior fijo y dominio admisible','Refinamiento geodésico local','Comparación del apoyo de frontera','Máscara aceptada o conservada'],
 'La revisión local utiliza el foco y un interior protegido para restringir la evolución geodésica. El cambio solo se acepta si cumple los controles de extensión, cromatina y soporte de la frontera.')
profile('mel_reconstruct','Doble umbral y reconstrucción',
 ['Respuesta Blue Ratio y hematoxilina','Semillas por umbral alto','Región débil por umbral bajo','Reconstrucción por conectividad','Candidato cromatínico revisado'],
 'Dos umbrales separan semillas intensas y una región de apoyo más débil. La reconstrucción por conectividad recupera material unido a las semillas y permite revisar los componentes cromatínicos.',
 'reconstruction',rows=[[0],[1,2],[3],[4]])
profile('dense','Identificación de focos densos',
 ['Densidad óptica y Blue Ratio','Respuesta ponderada de tinción','Componentes de material oscuro','Separación de focos y material incierto'],
 'La densidad óptica y Blue Ratio se combinan con señales de tinción para localizar focos oscuros. Los componentes se separan y se prepara su revisión como foco dominante, competidor o material incierto.')
profile('competitors','Revisión de competidores externos',
 ['Focos oscuros y regiones vecinas','Separación y contacto territorial','Tinción y barrera entre focos','Clasificación de competidores y ambiguos'],
 'Los focos externos se revisan mediante su separación, su contacto territorial y la continuidad de tinción. Las barreras de la imagen ayudan a distinguir material asociado de competidores o grupos ambiguos.')
profile('dominant','Selección del foco dominante',
 ['Focos cromatínicos candidatos','Dominancia y posición espacial','Apoyo de tinción y separación','Selección o conservación del foco'],
 'La elección del foco combina dominancia, posición y evidencia de tinción con controles sobre sus competidores. La propuesta disponible se conserva cuando ninguna alternativa reúne evidencia suficiente.')
profile('mel_refine','Refinamiento celular por evidencia',
 ['Foco seleccionado y cuerpo provisional','Propuesta de banda exterior, si procede','Apoyo de tinción, borde y vecindad','Aceptación o conservación del cuerpo'],
 'La ampliación por banda exterior se considera únicamente cuando se activa la condición del pipeline. La aceptación exige apoyo de apariencia y frontera, mejor cobertura cromatínica y ausencia de intrusión excesiva.')
profile('neuro_territory','Territorios de tejido y tinción',
 ['Mapas de tinción y fronteras','Jerarquía regional del tejido','Asociación de focos cromatínicos','Competencia por costes territoriales','Máscaras de cuerpo y cromatina'],
 'La jerarquía de regiones organiza el tejido y la tinción alrededor de los focos cromatínicos. La asociación de grupos y la competencia por costes delimitan territorios celulares frente a regiones rivales.',
 'cost')
profile('neuro_intensity','Intensidad y densidad cromatínica',
 ['Componentes cromatínicos','Otsu dentro del componente','Densidad y contraste local','Reconstrucción de material apoyado','Propuestas cromatínicas revisadas'],
 'Los componentes extensos se revisan mediante Otsu intracomponente o densidad y contraste local. Las condiciones de tamaño, soporte y conectividad limitan la retirada de material débil antes de comparar las propuestas.',
 'reconstruction',rows=[[0],[1,2],[3],[4]])
profile('neuro_walker','Multi-Otsu y Random Walker',
 ['Tinción multicanal','Multi-Otsu y marcadores','Propagación Random Walker','Asociación de grupos y control de rivales','Territorios celulares candidatos'],
 'Cuando la evidencia cromatínica inicial es débil, Multi-Otsu prepara candidatos y marcadores para Random Walker. La propagación y la revisión de contacto entre grupos producen una alternativa territorial, sin sustituir automáticamente la ruta inicial.',
 'walker',notes=[(2,'Alternativa condicional')])

ASSIGNMENTS = {
 'mastocitoma': ['bilateral','multiscale','stain','markers','grabcut','reconstruct_mast','association','rescue_mast','local_contour','holes','independent'],
 'pulmon': ['bilateral','multiscale','appearance_maps','multi_competition','color_territory','graph_merge','relief','select_chrom','appearance_cut','select_body','connect','holes','independent'],
 'linfoma': ['bilateral','stain','gradient','lymph_basins','basin_hierarchy','graph_merge','contact_lymph','band','joint','connect','holes','independent'],
 'sarcoma': ['bilateral','stain','gabor','sarcoma_graph','fixed_chrom','samples','band','spatial','select_body','connect','holes','fixed_chrom'],
 'mama': ['bilateral','multiscale','gradient','breast_geodesic','breast_reconstruct','band','spatial','regional_contour','select_body','connect','holes','independent'],
 'melanoma': ['bilateral','blue_density','stain','mel_local','mel_reconstruct','dense','competitors','dominant','band','mel_refine','connect','holes'],
 'neuroendocrino': ['bilateral','appearance_maps','gradient','neuro_territory','neuro_intensity','neuro_walker','select_chrom','samples','spatial','select_body','connect','holes','independent'],
}

FORMULAS = {
 'bilateral': (r'$I_f(p)=\frac{\sum_{q\in\Omega_p}G_{\sigma_s}(\|p-q\|)\,G_{\sigma_c}(\|I(p)-I(q)\|)\,I(q)}{\sum_{q\in\Omega_p}G_{\sigma_s}(\|p-q\|)\,G_{\sigma_c}(\|I(p)-I(q)\|)}$', 'Promedio ponderado por proximidad espacial y semejanza cromática'),
 'density': (r'$\mathrm{DO}(I)=-\log\!\left(\frac{I+1}{256}\right)$','Densidad óptica de los canales de intensidad'),
 'blueratio': (r'$\mathrm{BR}=\frac{100B}{1+R+G}\,\frac{256}{1+R+G+B}$','Blue Ratio calculado a partir de los canales RGB'),
 'gradient': (r'$\|\nabla I\|=\sqrt{I_x^2+I_y^2}$','Magnitud de las derivadas horizontales y verticales'),
 'gabor': (r'$g(x,y)=\exp\!\left(-\frac{x^{\prime 2}+\gamma^2 y^{\prime 2}}{2\sigma^2}\right)\cos\!\left(\frac{2\pi x^{\prime}}{\lambda}+\psi\right)$','Filtro de Gabor: envolvente gaussiana y oscilación orientada'),
 'multiotsu': (r'$\boldsymbol{t}^{*}=\mathrm{arg\,max}_{\boldsymbol{t}}\ \sum_{k=0}^{K-1}\omega_k\,(\mu_k-\mu)^2$','Multi-Otsu: maximización de la varianza entre clases'),
 'energy': (r'$E(L)=\sum_p D_p(L_p)+\eta\sum_{(p,q)}w_{pq}\,\mathbf{1}[L_p\ne L_q]$','Energía de apariencia y continuidad entre regiones vecinas'),
 'gmm': (r'$p(\mathbf{x})=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$','Mezcla de distribuciones gaussianas para la apariencia'),
 'reconstruction': (r'$\mathcal{R}_{W}(H)=\bigcup_{C\in\mathcal{C}(W):\,C\cap H\ne\varnothing} C$','Reconstrucción: componentes de la región admisible que contactan con las semillas'),
 'connectivity': (r'$M_{\mathrm{conectada}}=\bigcup_{C\in\mathcal{C}(M):\,C\cap F\ne\varnothing} C$','Componentes de la máscara que contactan con el foco'),
 'gain': (r'$S_{\mathrm{nueva}}\ \geq\ S_{\mathrm{mejor}}+\Delta$','Condición de ganancia mínima de evidencia cromatínica'),
 'joint': (r'$S=0.74\,S_{\mathrm{celular}}+0.26\,S_{\mathrm{cromatina}}$','Puntuación conjunta entre las propuestas consistentes'),
 'walker': (r'$\mathbf{L}_{uu}\,\mathbf{p}_{u}=-\mathbf{L}_{um}\,\mathbf{p}_{m}$','Propagación armónica con marcadores fijados en el grafo'),
 'harmonic': (r'$(\mathbf{L}+\mathbf{C})\,\mathbf{p}=\mathbf{C}\,\mathbf{p}_0$','Pertenencia armónica con afinidades regionales y confianza en el mapa inicial'),
 'cost': (r'$d(s,p)=\min_{\pi:s\to p}\ \sum_{(u,v)\in\pi}c(u,v)$','Distancia geodésica de coste mínimo entre semilla y píxel'),
}


def text(lines,x,y,size=16,bold=False,anchor='middle',attrs='',leading=19):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"'+
            (' font-weight="bold"' if bold else '')+attrs+'>'+''.join(
                f'<tspan x="{x}" dy="{0 if i==0 else leading}">{escape(line)}</tspan>'
                for i,line in enumerate(lines))+'</text>')


def figure(stage,selected,heading,recipe):
    stage_height=58+len(stage)*80
    row_heights=[]
    for row in recipe['rows']:
        max_lines=max(len(textwrap.wrap(recipe['blocks'][index],18 if len(row)>1 else 33)) for index in row)
        row_heights.append(max(58, max_lines*19+18))
    graph_height=sum(row_heights)+24*(len(row_heights)-1)
    height=max(stage_height,graph_height+100)+24
    content=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc" data-block-count="{len(recipe["blocks"])}">',
        f'<title id="title">{escape(recipe["title"])}</title>',
        '<desc id="desc">Etapa principal a la izquierda, operación seleccionada en lavanda y resumen metodológico a la derecha. Las imágenes del análisis se muestran debajo de esta figura.</desc>',
        '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="black"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="black" stroke-width="1">',
        f'<rect x="8" y="12" width="220" height="{stage_height}" fill="white" stroke="black"/>',
        '<rect x="8" y="12" width="220" height="28" fill="#EBE0F3" stroke="black"/>',text([heading],118,32,bold=True)]
    for i,node in enumerate(stage):
        y=58+i*80;active=node.get('data-operation')==selected
        content.append(f'<g data-stage-operation="{node.get("data-operation")}" data-selected="{str(active).lower()}"><rect x="26" y="{y}" width="184" height="60" fill="{"#EBE0F3" if active else "white"}" stroke="black"/>')
        lines=[part.text or '' for part in node.iter() if part.tag.endswith('tspan')]
        content.append(text(lines,118,y+35-(len(lines)-1)*8.5,size=15,bold=active,leading=17));content.append('</g>')
        if i<len(stage)-1:content.append(f'<path d="M118 {y+60} V{y+76}" stroke="black" fill="none" marker-end="url(#arrow)"/>')
        if active:
            content.append(f'<path d="M210 {y} L300 12 M210 {y+60} L300 {height-12}" stroke="black" fill="none" data-expansion-lines="2"/>')
    content += [f'<rect x="300" y="12" width="688" height="{height-24}" fill="white" stroke="black"/>',
                '<rect x="300" y="12" width="688" height="28" fill="#EBE0F3" stroke="black"/>',text([recipe['title']],644,32,bold=True)]
    positions={};y=56+max(0,(height-78-graph_height)/2)
    for row,rh in zip(recipe['rows'],row_heights):
        for column,index in enumerate(row):
            width=340 if len(row)==1 else 160;x=340 if len(row)==1 else 340+column*180
            positions[index]=(x,y,width,rh)
        y+=rh+24
    for first,last in recipe['edges']:
        x,y,w,h=positions[first];tx,ty,tw,_=positions[last];middle=(y+h+ty)/2
        content.append(f'<path d="M{x+w/2} {y+h} V{middle} H{tx+tw/2} V{ty-3}" stroke="black" fill="none" marker-end="url(#arrow)"/>')
    for index,label in enumerate(recipe['blocks']):
        x,y,w,h=positions[index];lines=textwrap.wrap(label,18 if w<200 else 33)
        content.append(f'<g data-method-block="{index}"><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="black"/>')
        content.append(text(lines,x+w/2,y+h/2+5-(len(lines)-1)*9.5,size=16));content.append('</g>')
    for slot,(index,label) in enumerate(recipe['notes']):
        x,y,w,h=positions[index];center=y+h/2
        # Separate two annotations of one operation; arrows remain short and straight.
        same=[i for i,(target,_) in enumerate(recipe['notes']) if target==index]
        note_y=center+(same.index(slot)*52-26 if len(same)>1 else 0)
        content.append(f'<path d="M718 {note_y} H{x+w+3}" fill="none" stroke="black" marker-end="url(#arrow)"/>')
        content.append(text(textwrap.wrap(label,27),730,note_y-8,size=14,anchor='start',attrs=f' data-parameter-slot="{slot}"'))
    content.append('</g></svg>')
    return '\n'.join(content)


def generate():
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties
    import matplotlib
    matplotlib.rcParams.update({'mathtext.fontset':'stix','svg.fonttype':'path','svg.hashsalt':'histovision-methods'})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'formulas').mkdir(exist_ok=True)
    for key,(formula,label) in FORMULAS.items():
        path=OUT/'formulas'/f'{key}.svg'
        math_to_image(formula,path,prop=FontProperties(size=23),dpi=144,format='svg',color='black')
        source=path.read_text(encoding='utf8')
        # Strip variable creation timestamps to keep resource generation deterministic.
        import re
        source=re.sub(r'<dc:date>.*?</dc:date>','',source)
        path.write_text(source,encoding='utf8')
    catalog={}
    for main in sorted((ROOT/'assets'/'diagrams').glob('*.svg')):
        tree=ET.fromstring(main.read_text(encoding='utf8'));slug=main.stem.split('_',1)[1]
        nodes=[node for node in tree.iter() if node.get('data-operation')]
        assert len(nodes)==len(ASSIGNMENTS[slug])
        for node,key in zip(nodes,ASSIGNMENTS[slug]):
            operation=node.get('data-operation');recipe=PROFILES[key]
            assert 2<=len(recipe['blocks'])<=6 and len(recipe['notes'])<=2
            rect=next(child for child in node if child.tag.endswith('rect'));x=rect.get('x')
            stage=[other for other in nodes if next(child for child in other if child.tag.endswith('rect')).get('x')==x]
            center=float(x)+float(rect.get('width'))/2
            heading=next(item.text for item in tree.iter() if item.tag.endswith('text') and item.get('font-weight')=='bold' and float(item.get('x','-1'))==center)
            path=OUT/f'{operation}.svg'
            path.write_text(figure(stage,operation,heading,recipe),encoding='utf8')
            catalog[operation]=dict(figure=path.relative_to(ROOT).as_posix(),profile=key,
                explanation=recipe['explanation'],title=recipe['title'],blocks=len(recipe['blocks']))
            if recipe['formula']:
                formula=recipe['formula'];catalog[operation]['formula']=dict(
                    image=f'assets/methods/formulas/{formula}.svg',label=FORMULAS[formula][1])
    (OUT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f'{len(catalog)} láminas metodológicas y {len(FORMULAS)} fórmulas SVG generadas.')


if __name__=='__main__':
    generate()
