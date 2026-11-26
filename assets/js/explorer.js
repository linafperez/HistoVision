const NS="http://www.w3.org/2000/svg";
const element=(tag,text,className)=>{
 const node=document.createElement(tag);
 if(text!==undefined) node.textContent=text;
 if(className) node.className=className;
 return node;
};
// Internal identifiers stay in memory; presentation uses methodological labels.
const views={
 bilateralFilter:"Imagen suavizada",cvtColor:"Conversión de color",grad:"Mapa de gradiente",
 threshold:"Umbralización",threshold_otsu:"Umbralización de Otsu",threshold_multiotsu:"Umbralización Multi-Otsu",
 getGaborKernel:"Filtro orientado",filter2D:"Respuesta de textura",energia_orientada:"Energía y orientación",
 mapas_tincion_multiescala:"Tinción multiescala",base_maps:"Canales cromáticos",
 mapas_region_tincion:"Canales de tinción",mapas_apariencia:"Apariencia y textura",
 grabCut:"Segmentación con marcadores",watershed:"Cuencas de segmentación",
 select_body:"Selección celular",select_chromatin:"Selección cromatínica",
 ajustar_mezcla:"Modelo de apariencia",modelos_de_apariencia:"Apariencia regional",
 corte_en_banda:"Banda de transición",corte_espacial:"Corte espacial",
 small_holes:"Relleno de huecos",historical_holes:"Relleno de huecos",
 attached:"Conectividad con el foco",attached_clean:"Región conectada",
 formar_resultado_regional:"Máscaras regionales",segmentar_recorte:"Máscaras de la ruta completa"
};
const names={
 rgb:"Imagen RGB",src:"Imagen de entrada",img:"Imagen RGB",image:"Imagen de entrada",
 cell:"Región celular",chrom:"Cromatina",chromatin:"Cromatina",filtered:"Imagen suavizada",smooth:"Imagen suavizada",
 h:"Hematoxilina",e:"Eosina",od:"Densidad óptica",ratio:"Blue Ratio",l:"Luminancia",a:"Canal a de CIELAB",b:"Canal b de CIELAB",
 mask:"Máscara",labels:"Regiones etiquetadas",markers:"Marcadores",seeds:"Semillas",probability:"Mapa de pertenencia",
 target:"Foco celular",other:"Regiones competidoras",rivals:"Regiones competidoras",excluded:"Regiones excluidas",
 omitted:"Material ambiguo",candidate:"Máscara candidata",edge:"Fronteras",edge4:"Fronteras multicanal",edge6:"Fronteras regionales",
 strong:"Semillas intensas",weak:"Región de apoyo",high:"Semillas intensas",low:"Región de apoyo",domain:"Dominio admisible",
 initial:"Región inicial",raw:"Máscara candidata",response:"Respuesta de tinción",score:"Respuesta de tinción",
 gabor:"Textura orientada",fibers:"Textura de fibras",texture:"Textura local",exterior:"Muestras exteriores",
 source:"Muestras de referencia",core:"Interior protegido",allowed:"Dominio admisible",owner:"Territorios etiquetados"
};
const preferred={
 bilateral:["bilateralFilter"],multiscale:["mapas_tincion_multiescala"],stain:["mapas_region_tincion","base_maps"],
 appearance_maps:["mapas_apariencia"],markers:["marker_image","build_markers"],grabcut:["grabCut"],
 reconstruct_mast:["reconstruir_mastocitoma"],rescue_mast:["rescatar_mastocitoma"],
 local_contour:["resolve_mast_band","refine_cell"],holes:["small_holes","historical_holes"],
 independent:["segmentar_recorte"],multi_competition:["segmentar_competencia_multicanal"],
 color_territory:["segmentar_territorios_de_tincion"],graph_merge:["fusion_adyacente","grafo_multicanal"],
 relief:["reconstruccion_relieve"],select_chrom:["select_chromatin"],appearance_cut:["corte_espacial","corte_en_banda"],
 select_body:["select_body"],connect:["attached_clean","attached","formar_resultado_regional"],gradient:["grad"],
 lymph_basins:["segmentar_cuencas_multicanal","run_lymph"],basin_hierarchy:["segmentar_cuencas_jerarquicas"],
 contact_lymph:["asociacion_por_contacto"],band:["corte_en_banda"],joint:["segmentar_recorte"],
 gabor:["energia_orientada","getGaborKernel"],sarcoma_graph:["sarcoma_partition","run_sarcoma"],
 fixed_chrom:["full_chromatin"],samples:["modelos_de_apariencia","ajustar_mezcla"],spatial:["corte_espacial","corte_en_banda"],
 breast_geodesic:["run_breast","breast_evolution"],breast_reconstruct:["reconstruir_mama"],
 regional_contour:["contorno_regional"],blue_density:["mapas_tincion_multiescala"],mel_local:["run_melanoma"],
 mel_reconstruct:["reconstruccion_melanoma"],dense:["preparar_focos_densos"],competitors:["seleccionar_foco_melanoma"],
 dominant:["seleccionar_foco_melanoma"],mel_refine:["formar_resultado_regional","segmentar_recorte"],
 neuro_territory:["segmentar_territorios_neuro"],neuro_intensity:["nucleo_por_intensidad","nucleo_por_densidad"],
 neuro_walker:["segmentar_random_walker"]
};
function pictureName(item,index,kind,hint) {
 const parts=item.nombre.split(" · ");
 const translated=parts.map(part=>names[part]||(/^[A-ZÁÉÍÓÚ]/u.test(part) ? part : null)).filter(Boolean);
 const channel=parts.find(part=>/^canal \d+$/.test(part));
 const rgb=item.tipo==="uint8" && item.dimensiones.length===3 && item.dimensiones[2]===3;
 const title=translated.at(-1)||hint||(rgb ? "Imagen de "+kind : item.tipo==="bool" ? "Máscara de "+kind : "Mapa de "+kind);
 return title+(channel ? " · "+channel : "")+" · "+(index+1);
}
function pictures(items,library,kind,hint) {
 const grid=element("div",undefined,"step-images");
 items.forEach((item,index)=>{
   const figure=element("figure"),img=element("img"),title=pictureName(item,index,kind,hint);
   img.src=library[item.imagen_id];img.alt=title;img.loading="lazy";
   figure.append(img,element("figcaption",title));grid.append(figure);
 });
 return grid;
}
function comparisonItems(items,profile,kind) {
 const chromatic=["reconstruct_mast","mel_reconstruct","neuro_intensity","markers","dense","relief"];
 let patterns=[];
 if(kind==="entrada" && chromatic.includes(profile)) patterns=[/Blue Ratio/,/Hematoxilina/,/Densidad óptica/];
 if(kind==="salida" && ["multiscale","stain","blue_density"].includes(profile)) patterns=[/Hematoxilina/,/Blue Ratio/,/Densidad óptica/];
 if(kind==="salida" && ["reconstruct_mast","mel_reconstruct","select_chrom","fixed_chrom","dominant","competitors","neuro_intensity"].includes(profile)) patterns=[/Cromatina/];
 const ordered=[...items].sort((a,b)=>{
   const rank=item=>{const index=patterns.findIndex(pattern=>pattern.test(item.nombre));return index<0 ? 99 : index;};
   return rank(a)-rank(b);
 });
 const limit=kind==="salida" && !["multiscale","stain","blue_density","independent","joint"].includes(profile) ? 1 : 2;
 return ordered.slice(0,limit);
}
function wrap(text,limit=27) {
 const words=text.split(/\s+/),lines=[];let line="";
 for(const word of words) {
   if(line && (line+" "+word).length>limit) {lines.push(line);line="";}
   line+=(line ? " " : "")+word;
 }
 if(line) lines.push(line);
 return lines;
}
function annotations(profile,call) {
 const p=call?.parametros||{};
 if(profile==="bilateral" && p.d!==undefined) return ["Diámetro: "+p.d+" px","σ espacial = "+p.sigmaSpace+"; σ color = "+p.sigmaColor];
 if(profile==="holes") {
   const area=p.max_area??p.max_size;
   return area===undefined ? [] : ["Umbral de área: "+area+" px²"];
 }
 if(profile==="gradient" && p.sigma!==undefined) return ["Suavizado: σ = "+p.sigma+" px"];
 if(profile==="select_chrom" && p.required_gain!==undefined) return ["Ganancia mínima: "+p.required_gain];
 if(profile==="grabcut" && p.iterCount!==undefined) return ["Iteraciones: "+p.iterCount,"Marcadores de foco y fondo"];
 if(profile==="gabor" && p.lambd!==undefined) return ["Longitud de onda: "+p.lambd+" px","Orientación: "+p.theta+" rad"];
 if(profile==="band" && p.exterior_width!==undefined) return ["Interior citoplasmático","Anchura exterior: "+p.exterior_width+" px"];
 if(profile==="samples" && p.order!==undefined) return ["Máximo de componentes: "+p.order];
 return [];
}
function histogram(data) {
 const svg=document.createElementNS(NS,"svg");svg.setAttribute("viewBox","0 0 520 180");
 svg.setAttribute("role","img");svg.setAttribute("aria-label","Histograma del canal analizado y umbrales obtenidos");
 const peak=Math.max(...data.frecuencias,1),n=data.frecuencias.length;
 data.frecuencias.forEach((v,i)=>{
   const bar=document.createElementNS(NS,"rect");
   for(const [key,value] of Object.entries({x:25+i*470/n,y:145-v/peak*125,width:470/n,height:v/peak*125,fill:"#9c82ad"})) bar.setAttribute(key,value);
   svg.append(bar);
 });
 for(const value of data.umbrales) {
   const x=25+470*(value-data.bordes[0])/(data.bordes.at(-1)-data.bordes[0]||1);
   const line=document.createElementNS(NS,"path");line.setAttribute("d",`M${x} 12 V150`);line.setAttribute("stroke","#9b2e35");svg.append(line);
 }
 const label=document.createElementNS(NS,"text");label.setAttribute("x",25);label.setAttribute("y",174);label.setAttribute("font-size",12);
 label.textContent="Umbrales obtenidos: "+data.umbrales.join(", ");svg.append(label);return svg;
}

export class Explorer {
 constructor(container,panel) {this.container=container;this.panel=panel;this.version=0;this.callVersion=0;this.urls=[];this.clear();}
 releaseFigures() {this.urls.forEach(url=>URL.revokeObjectURL(url));this.urls=[];}
 clear() {
   this.version++;this.callVersion++;this.releaseFigures();this.operations=[];this.library={};this.catalog={};this.resources=new Map();this.selected=null;
   this.panel.hidden=true;this.panel.replaceChildren();this.container.replaceChildren();this.container.removeAttribute("aria-busy");
 }
 async load(url,operations,library) {
   this.clear();const version=this.version;this.operations=operations||[];this.library=library||{};
   this.container.setAttribute("aria-busy","true");
   try {
     const [diagram,catalog]=await Promise.all([fetch(url),fetch("assets/methods/catalog.json")]);
     if(!diagram.ok || !catalog.ok) throw new Error("No se pudieron cargar las figuras del método.");
     const [source,definitions]=await Promise.all([diagram.text(),catalog.json()]);
     if(version!==this.version) return;
     this.catalog=definitions;
     const xml=new DOMParser().parseFromString(source,"image/svg+xml");
     if(xml.querySelector("parsererror")) throw new Error("El diagrama no es válido.");
     const svg=document.importNode(xml.documentElement,true);
     for(const group of svg.querySelectorAll("[data-operation]")) {
       const activate=()=>this.select(group.dataset.operation);
       group.addEventListener("click",activate);
       group.addEventListener("keydown",event=>{
         if(event.key==="Enter" || event.key===" ") {event.preventDefault();activate();}
       });
     }
     this.container.replaceChildren(svg);
   } catch(error) {if(version===this.version) this.container.textContent=error.message;}
   finally {if(version===this.version) this.container.removeAttribute("aria-busy");}
 }
 select(id) {
   const operation=this.operations.find(item=>item.id===id),definition=this.catalog[id];
   if(!operation || !definition) return;
   if(this.selected===id) {this.deselect();return;}
   this.callVersion++;this.releaseFigures();this.selected=id;
   for(const group of this.container.querySelectorAll("[data-operation]")) group.setAttribute("aria-pressed",String(group.dataset.operation===id));
   this.panel.hidden=false;this.panel.replaceChildren();
   const header=element("div",undefined,"step-heading"),close=element("button","Cerrar","secondary");
   close.type="button";close.addEventListener("click",()=>this.deselect());header.append(element("h3",operation.nombre),close);this.panel.append(header);
   const content=element("div");this.panel.append(content);
   const defaultIndex=(preferred[definition.profile]||[]).map(name=>operation.llamadas.findIndex(call=>call.funcion===name)).find(index=>index>=0)??0;
   // Recorded views change the image comparison and two annotation slots only.
   // The composed figure has no selectable blocks or deeper methodological levels.
   if(operation.llamadas.length>1) {
     const label=element("label","Resultado intermedio"),select=element("select");select.setAttribute("aria-label","Resultado intermedio");
     operation.llamadas.forEach((call,index)=>{
       const option=element("option",`${index+1} · ${views[call.funcion]||definition.title}`);option.value=index;select.append(option);
     });
     select.value=defaultIndex;
     select.addEventListener("change",()=>this.call(operation.llamadas[Number(select.value)],content,operation,definition));
     label.append(select);this.panel.insertBefore(label,content);
   }
   this.call(operation.llamadas[defaultIndex],content,operation,definition);
 }
 deselect() {
   this.selected=null;this.callVersion++;this.releaseFigures();this.panel.hidden=true;this.panel.replaceChildren();
   for(const group of this.container.querySelectorAll("[data-operation]")) group.setAttribute("aria-pressed","false");
 }
 async figure(definition,call,img,version) {
   try {
     if(!this.resources.has(definition.figure)) this.resources.set(definition.figure,fetch(definition.figure).then(response=>{
       if(!response.ok) throw new Error("Figura metodológica no disponible.");return response.text();
     }));
     const source=await this.resources.get(definition.figure);
     if(version!==this.callVersion) return;
     const xml=new DOMParser().parseFromString(source,"image/svg+xml");
     const values=annotations(definition.profile,call);
     values.forEach((value,index)=>{
       const slot=xml.querySelector(`[data-parameter-slot="${index}"]`);if(!slot) return;
       const x=slot.getAttribute("x");slot.replaceChildren();
       wrap(value).forEach((line,i)=>{
         const span=xml.createElementNS(NS,"tspan");span.setAttribute("x",x);span.setAttribute("dy",i ? "19" : "0");span.textContent=line;slot.append(span);
       });
     });
     const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(xml)],{type:"image/svg+xml"}));
     this.urls.push(url);img.src=url;img.dataset.ready="true";
   } catch(error) {
     if(version===this.callVersion) img.replaceWith(element("p",error.message,"step-state"));
   }
 }
 call(call,container,operation,definition) {
   const version=++this.callVersion;this.releaseFigures();container.replaceChildren();
   const frame=element("figure",undefined,"method-figure"),img=element("img");
   img.alt=definition.title+". Etapa seleccionada a la izquierda y resumen metodológico a la derecha.";
   img.dataset.operation=operation.id;img.dataset.ready="false";frame.append(img);container.append(frame);
   this.figure(definition,call,img,version);
   container.append(element("p",definition.explanation,"step-explanation"));
   if(definition.formula) {
     const equation=element("figure",undefined,"step-equation"),formula=element("img");
     formula.src=definition.formula.image;formula.alt=definition.formula.label;equation.append(formula);container.append(equation);
   }
   if(!call) {
     container.append(element("p","Esta operación es condicional y no se ejecutó para la imagen actual. La figura resume su método; no se muestran resultados simulados.","step-state"));return;
   }
   const comparison=element("div",undefined,"step-comparison");
   const primary=new Set();
   for(const [title,items,kind] of [["Entrada real",call.entrada,"entrada"],["Salida real",call.salida,"salida"]]) {
     const selected=comparisonItems(items,definition.profile,kind);selected.forEach(item=>primary.add(item));
     const section=element("div");section.append(element("h4",title));
     section.append(items.length ? pictures(selected,this.library,kind,kind==="salida" ? views[call.funcion] : undefined) : element("p","Este resultado contiene medidas o una decisión de selección.","step-state"));comparison.append(section);
   }
   container.append(comparison);
   const remaining=[...call.entrada,...call.salida,...call.intermedios].filter(item=>!primary.has(item));
   if(remaining.length || call.histograma) {
     const detail=element("details");detail.append(element("summary","Otros canales y máscaras de esta operación"));
     if(call.histograma) detail.append(histogram(call.histograma));
     if(remaining.length) detail.append(pictures(remaining,this.library,"la operación"));container.append(detail);
   }
   if(call.propuestas) {
     const detail=element("details");detail.append(element("summary","Comparar propuestas evaluadas"));
     const choices=element("div",undefined,"step-proposals");
     for(const choice of call.propuestas) {
       const section=element("section");section.append(element("p",choice.nombre+" · "+choice.estado,choice.estado==="seleccionada" ? "chosen-proposal" : ""));
       section.append(pictures(choice.imagenes,this.library,"la propuesta"));choices.append(section);
     }
     detail.append(choices);container.append(detail);
   }
 }
}
