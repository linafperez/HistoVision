const NS = "http://www.w3.org/2000/svg";
const titles = {
 bilateralFilter:"Filtro bilateral",cvtColor:"Transformación de color",threshold:"Umbralización",
 grabCut:"GrabCut",watershed:"Watershed",getGaborKernel:"Núcleo de Gabor",filter2D:"Correlación lineal",
 grad:"Gradiente",base_maps:"Mapas cromáticos",mapas_region_tincion:"Mapas H&E y densidad óptica",
 mapas_tincion_multiescala:"Mapas de tinción multiescala",mapas_apariencia:"Mapas de apariencia",
 select_body:"Selección de región celular",select_chromatin:"Selección cromatínica",
 ajustar_mezcla:"Ajuste de mezcla gaussiana",segmentar_recorte:"Resultado de la ruta principal",
 small_holes:"Relleno de huecos",attached:"Conectividad con el foco",
 support:"Evaluación conjunta de la propuesta"
};
const formulas = {
 bilateral:'<mrow><msub><mi>I</mi><mi>f</mi></msub><mo>(</mo><mi>p</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><msub><mi>W</mi><mi>p</mi></msub></mfrac><munder><mo>∑</mo><mrow><mi>q</mi><mo>∈</mo><mi>Ω</mi></mrow></munder><msub><mi>G</mi><msub><mi>σ</mi><mi>s</mi></msub></msub><mo>(</mo><mo>‖</mo><mi>p−q</mi><mo>‖</mo><mo>)</mo><msub><mi>G</mi><msub><mi>σ</mi><mi>r</mi></msub></msub><mo>(</mo><mo>‖</mo><msub><mi>I</mi><mi>p</mi></msub><mo>−</mo><msub><mi>I</mi><mi>q</mi></msub><mo>‖</mo><mo>)</mo><msub><mi>I</mi><mi>q</mi></msub></mrow>',
 density:'<mrow><mi>DO</mi><mo>=</mo><mo>−</mo><mi>log</mi><mo>(</mo><mfrac><mrow><mi>I</mi><mo>+</mo><mn>1</mn></mrow><mn>256</mn></mfrac><mo>)</mo></mrow>',
 gradient:'<mrow><mo>‖</mo><mi>∇I</mi><mo>‖</mo><mo>=</mo><msqrt><mrow><msup><msub><mi>I</mi><mi>x</mi></msub><mn>2</mn></msup><mo>+</mo><msup><msub><mi>I</mi><mi>y</mi></msub><mn>2</mn></msup></mrow></msqrt></mrow>',
 gabor:'<mrow><mi>g</mi><mo>(</mo><mi>x,y</mi><mo>)</mo><mo>=</mo><msup><mi>e</mi><mrow><mo>−</mo><mfrac><mrow><msup><mi>x′</mi><mn>2</mn></msup><mo>+</mo><msup><mi>γ</mi><mn>2</mn></msup><msup><mi>y′</mi><mn>2</mn></msup></mrow><mrow><mn>2</mn><msup><mi>σ</mi><mn>2</mn></msup></mrow></mfrac></mrow></msup><mi>cos</mi><mo>(</mo><mfrac><mrow><mn>2π</mn><mi>x′</mi></mrow><mi>λ</mi></mfrac><mo>+</mo><mi>ψ</mi><mo>)</mo></mrow>',
 otsu:'<mrow><msup><mi>t</mi><mo>*</mo></msup><mo>=</mo><munder><mi>arg max</mi><mi>t</mi></munder><msub><mi>ω</mi><mn>0</mn></msub><mo>(</mo><mi>t</mi><mo>)</mo><msub><mi>ω</mi><mn>1</mn></msub><mo>(</mo><mi>t</mi><mo>)</mo><msup><mrow><mo>[</mo><msub><mi>μ</mi><mn>0</mn></msub><mo>−</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>]</mo></mrow><mn>2</mn></msup></mrow>',
 energy:'<mrow><mi>E</mi><mo>(</mo><mi>L</mi><mo>)</mo><mo>=</mo><munder><mo>∑</mo><mi>p</mi></munder><msub><mi>D</mi><mi>p</mi></msub><mo>(</mo><msub><mi>L</mi><mi>p</mi></msub><mo>)</mo><mo>+</mo><mi>λ</mi><munder><mo>∑</mo><mi>p,q</mi></munder><msub><mi>V</mi><mi>pq</mi></msub><mo>[</mo><msub><mi>L</mi><mi>p</mi></msub><mo>≠</mo><msub><mi>L</mi><mi>q</mi></msub><mo>]</mo></mrow>',
 gmm:'<mrow><mi>p</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><munderover><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msub><mi>π</mi><mi>k</mi></msub><mi>𝒩</mi><mo>(</mo><mi>x</mi><mo>;</mo><msub><mi>μ</mi><mi>k</mi></msub><mo>,</mo><msub><mi>Σ</mi><mi>k</mi></msub><mo>)</mo></mrow>'
};
function equationFor(call) {
 if(!call) return null;
 const name=call.funcion;
 if(name==="bilateralFilter") return "bilateral";
 if(name==="grad") return "gradient";
 if(["getGaborKernel","energia_orientada"].includes(name)) return "gabor";
 if(["mapas_tincion_multiescala","mapas_region_tincion","mapas_apariencia"].includes(name)) return "density";
 if(["ajustar_mezcla","appearance6","modelos_de_apariencia"].includes(name)) return "gmm";
 if(["grabCut","corte_espacial"].includes(name)) return "energy";
 if(name==="threshold_otsu" || (name==="threshold" && call.parametros.criterio==="Otsu")) return "otsu";
 return null;
}
const el = (tag,text,className) => {
 const node=document.createElement(tag);
 if(text!==undefined) node.textContent=text;
 if(className) node.className=className;
 return node;
};
function pictureGrid(items,library) {
 const grid=el("div",undefined,"step-images");
 for(const item of items) {
   const figure=el("figure");
   const img=el("img");img.src=library[item.imagen_id];img.alt=item.nombre;img.loading="lazy";
   figure.append(img,el("figcaption",item.nombre+" · "+item.dimensiones.slice(0,2).join(" × ")));
   grid.append(figure);
 }
 return grid;
}
function parameters(values) {
 const dl=el("dl",undefined,"step-parameters");
 for(const [name,value] of Object.entries(values)) {
   const row=el("div");
   row.append(el("dt",name),el("dd",typeof value==="string" ? value : JSON.stringify(value)));
   dl.append(row);
 }
 return dl;
}
function svgNode(tag,attributes={},text) {
 const node=document.createElementNS(NS,tag);
 for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,value);
 if(text!==undefined) node.textContent=text;
 return node;
}
function wrap(text,limit=48) {
 const lines=[];
 for(const paragraph of String(text).split("\n")) {
   let rest=paragraph;
   while(rest.length>limit) {
     let cut=rest.lastIndexOf(" ",limit);
     if(cut<limit/3) cut=limit;
     lines.push(rest.slice(0,cut));rest=rest.slice(cut).trimStart();
   }
   lines.push(rest);
 }
 return lines;
}
function svgText(svg,lines,x,y,attributes={}) {
 const text=svgNode("text",{x,y,"font-size":13,...attributes});
 lines.forEach((line,index)=>text.append(svgNode("tspan",{x,dy:index ? 17 : 0},line)));
 svg.append(text);return text;
}
let graphNumber=0;
function flowDiagram(schema,values={},calculated={}) {
 const svg=svgNode("svg",{width:850,role:"img","aria-label":"Suboperaciones, dependencias y condiciones de "+schema.fuente,
   "font-family":"Arial, Helvetica, sans-serif","data-flow-source":schema.fuente});
 const arrowId="flow-arrow-"+(++graphNumber);
 const defs=svgNode("defs"),marker=svgNode("marker",{id:arrowId,markerWidth:7,markerHeight:7,refX:6,refY:3.5,orient:"auto"});
 marker.append(svgNode("path",{d:"M0 0 L7 3.5 L0 7 Z",fill:"black"}));defs.append(marker);svg.append(defs);
 const edges=svgNode("g",{fill:"none",stroke:"black","stroke-width":0.8});svg.append(edges);
 let y=50;
 const positions=new Map();
 const parametersText=Object.entries({...schema.constantes,...values}).map(([k,v])=>k+" = "+(typeof v==="string" ? v : JSON.stringify(v))).join("; ");
 const annotationFor=(node,index)=>{
   const lines=wrap(node.expresion,52);
   if(index===0 && parametersText) lines.push(...wrap("Valores efectivos: "+parametersText,52));
   if(node.tipo==="output" && calculated.umbrales_obtenidos) lines.push(...wrap("Umbrales obtenidos = "+JSON.stringify(calculated.umbrales_obtenidos),52));
   return lines;
 };
 schema.nodos.forEach((node,index)=>{
   const title=wrap(node.titulo,34), expression=annotationFor(node,index);
   const height=Math.max(60,title.length*17+24,expression.length*17+20);
   positions.set(node.id,{x:100,y,width:280,height,center:y+height/2,node});y+=height+42;
 });
 svg.setAttribute("height",y);svg.setAttribute("viewBox",`0 0 850 ${y}`);
 svg.append(svgNode("rect",{x:0,y:0,width:850,height:28,fill:"#ebe0f3",stroke:"black","stroke-width":1}));
 svgText(svg,["Desarrollo de la operación · "+schema.fuente.split(".").at(-1)],425,19,{"text-anchor":"middle","font-weight":"bold"});
 const merged=new Map();
 for(const edge of schema.conexiones) {
   const key=edge.desde+"|"+edge.hasta+"|"+edge.tipo;
   if(!merged.has(key)) merged.set(key,{...edge,labels:[]});
   if(edge.etiqueta) merged.get(key).labels.push(edge.etiqueta);
 }
 let index=0;
 const branchLabels=new Map();
 for(const edge of merged.values()) {
   const from=positions.get(edge.desde),to=positions.get(edge.hasta);
   if(!from || !to) continue;
   let d;
   if(edge.tipo==="data") {
     // Operands enter from the left; control flow enters vertically or from the right.
     const lane=12+(index++%10)*7;
     d=`M${from.x} ${from.center} H${lane} V${to.center} H${to.x}`;
   } else if(to.y>=from.y+from.height && to.y-from.y-from.height<=43) {
     d=`M240 ${from.y+from.height} V${to.y}`;
   } else {
     const lane=395+(index++%4)*8;
     d=`M380 ${from.center} H${lane} V${to.center} H380`;
   }
   const line=svgNode("path",{d,"marker-end":`url(#${arrowId})`,"data-edge-type":edge.tipo});
   line.append(svgNode("title",{},edge.labels.join(", ")||"Secuencia de ejecución"));edges.append(line);
   if(edge.tipo!=="data" && edge.labels.length) {
     if(!branchLabels.has(edge.hasta)) branchLabels.set(edge.hasta,new Set());
     edge.labels.forEach(label=>branchLabels.get(edge.hasta).add(label));
   }
 }
 for(const [target,labels] of branchLabels) {
   const position=positions.get(target);
   svgText(svg,wrap([...labels].join(" / "),42),240,position.y-9,{"font-size":11,"text-anchor":"middle"});
 }
 let annotationIndex=0;
 for(const position of positions.values()) {
   const {node,y,height}=position;
   const group=svgNode("g",{"data-suboperation":node.id});
   group.append(svgNode("rect",{x:100,y,width:280,height,fill:"white",stroke:"black","stroke-width":1}));
   svgText(group,wrap(node.titulo,34),240,y+24,{"text-anchor":"middle"});
   if(node.expresion || position===positions.values().next().value) {
     const annotation=annotationFor(node,annotationIndex);
     svgText(group,annotation,446,y+19,{"data-annotation-for":node.id});
     group.append(svgNode("path",{d:`M438 ${position.center} H382`,stroke:"black",fill:"none",
       "marker-end":`url(#${arrowId})`,"data-edge-type":"annotation"}));
   }
   svg.append(group);
   annotationIndex++;
 }
 return svg;
}
function hierarchy(main,id,schema,values,flows,calculated={}) {
 const selected=main.querySelector(`[data-operation="${id}"]`);
 const box=selected.querySelector("rect"),x=box.getAttribute("x");
 const stage=[...main.querySelectorAll("[data-operation]")].filter(node=>node.querySelector("rect").getAttribute("x")===x);
 const center=Number(x)+Number(box.getAttribute("width"))/2;
 const heading=[...main.querySelectorAll("text")].find(node=>Number(node.getAttribute("x"))===center && node.getAttribute("font-weight")==="bold");
 const stageHeight=stage.length*94+65, visibleHeight=Math.min(780,Math.max(stageHeight,560));
 const frame=el("div",undefined,"step-hierarchy");
 frame.setAttribute("aria-label","Expansión jerárquica de la operación seleccionada");
 const context=svgNode("svg",{width:310,height:Math.max(stageHeight,visibleHeight),role:"img",
   "aria-label":"Etapa seleccionada y líneas de expansión","font-family":"Arial, Helvetica, sans-serif"});
 context.append(svgNode("rect",{x:8,y:0,width:220,height:stageHeight,fill:"white",stroke:"black","stroke-width":1}));
 context.append(svgNode("rect",{x:8,y:0,width:220,height:28,fill:"#ebe0f3",stroke:"black","stroke-width":1}));
 svgText(context,[heading?.textContent||"Etapa seleccionada"],118,20,{"text-anchor":"middle","font-size":15,"font-weight":"bold"});
 stage.forEach((item,index)=>{
   const y=48+index*94,active=item.dataset.operation===id;
   const group=svgNode("g",{"data-stage-operation":item.dataset.operation,"data-selected":String(active)});
   group.append(svgNode("rect",{x:28,y,width:180,height:65,fill:active ? "#ebe0f3" : "white",stroke:"black","stroke-width":1}));
   const lines=[...item.querySelectorAll("tspan")].map(node=>node.textContent);
   svgText(group,lines,118,y+33-(lines.length-1)*8.5,{"text-anchor":"middle","font-size":14,"font-weight":active ? "bold" : "normal"});
   context.append(group);
   if(index<stage.length-1) context.append(svgNode("path",{d:`M118 ${y+65} V${y+89} M114 ${y+83} L118 ${y+89} L122 ${y+83}`,stroke:"black",fill:"none"}));
   if(active) {
     context.append(svgNode("path",{d:`M208 ${y} L310 0`,stroke:"black",fill:"none","data-expansion-line":"upper"}));
     context.append(svgNode("path",{d:`M208 ${y+65} L310 ${visibleHeight}`,stroke:"black",fill:"none","data-expansion-line":"lower"}));
   }
 });
 const detail=el("div",undefined,"step-flow-viewport");detail.style.height=visibleHeight+"px";
 detail.tabIndex=0;detail.setAttribute("aria-label","Diagrama explicativo; desplazar para recorrer las suboperaciones");
 detail.append(flowDiagram(schema,values,calculated));
 // Helpers have distinct, code-derived flows. Opening their documentation never
 // selects a secondary box or invokes the analysis worker.
 for(const key of schema.detalles||[]) {
   const child=flows[key];if(!child) continue;
   const section=el("details",undefined,"step-helper");
   section.append(el("summary","Desarrollo interno · "+child.fuente.split(".").at(-1)));
   section.addEventListener("toggle",()=>{
     if(section.open && !section.querySelector("svg")) section.append(flowDiagram(child));
   });
   detail.append(section);
 }
 frame.append(context,detail);
 const scroll=el("div",undefined,"step-hierarchy-scroll");scroll.tabIndex=0;
 scroll.setAttribute("aria-label","Etapa a la izquierda y desarrollo a la derecha; desplazamiento horizontal disponible");
 scroll.append(frame);return scroll;
}
function histogram(data) {
 const svg=document.createElementNS(NS,"svg");svg.setAttribute("viewBox","0 0 520 180");
 svg.setAttribute("role","img");svg.setAttribute("aria-label","Histograma del canal de entrada y umbral calculado");
 const peak=Math.max(...data.frecuencias,1), n=data.frecuencias.length;
 data.frecuencias.forEach((v,i)=>{
   const r=document.createElementNS(NS,"rect");
   r.setAttribute("x",25+i*470/n);r.setAttribute("y",145-v/peak*125);
   r.setAttribute("width",470/n);r.setAttribute("height",v/peak*125);r.setAttribute("fill","#9c82ad");svg.append(r);
 });
 for(const threshold of data.umbrales) {
 const x=25+470*(threshold-data.bordes[0])/(data.bordes.at(-1)-data.bordes[0]||1);
 const line=document.createElementNS(NS,"path");line.setAttribute("d",`M${x} 12 V150`);line.setAttribute("stroke","#9b2e35");svg.append(line);
 }
 const label=document.createElementNS(NS,"text");label.setAttribute("x","25");label.setAttribute("y","174");
 label.setAttribute("font-size","12");label.textContent="Umbrales calculados = "+data.umbrales.join(", ");svg.append(label);
 return svg;
}
export class Explorer {
 constructor(container,panel) {this.container=container;this.panel=panel;this.version=0;this.clear();}
 clear() {
   this.version++;this.operations=[];this.library={};this.flows={};this.selected=null;this.panel.hidden=true;
   this.panel.replaceChildren();this.container.replaceChildren();this.container.removeAttribute("aria-busy");
 }
 async load(url,operations,library,flows) {
   this.clear();const version=this.version;this.operations=operations||[];this.library=library||{};this.flows=flows||{};
   this.container.setAttribute("aria-busy","true");
   try {
     const response=await fetch(url);
     if(!response.ok) throw new Error("No se pudo cargar el diagrama.");
     const xml=new DOMParser().parseFromString(await response.text(),"image/svg+xml");
     if(version!==this.version) return;
     if(xml.querySelector("parsererror")) throw new Error("El SVG no es válido.");
     const svg=document.importNode(xml.documentElement,true);
     for(const group of svg.querySelectorAll("[data-operation]")) {
       const activate=()=>this.select(group.dataset.operation);
       group.addEventListener("click",activate);
       group.addEventListener("keydown",event=>{
         if(event.key==="Enter" || event.key===" ") {event.preventDefault();activate();}
       });
     }
     this.container.replaceChildren(svg);
   } catch(error) {
     if(version===this.version) this.container.textContent=error.message;
   } finally {if(version===this.version) this.container.removeAttribute("aria-busy");}
 }
 select(id) {
   const operation=this.operations.find(item=>item.id===id);
   if(!operation) return;
   if(this.selected===id) {this.deselect();return;}
   this.selected=id;
   for(const group of this.container.querySelectorAll("[data-operation]")) group.setAttribute("aria-pressed",String(group.dataset.operation===id));
   this.panel.hidden=false;this.panel.replaceChildren();
   const header=el("div",undefined,"step-heading"),title=el("h3",operation.nombre),close=el("button","Cerrar","secondary");
   close.type="button";close.addEventListener("click",()=>this.deselect());header.append(title,close);
   this.panel.append(header);
   if(operation.estado==="no_ejecutada") {
     const schema=this.flows[operation.flujo_ids[0]];
     if(schema) this.panel.append(hierarchy(this.container.querySelector("svg"),this.selected,schema,{},this.flows));
     this.explain(operation,this.panel);
     this.panel.append(el("p","Esta operación no se ejecutó en la ruta de la imagen actual. Sus condiciones no se activaron o la imagen no ofrecía evidencia suficiente.","step-state"));
     return;
   }
   this.panel.append(el("p",`${operation.llamadas.length} llamada(s) registradas en la imagen original. Las propuestas y las operaciones condicionales se identifican en sus parámetros.`,"step-state"));
   const content=el("div");
   if(operation.llamadas.length>1) {
     const label=el("label","Llamada registrada"),select=el("select");
     select.setAttribute("aria-label","Llamada registrada");
     operation.llamadas.forEach((call,index)=>{
       const option=el("option",`${index+1} · ${titles[call.funcion]||call.funcion}${call.parametros.mode ? " · "+call.parametros.mode : ""}`);
       option.value=index;select.append(option);
     });
     select.addEventListener("change",()=>this.call(operation.llamadas[Number(select.value)],content,operation));
     label.append(select);this.panel.append(label);
   }
   this.panel.append(content);this.call(operation.llamadas[0],content,operation);
 }
 deselect() {
   this.selected=null;this.panel.hidden=true;this.panel.replaceChildren();
   for(const group of this.container.querySelectorAll("[data-operation]")) group.setAttribute("aria-pressed","false");
 }
 explain(operation,container,call) {
   container.append(el("p",operation.explicacion,"step-explanation"));
   const equation=equationFor(call);
   if(equation && formulas[equation]) {
     const div=el("div",undefined,"step-equation");
     div.innerHTML='<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'+formulas[equation]+"</math>";
     container.append(div);
   }
 }
 call(call,container,operation) {
   container.replaceChildren(el("p","Función ejecutada · "+call.funcion,"step-source"));
   const schema=this.flows[call.flujo_id];
   if(schema) {
     container.append(hierarchy(this.container.querySelector("svg"),this.selected,schema,call.parametros,this.flows,call.calculados));
     container.append(el("p",schema.alcance,"step-state"));
   }
   this.explain(operation,container,call);
   const config=el("details",undefined,"step-config-values");
   config.append(el("summary","Parámetros efectivos y valores calculados"));
   const annotation=el("div");
   if(Object.keys(call.parametros).length) annotation.append(el("h4","Parámetros efectivos de la llamada"),parameters(call.parametros));
   if(Object.keys(call.calculados).length) {
     const detail=el("details"),summary=el("summary","Valores calculados en esta ejecución");
     detail.append(summary,parameters(call.calculados));annotation.append(detail);
   }
   if(call.fijos.length) {
     const detail=el("details"),summary=el("summary","Expresiones y condiciones exactas del código");
     detail.append(summary,el("pre",call.fijos.join("\n")));annotation.append(detail);
   }
   config.append(annotation);container.append(config);
   const comparison=el("div",undefined,"step-comparison");
   for(const [title,images] of [["Entrada real",call.entrada],["Salida real",call.salida]]) {
     const part=el("div");part.append(el("h4",title));
     part.append(images.length ? pictureGrid(images,this.library) : el("p","La llamada devuelve valores o estructuras sin una imagen bidimensional.","step-state"));
     comparison.append(part);
   }
   container.append(comparison);
   if(call.histograma) container.append(histogram(call.histograma));
   if(call.intermedios.length) {
     const detail=el("details"),summary=el("summary","Canales, marcadores y estados internos registrados");
     detail.append(summary,pictureGrid(call.intermedios,this.library));container.append(detail);
   }
   if(call.propuestas) {
     container.append(el("h4","Propuestas evaluadas"));
     const choices=el("div",undefined,"step-proposals");
     for(const choice of call.propuestas) {
       const part=el("section");
       part.append(el("p",choice.nombre+" · "+choice.estado,choice.estado==="seleccionada" ? "chosen-proposal" : ""));
       part.append(pictureGrid(choice.imagenes,this.library));choices.append(part);
     }
     container.append(choices);
   }
 }
}
