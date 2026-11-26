const NS = "http://www.w3.org/2000/svg";
const titles = {
 bilateralFilter:"Filtro bilateral",cvtColor:"Transformación de color",threshold:"Umbralización",
 grabCut:"GrabCut",watershed:"Watershed",getGaborKernel:"Núcleo de Gabor",filter2D:"Convolución",
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
function secondary(names) {
 const unique=[...new Set(names)];
 if(unique.length<2) return null;
 const height=unique.length*76+20;
 const svg=document.createElementNS(NS,"svg");
 svg.setAttribute("viewBox",`0 0 430 ${height}`);
 svg.setAttribute("role","img");svg.setAttribute("aria-label","Secuencia de suboperaciones observadas");
 unique.forEach((name,index)=>{
   const y=10+index*76, box=document.createElementNS(NS,"rect"),text=document.createElementNS(NS,"text");
   box.setAttribute("x","12");box.setAttribute("y",y);box.setAttribute("width","406");box.setAttribute("height","48");
   box.setAttribute("fill","white");box.setAttribute("stroke","black");
   text.setAttribute("x","215");text.setAttribute("y",y+29);text.setAttribute("text-anchor","middle");
   text.setAttribute("font-family","Arial");text.setAttribute("font-size","13");
   text.textContent=titles[name]||name;
   svg.append(box,text);
   if(index<unique.length-1) {
     const arrow=document.createElementNS(NS,"path");
     arrow.setAttribute("d",`M215 ${y+48} V${y+72} M211 ${y+66} L215 ${y+72} L219 ${y+66}`);
     arrow.setAttribute("stroke","black");arrow.setAttribute("fill","none");svg.append(arrow);
   }
 });
 return svg;
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
   this.version++;this.operations=[];this.library={};this.selected=null;this.panel.hidden=true;
   this.panel.replaceChildren();this.container.replaceChildren();this.container.removeAttribute("aria-busy");
 }
 async load(url,operations,library) {
   this.clear();const version=this.version;this.operations=operations||[];this.library=library||{};
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
   this.panel.append(header,el("p",operation.explicacion));
   if(operation.ecuacion && formulas[operation.ecuacion]) {
     const div=el("div",undefined,"step-equation");
     div.innerHTML='<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'+formulas[operation.ecuacion]+"</math>";
     this.panel.append(div);
   }
   if(operation.estado==="no_ejecutada") {
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
     select.addEventListener("change",()=>this.call(operation.llamadas[Number(select.value)],content));
     label.append(select);this.panel.append(label);
   }
   this.panel.append(content);this.call(operation.llamadas[0],content);
 }
 deselect() {
   this.selected=null;this.panel.hidden=true;this.panel.replaceChildren();
   for(const group of this.container.querySelectorAll("[data-operation]")) group.setAttribute("aria-pressed","false");
 }
 call(call,container) {
   container.replaceChildren(el("p","Función ejecutada · "+call.funcion,"step-source"));
   const config=el("div",undefined,"step-config");
   let diagram=secondary(call.hijas);
   if(!diagram && call.propuestas) {
     const steps=call.funcion==="select_chromatin"
       ? ["Propuestas cromatínicas registradas","Evidencia de tinción por propuesta","Comparación con la ganancia requerida","Propuesta seleccionada"]
       : ["Parejas de máscaras registradas","Indicadores de consistencia y contorno","Ordenación por los criterios del código","Propuesta seleccionada"];
     diagram=secondary(steps);
   }
   if(diagram) {
     const figure=el("figure",undefined,"step-secondary");
     figure.append(diagram,el("figcaption",call.hijas.length>1
       ? "Suboperaciones observadas, en orden de primera llamada"
       : "Secuencia de selección de las propuestas registradas"));config.append(figure);
   }
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
