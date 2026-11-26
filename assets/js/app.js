import { Explorer } from "./explorer.js";
const $ = selector => document.querySelector(selector);
const category = $("#category");
const file = $("#image");
const results = $("#results");
const status = $("#status");
const button = $("#analyze");
const diagrams = {
 canine_cutaneous_mast_cell_tumor:"01_mastocitoma",
 canine_lung_cancer:"02_pulmon",
 canine_lymphoma:"03_linfoma",
 canine_soft_tissue_sarcoma:"04_sarcoma",
 human_breast_cancer:"05_mama",
 human_melanoma:"06_melanoma",
 human_neuroendocrine_tumor:"07_neuroendocrino"
};
const explorer = new Explorer($("#diagram"),$("#operation-panel"));
let worker, engineReady = false, running = false, revision = 0, validFile = false;
function fail(message) { $("#error").textContent = message; $("#error").hidden = false; }
function controls() { button.disabled = !engineReady || running || !category.value || !validFile; }
function releaseResults() {
 explorer.clear();
 for(const id of ["original","cell","chromatin"]) $("#"+id).removeAttribute("src");
 $("#measurements").replaceChildren();
 $("#warning-list").replaceChildren();
 $("#method-text").textContent="";
 $("#report-method").textContent="";
}
function invalidate() {
 revision++;
 releaseResults();
 results.hidden = true;
 $("#error").hidden = true;
 $("#input-info").textContent = "";
 controls();
}
function startWorker() {
 worker = new Worker(new URL("./python_worker.js",import.meta.url));
 worker.onmessage = ({data}) => {
   if(data.type === "status") { status.textContent=data.text; status.className="busy"; }
   if(data.type === "ready") { engineReady=true; status.textContent="Motor de análisis listo"; status.className=""; controls(); }
   if(data.type === "result") {
     running=false;
     if(data.id === revision) render(data.result);
     status.textContent="Análisis terminado"; status.className=""; controls();
   }
   if(data.type === "error") {
     running=false;
     status.className="";
     status.textContent = engineReady ? "El motor está listo para otro análisis" : "Motor no disponible. Recarga la página para reintentar.";
     if(data.id === undefined || data.id === revision) fail(data.message);
     controls();
   }
 };
 worker.onerror = () => { running=false;engineReady=false;status.className="";fail("El motor no pudo ejecutarse. Recarga la página y comprueba la conexión.");controls(); };
 worker.postMessage({type:"init"});
}
category.addEventListener("change",invalidate);
file.addEventListener("change",async () => {
 validFile=false;invalidate();
 const request=revision, input=file.files[0];
 if(!input) return;
 if(!input.name.toLowerCase().endsWith(".png") || input.size>20*1024*1024) {fail("Selecciona un archivo PNG de hasta 20 MB."); return;}
 try {
   const bitmap=await createImageBitmap(input);
   const {width,height}=bitmap;bitmap.close();
   if(request!==revision) return;
   if(width!==height) throw new Error("Selecciona un recorte cuadrado. Recorta la imagen rectangular antes de analizarla.");
   if(width<8 || width>4096) throw new Error("El lado de la imagen debe estar entre 8 y 4096 píxeles.");
   $("#input-info").textContent = width===128 ? "Imagen de 128 × 128 px." : `Entrada de ${width} × ${height} px. Se analizará en 128 × 128 px y las mediciones corresponderán a esa malla.`;
   validFile=true;controls();
 } catch(error) {if(request===revision) fail(error.message || "No se pudo abrir el PNG.");}
});
$("#analysis-form").addEventListener("submit",async event => {
 event.preventDefault();
 if(button.disabled || running) return;
 const request=revision, input=file.files[0], tumor=category.value;
 running=true;controls();$("#error").hidden=true;results.hidden=true;releaseResults();
 status.textContent="Analizando la imagen y comprobando la sensibilidad…";status.className="busy";
 try {
   const bytes=await input.arrayBuffer();
   if(request!==revision) {running=false;controls();return;}
   worker.postMessage({type:"analyze",id:request,category:tumor,name:input.name,bytes},[bytes]);
 } catch(error) {running=false;fail("No se pudo leer el archivo.");controls();}
});
function render(result) {
 const tumor=category.options[category.selectedIndex].text;
 $("#filename").textContent=result.nombre;
 $("#tumor").textContent=tumor;
 $("#resolution").textContent=`128 × 128 px · entrada de ${result.resolucion_original.join(" × ")} px`;
 $("#method-text").textContent=result.metodo;
 $("#report-method").textContent=result.metodo;
 for(const [index,id] of ["original","cell","chromatin"].entries()) $("#"+id).src=result.imagenes[index];
 $("#measurements").replaceChildren(...result.caracteristicas.map(item=>{
    const row=document.createElement("tr"),label=document.createElement("th"),value=document.createElement("td");
    label.scope="row";label.textContent=item.nombre;
    value.textContent=item.valor===null ? "No disponible" : new Intl.NumberFormat("es-CO",{maximumFractionDigits:3}).format(item.valor)+(item.unidad ? " "+item.unidad : "");
    row.append(label,value);return row;
 }));
 const path="assets/diagrams/"+diagrams[result.categoria]+".svg";
 explorer.load(path,result.etapas,result.imagenes_etapas);
 $("#diagram-download").href=path;
 $("#warning-list").replaceChildren(...result.advertencias.map(text=>{const li=document.createElement("li");li.textContent=text;return li;}));
 $("#warnings").hidden=result.advertencias.length===0;
 results.hidden=false;
}
let reportWasOpen=true;
function beforePrint(){reportWasOpen=$("#report").open;$("#report").open=true;}
function afterPrint(){$("#report").open=reportWasOpen;}
window.addEventListener("beforeprint",beforePrint);window.addEventListener("afterprint",afterPrint);
$("#print").addEventListener("click",()=>window.print());
startWorker();
