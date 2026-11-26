const BASE = new URL("../../", self.location.href);
const RUNTIME = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let ready;
let busy = false;
async function initialize() {
  postMessage({type:"status", text:"Cargando Python y las bibliotecas científicas…"});
  importScripts(RUNTIME + "pyodide.js");
  const py = await loadPyodide({indexURL:RUNTIME});
  await py.loadPackage(["opencv-python","numpy","scipy","scikit-image","pillow","pandas"]);
  const response = await fetch(new URL("python-files.json", BASE));
  if (!response.ok) throw new Error("No se encontró el inventario de módulos.");
  const files = await response.json();
  files.push(...["01_mastocitoma","02_pulmon","03_linfoma","04_sarcoma","05_mama","06_melanoma","07_neuroendocrino"].map(name=>"assets/diagrams/"+name+".svg"));
  await Promise.all(files.map(async path => {
    const source = await fetch(new URL(path,BASE));
    if (!source.ok) throw new Error("No se pudo cargar " + path);
    const target = "/home/pyodide/" + path;
    py.FS.mkdirTree(target.substring(0,target.lastIndexOf("/")));
    py.FS.writeFile(target,await source.text());
  }));
  const versions = await py.runPythonAsync(`
import json
from app.compatibilidad import verificar_runtime
json.dumps(verificar_runtime())
`);
  postMessage({type:"ready", versions:JSON.parse(versions)});
  return py;
}
self.onmessage = async ({data}) => {
  if (data.type === "init") {
    ready ??= initialize();
    try { await ready; } catch(error) { ready = null; postMessage({type:"error", message:"No se pudo iniciar el motor. " + error.message}); }
    return;
  }
  if (data.type !== "analyze" || busy) return;
  busy = true;
  try {
    ready ??= initialize();
    const py = await ready;
    py.globals.set("upload_bytes",new Uint8Array(data.bytes));
    py.globals.set("upload_category",data.category);
    py.globals.set("upload_name",data.name);
    const text = await py.runPythonAsync(`
from app.servicio import analizar
import json
json.dumps(analizar(bytes(upload_bytes.to_py()), upload_category, upload_name), ensure_ascii=False, allow_nan=False)
`);
    postMessage({type:"result",id:data.id,result:JSON.parse(text)});
  } catch(error) {
    postMessage({type:"error",id:data.id,message:"No se pudo analizar la imagen. " + error.message});
  } finally {
    if (ready) {
      const py = await ready.catch(()=>null);
      if (py) for (const key of ["upload_bytes","upload_category","upload_name"]) py.globals.delete(key);
    }
    busy = false;
  }
};

