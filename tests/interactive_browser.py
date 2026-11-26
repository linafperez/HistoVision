import pathlib,json,sys,argparse,tempfile,re
import numpy as np
from playwright.sync_api import sync_playwright
def numeric_equal(a,b,path=""):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for key in a:numeric_equal(a[key],b[key],path+"."+key)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(first,last) in enumerate(zip(a,b)):numeric_equal(first,last,path+str(i))
 elif isinstance(a,(float,int)) and not isinstance(a,bool):
  assert np.isclose(a,b,rtol=1e-6,atol=1e-6),(path,a,b)
 else:assert a==b,(path,a,b)

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.servicio import analizar
from threadpoolctl import threadpool_limits
import cv2
cv2.setNumThreads(1)
parser=argparse.ArgumentParser()
parser.add_argument("--browser",help="Ejecutable opcional de Chrome o Edge")
parser.add_argument("--url",default="http://localhost:8000/")
parser.add_argument("--data",type=pathlib.Path,default=ROOT/"data")
parser.add_argument("--output",type=pathlib.Path,default=pathlib.Path(tempfile.mkdtemp(prefix="histovision-browser-")))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p, threadpool_limits(limits=1):
 browser=p.chromium.launch(executable_path=args.browser,headless=True)
 page=browser.new_page(viewport={'width':1440,'height':1000})
 errors=[]
 page.on('pageerror',lambda error:errors.append(str(error)))
 page.add_init_script('''
window.receivedResults=[];window.analysisCount=0;
const NativeWorker=window.Worker;
window.Worker=class extends NativeWorker {
 constructor(...args) {
  super(...args);this.addEventListener('message',event=>{
   if(event.data.type==='result') window.receivedResults.push(event.data.result);
  });
 }
 postMessage(data,...args){if(data.type==='analyze')window.analysisCount++;return super.postMessage(data,...args);}
};
''')
 page.goto(args.url)
 page.wait_for_function('document.querySelector("#status").textContent.includes("listo")',timeout=240000)
 report=[]
 for folder in sorted(args.data.iterdir()):
  if not folder.is_dir():continue
  sample=next(folder.glob('*.png'))
  page.select_option('#category',folder.name)
  page.set_input_files('#image',str(sample))
  page.wait_for_function('!document.querySelector("#analyze").disabled')
  page.click('#analyze')
  page.wait_for_function('!document.querySelector("#results").hidden || !document.querySelector("#error").hidden',timeout=300000)
  assert page.locator('#error').is_hidden(),page.locator('#error').inner_text()
  local=analizar(sample.read_bytes(),folder.name,sample.name,registrar_etapas=False)
  assert page.evaluate('window.receivedResults.at(-1).huellas_mascaras')==local['huellas_mascaras']
  numeric_equal(page.evaluate('window.receivedResults.at(-1).caracteristicas'),local['caracteristicas'])
  assert page.evaluate('window.receivedResults.at(-1).metodo')==local['metodo']
  wasm_indicators=page.evaluate('window.receivedResults.at(-1).indicadores')
  numeric_equal(wasm_indicators,local['indicadores'])
  print(folder.name,'análisis real terminado',flush=True)
  # Compact result metadata stays in this test; user images are never saved by the app.
  page.evaluate('window.receivedResults=window.receivedResults.slice(-1)')
  page.click('#method summary')
  page.wait_for_selector('#diagram svg')
  count=page.locator('#diagram [data-operation]').count()
  sent=page.evaluate('window.analysisCount')
  catalog=json.loads((ROOT/'assets/methods/catalog.json').read_text(encoding='utf8'))
  total_calls=0
  for i in range(count):
   group=page.locator('#diagram [data-operation]').nth(i)
   group.click()
   assert group.get_attribute('aria-pressed')=='true'
   assert page.locator('#diagram [aria-pressed=true]').count()==1
   assert page.locator('#operation-panel').is_visible()
   step=page.evaluate('window.receivedResults.at(-1).etapas['+str(i)+']')
   assert page.locator('#operation-panel h3').inner_text()==step['nombre']
   page.wait_for_selector('#operation-panel .method-figure img[data-ready=true]')
   figure=page.evaluate("""async()=>{
    const image=document.querySelector('.method-figure img');
    const svg=new DOMParser().parseFromString(await (await fetch(image.src)).text(),'image/svg+xml');
    return {blocks:svg.querySelectorAll('[data-method-block]').length,
     selected:svg.querySelector('[data-selected=true]').getAttribute('data-stage-operation'),
     notes:svg.querySelectorAll('[data-parameter-slot]').length,
     buttons:svg.querySelectorAll('[role=button]').length};
   }""")
   assert 2<=figure['blocks']<=6 and figure['notes']<=2 and figure['buttons']==0
   assert figure['selected']==step['id']
   assert page.locator('#operation-panel .step-helper,#operation-panel pre,#operation-panel math').count()==0
   assert page.locator('#operation-panel .step-explanation').count()==1
   assert page.locator('#operation-panel .step-equation img').count()==int('formula' in catalog[step['id']])
   page.wait_for_function('[...document.querySelectorAll(".method-figure img,.step-equation img")].every(n=>n.complete&&n.naturalWidth>0)')
   assert not re.search(r'\b(previous|prepared|return|pair_evidence|contour5|details|mode)\b',page.locator('#operation-panel').inner_text())
   if step['estado']=='no_ejecutada':
    assert page.locator('#operation-panel .step-images img').count()==0
   else:
    total_calls+=len(step['llamadas'])
    for index,call in enumerate(step['llamadas']):
     selector=page.locator('#operation-panel select')
     if selector.count():selector.select_option(str(index))
     page.wait_for_selector('#operation-panel .method-figure img[data-ready=true]')
     page.wait_for_function('[...document.querySelectorAll(".method-figure img,.step-equation img")].every(n=>n.complete&&n.naturalWidth>0)')
     for detail in page.locator('#operation-panel details').all():detail.evaluate('(node)=>node.open=true')
     image_refs=[pic for field in ('entrada','salida','intermedios') for pic in call[field]]
     ids=[ref['imagen_id'] for ref in image_refs]
     assert page.evaluate('''keys => {
       const sources=[...document.querySelectorAll('#operation-panel img')].map(node=>node.src);
       const images=window.receivedResults.at(-1).imagenes_etapas;
       return keys.every(key=>sources.includes(images[key]));
     }''',ids)
     assert page.locator('#operation-panel .step-parameters').count()==0
   assert page.evaluate('window.analysisCount')==sent
   print(step['id'],len(step['llamadas']),'llamadas comprobadas',flush=True)
  if folder.name=='canine_cutaneous_mast_cell_tumor':
   page.locator('#diagram [data-operation="mastocitoma-06"]').click()
   page.wait_for_selector('#operation-panel .method-figure img[data-ready=true]')
   page.locator('#operation-panel').screenshot(path=str(args.output/'academic-reconstruction.png'))
  if folder.name=='canine_lung_cancer':
   page.locator('#diagram [data-operation]').first.click()
   page.locator('#diagram [data-operation]').first.click()
   assert page.locator('#operation-panel').is_hidden()
   page.locator('#diagram [data-operation]').first.press('Enter')
   page.screenshot(path=str(args.output/'interactive.png'),full_page=True)
   page.set_viewport_size({'width':390,'height':844})
   page.screenshot(path=str(args.output/'interactive-mobile.png'),full_page=True)
   page.set_viewport_size({'width':1440,'height':1000})
  page.locator('#operation-panel .step-heading button').click()
  assert page.locator('#operation-panel').is_hidden()
  assert page.locator('#diagram [aria-pressed=true]').count()==0
  page.select_option('#category','')
  assert page.locator('#operation-panel img').count()==0
  assert page.locator('#diagram svg').count()==0
  assert page.locator('#original').get_attribute('src') is None
  page.locator('#method').evaluate('(node)=>node.open=false')
  report.append({'category':folder.name,'boxes':count,'calls':total_calls,'unchanged':True})
  print(report[-1],flush=True)
 assert not errors,errors
 (args.output/'interactive-test.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 browser.close()
