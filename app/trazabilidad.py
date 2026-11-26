"""Observación opcional de llamadas reales, limitada a la primera ruta de segmentación."""
from contextlib import AbstractContextManager
from functools import wraps
from importlib import import_module
import ast
import hashlib
import inspect
import sys
import textwrap
import cv2
import numpy as np
from .operaciones import BOXES, OPERATIONS, AUXILIARY, CV_ARGUMENTS

_IMAGE_KEYS = ("rgb","filtered","smooth","cell","chromatin","candidate","excluded","omitted",
               "labels","markers","seeds","h","e","ratio","edge","edge4","edge6","outer","gabor",
               "texture","fibers","probability","l","a","b","od","source","exterior","target","other")
_NAMES = {"cell":"Región celular","chromatin":"Cromatina","excluded":"Excluidos","omitted":"Ambiguos",
          "filtered":"Imagen bilateral","smooth":"Imagen suavizada","rgb":"RGB",
          "h":"Hematoxilina","e":"Eosina","ratio":"Blue Ratio","labels":"Parcelas",
          "markers":"Marcadores","seeds":"Semillas","candidate":"Candidatos","probability":"Pertenencia",
          "edge":"Fronteras","edge4":"Fronteras multicanal","edge6":"Fronteras de apariencia"}


def _small(value):
    if isinstance(value,np.generic):
        value=value.item()
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value,dict) and len(value)<=64:
        children={key:_small(item) for key,item in value.items()}
        if all(item is not ... for item in children.values()):
            return children
    if isinstance(value,(tuple,list)) and len(value)<=24:
        children=[_small(item) for item in value]
        if all(item is not ... for item in children):
            return children
    return ...


def _arrays(value, prefix="", limit=16):
    """Copy bounded visual inputs now, before in-place algorithms can mutate them."""
    found=[]
    def visit(item,label,depth):
        if len(found)>=limit or depth>4:
            return
        if isinstance(item,np.ndarray):
            if item.ndim in (2,3) and item.shape[0]>=3 and item.shape[1]>=3:
                found.append((label,item.copy()))
        elif isinstance(item,dict):
            keys=[key for key in _IMAGE_KEYS if key in item]
            keys += [key for key in item if key not in keys and key not in ("previous","old_anchor")]
            for key in keys:
                visit(item[key],f"{label} · {key}" if label else key,depth+1)
        elif isinstance(item,(list,tuple)):
            for index,child in enumerate(item):
                visit(child,f"{label} · {index+1}",depth+1)
    visit(value,prefix,0)
    return found


def _fixed(function):
    try:
        source=textwrap.dedent(inspect.getsource(function))
        tree=ast.parse(source)
    except (OSError,TypeError,IndentationError):
        return []
    expressions=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.If,ast.Return,ast.Call)):
            if any(isinstance(child,ast.Constant) and isinstance(child.value,(int,float))
                   and not isinstance(child.value,bool) for child in ast.walk(node)):
                text=ast.unparse(node.test if isinstance(node,ast.If) else node)
                if "\n" not in text and len(text)<=260 and text not in expressions:
                    expressions.append(text)
    return expressions


class RegistroEtapas(AbstractContextManager):
    def __init__(self,rgb,categoria):
        self.rgb=rgb
        from pipelines_histologia import CATEGORIAS
        self.category=CATEGORIAS[categoria][1:]
        self.events=[]
        self.stack=[]
        self.active=False
        self.done=False
        self.saving=False
        self.restores=[]
        self.images={}
        self.serialized={}

    def __enter__(self):
        import pipelines_histologia
        for module in pipelines_histologia.CATEGORIAS.values():
            import_module("pipelines_histologia."+module)
        names=set(AUXILIARY["python"])
        for _,functions in BOXES[self.category]:
            names.update(functions)
        # All imported aliases must refer to the same observer wrapper.
        modules=[module for name,module in list(sys.modules.items())
                 if name.startswith("pipelines_histologia") and module is not None]
        wrappers={}
        for module in modules:
            for name,function in list(vars(module).items()):
                if inspect.isfunction(function) and function.__name__ in names:
                    wrapper=wrappers.setdefault(function,self._wrap(function))
                    self.restores.append((module,name,function))
                    setattr(module,name,wrapper)
        for name in AUXILIARY["cv"]:
            function=getattr(cv2,name)
            self.restores.append((cv2,name,function))
            setattr(cv2,name,self._wrap(function,name))
        from skimage import filters
        for name in ("threshold_multiotsu", "threshold_otsu"):
            function = getattr(filters, name)
            self.restores.append((filters, name, function))
            setattr(filters, name, self._wrap(function))
        return self

    def __exit__(self,*exc):
        for module,name,original in reversed(self.restores):
            setattr(module,name,original)
        self.restores.clear()
        self.active=False
        return False

    def _wrap(self,function,cv_name=None):
        name=cv_name or function.__name__
        fixed=_fixed(function) if not cv_name else []
        @wraps(function)
        def observed(*args,**kwargs):
            is_root=name=="segmentar_recorte"
            if is_root and not self.done:
                self.active=True
            recording=self.active and not self.saving
            if not recording:
                return function(*args,**kwargs)
            self.saving=True
            try:
                if cv_name:
                    arguments=dict(zip(CV_ARGUMENTS[name],args))
                    arguments.update(kwargs)
                else:
                    bound=inspect.signature(function).bind(*args,**kwargs)
                    bound.apply_defaults()
                    arguments=dict(bound.arguments)
                event={"funcion":name,"entrada":_arrays(arguments),"parametros":{},
                       "calculados":{},"fijos":fixed,"hijas":[]}
                for key,value in arguments.items():
                    scalar=_small(value)
                    if scalar is not ...:
                        event["parametros"][key]=scalar
                code=getattr(function,"__code__",None)
                for key in ("CONFIG","SETTINGS"):
                    if code is not None and key in code.co_names:
                        value=_small(function.__globals__[key])
                        if value is not ...:
                            event["parametros"][key]=value
                if self.stack:
                    self.stack[-1]["hijas"].append(name)
                self.stack.append(event)
            finally:
                self.saving=False
            # The profiler only observes this exact frame to retain computed scalar locals.
            old_profile=sys.getprofile()
            computed={}
            def profile(frame,kind,value):
                if kind=="return" and frame.f_code is getattr(function,"__code__",None):
                    computed.update(frame.f_locals)
            if not cv_name:
                sys.setprofile(profile)
            try:
                result=function(*args,**kwargs)
            finally:
                if not cv_name:
                    sys.setprofile(old_profile)
                if is_root:
                    self.active=False
                    self.done=True
            self.saving=True
            try:
                self.stack.pop()
                event["salida"]=_arrays(result)
                if name=="grabCut":
                    event["salida"]=_arrays({"markers":result[0]})
                elif name=="watershed":
                    event["salida"]=_arrays({"markers":result})
                if isinstance(result,dict):
                    for key in ("rescued","changed","retained","reference_retained"):
                        if key in result:
                            event["calculados"][key]=bool(result[key])
                if name=="rescatar_mastocitoma":
                    event["calculados"]["rescate_activado"]=bool(result.get("rescued",False))
                if name in ("select_body","support"):
                    from pipelines_histologia.seleccion import inspect_record
                    from pipelines_histologia.calidad import body_evidence
                    event["fijos"]=event["fijos"]+_fixed(inspect_record)+_fixed(body_evidence)
                for key,value in computed.items():
                    if key in arguments:
                        continue
                    scalar=_small(value)
                    if scalar is not ...:
                        event["calculados"][key]=scalar
                # Capture actual tuples and record dictionaries using semantic local names.
                useful={key:value for key,value in computed.items()
                        if key in _IMAGE_KEYS or key in ("strong","weak","initial","domain","raw",
                        "cell","chrom","candidate","selected","accepted","high","low","response","score",
                        "owner","barrier","terrain","votes","support","parts","groups","pending","neighbors")}
                event["intermedios"]=_arrays(useful)
                if name in ("threshold","threshold_otsu","threshold_multiotsu"):
                    levels=np.asarray(result[0] if name=="threshold" else result).ravel()
                    event["calculados"]["umbrales_obtenidos"]=levels.tolist()
                    src=arguments.get("src",arguments.get("image"))
                    if src is not None:
                        hist,edges=np.histogram(src[np.isfinite(src)],bins=64)
                        event["histograma"]={"frecuencias":hist.tolist(),"bordes":edges.tolist(),"umbrales":levels.tolist()}
                    if name=="threshold":
                        flag=arguments["type"]
                        event["parametros"]["criterio"]=("Otsu" if flag & cv2.THRESH_OTSU else
                            "Triángulo" if flag & cv2.THRESH_TRIANGLE else "Umbral fijo")
                if name in ("select_body","select_chromatin"):
                    event["propuestas"]=self._proposals(arguments["records"],result)
                if is_root:
                    if "choices" in computed and isinstance(computed["choices"],list):
                        event["propuestas"]=self._proposals([item[0] for item in computed["choices"]],result[0])
                event["orden"]=len(self.events)
                self.events.append(event)
            finally:
                self.saving=False
            return result
        return observed

    def _proposals(self,records,selected):
        return [{"nombre":record.get("method",f"Propuesta {i+1}"),
                 "estado":"seleccionada" if record is selected else "descartada",
                 "imagenes":_arrays({key:record[key] for key in ("cell","chromatin") if key in record},limit=2)}
                for i,record in enumerate(records)]

    def exportar(self,diagram):
        import xml.etree.ElementTree as ET
        tree=ET.fromstring(diagram)
        operation_nodes=[node for node in tree.iter() if node.get("data-operation")]
        definitions=BOXES[self.category]
        if len(operation_nodes)!=len(definitions):
            raise RuntimeError("El inventario de operaciones y el diagrama no coinciden.")
        operations=[]
        self.saving=True
        try:
            for node,(kind,names) in zip(operation_nodes,definitions):
                name=node.get("data-title")
                calls=[self._serialize(event) for event in self.events if event["funcion"] in names]
                paragraph,equation=OPERATIONS[kind]
                operations.append({"id":node.get("data-operation"),"nombre":name,"explicacion":paragraph,
                                   "ecuacion":equation,"estado":"ejecutada" if calls else "no_ejecutada",
                                   "llamadas":calls,"funciones":names})
            return operations
        finally:
            self.saving=False

    def _serialize(self,event):
        if id(event) in self.serialized:
            return self.serialized[id(event)]
        out={key:value for key,value in event.items()
             if key not in ("entrada","salida","intermedios","propuestas")}
        out["entrada"]=self._pictures(event["entrada"])
        out["salida"]=self._pictures(event["salida"])
        out["intermedios"]=self._pictures(event["intermedios"])
        if event.get("propuestas"):
            out["propuestas"]=[dict(nombre=item["nombre"],estado=item["estado"],
                                    imagenes=self._pictures(item["imagenes"]))
                              for item in event["propuestas"]]
        self.serialized[id(event)]=out
        return out

    def _pictures(self,arrays):
        from .servicio import _png
        pictures=[]
        for label,array in arrays:
            if array.shape[:2] != self.rgb.shape[:2]:
                # Actual kernels are shown at their own dimensions.
                rgb=None
            else:
                rgb=self.rgb
            channels=[(label,array)]
            if array.ndim==3 and (array.shape[-1]!=3 or array.dtype!=np.uint8):
                channels=[(f"{label} · canal {i+1}",array[...,i]) for i in range(array.shape[-1])]
            for title,values in channels:
                if values.ndim==3 and values.shape[-1]==3 and values.dtype==np.uint8:
                    image=values
                elif values.dtype==bool and rgb is not None:
                    tint=(255,220,0) if any(key in title for key in ("chrom","crom")) else (0,220,240)
                    if any(key in title for key in ("exclu","rival","neighbor")):
                        tint=(235,50,50)
                    image=rgb.astype(float).copy()
                    image[values]=.55*image[values]+.45*np.array(tint)
                    image=np.rint(image).astype(np.uint8)
                elif np.issubdtype(values.dtype,np.integer) and any(key in title for key in ("labels","marker","seed","owner","territor","cluster","mask")):
                    ids=values.astype(np.int64)
                    image=np.stack(((ids*67+43)%251,(ids*131+89)%251,(ids*193+17)%251),axis=-1).astype(np.uint8)
                    image[ids<=0]=255
                    if rgb is not None:
                        image=(.55*rgb+.45*image).astype(np.uint8)
                else:
                    pixels=np.nan_to_num(values.astype(float),nan=0,posinf=0,neginf=0)
                    low,high=float(pixels.min()),float(pixels.max())
                    gray=np.rint(255*(pixels-low)/max(high-low,1e-12)).astype(np.uint8)
                    image=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                key=hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()
                if key not in self.images:
                    self.images[key]=_png(image)
                display=title
                for raw,spanish in _NAMES.items():
                    parts=display.split(" · ")
                    display=" · ".join(spanish if part==raw else part for part in parts)
                pictures.append({"nombre":display,"imagen_id":key,
                                 "dimensiones":list(values.shape),"tipo":str(values.dtype)})
        return pictures
