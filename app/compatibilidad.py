import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.linalg import spsolve
from skimage.graph import MCP_Geometric
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour


def verificar_runtime():
    rgb = np.full((24,24,3),180,np.uint8)
    rgb[7:17,7:17] = [72,26,105]
    mask = np.full((24,24),cv2.GC_PR_BGD,np.uint8)
    mask[8:16,8:16] = cv2.GC_FGD
    mask[:2] = cv2.GC_BGD
    cv2.grabCut(rgb,mask,None,np.zeros((1,65)),np.zeros((1,65)),1,cv2.GC_INIT_WITH_MASK)
    markers = np.zeros((24,24),np.int32)
    markers[0,:] = 1
    markers[10,10] = 2
    cv2.watershed(rgb,markers)
    binary = (mask == cv2.GC_FGD).astype(np.uint8)
    cv2.connectedComponentsWithStats(binary,connectivity=8)
    cv2.distanceTransform(binary,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    capacity = sparse.csr_matrix(np.array([[0,3,0],[0,0,2],[0,0,0]],dtype=np.int64))
    assert maximum_flow(capacity,0,2).flow_value == 2
    assert np.allclose(spsolve(sparse.eye(2,format="csc"),[1.,2.]),[1.,2.])
    MCP_Geometric(np.ones((5,5))).find_costs([(2,2)])
    field = np.linspace(0,1,576).reshape(24,24)
    morphological_chan_vese(field,2,init_level_set=binary)
    morphological_geodesic_active_contour(field,2,init_level_set=binary)
    cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    np.linalg.inv(np.eye(3))
    em = cv2.ml.EM_create()
    em.setClustersNumber(2)
    em.trainEM(np.array([[i,i%3] for i in range(20)],np.float64))
    from importlib import import_module
    from pipelines_histologia import CATEGORIAS
    for module in CATEGORIAS.values():
        import_module("pipelines_histologia."+module)
    import scipy, skimage
    return dict(opencv=cv2.__version__,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__)
