import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.stats as st
from scipy import signal
from sympy import false


#######################################################################################
###################################### CONSTANTS ######################################
#######################################################################################


NUM_GRAYS=256
MIN_GRAY=0
MAX_GRAY=255


#######################################################################################
###################################### PLOTTING #######################################
#######################################################################################

def myFancyPlot(img, title=None, cmap=None, grid=(0,0), ticks=(0,0), saveto='', newFig = True):
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    planes = np.shape(img)[2] if len(np.shape(img)) > 2 else 1

    if newFig:
        plt.figure()
    
    if title == "auto":
        title = "{0} {1} x {2} px, {3} {4}".format(saveto, height, width, planes, "planes" if planes > 1 else "plane")
  
    plt.title(title)

    plt.imshow(img, cmap=cmap, vmin=0, vmax=1 if img.dtype == 'float16' or img.dtype == 'float32' or img.dtype == 'float64' else 255)

    ax = plt.gca()

    if ticks != (0,0):
        ax.set_yticks(np.arange(0, height, ticks[0]))
        ax.set_yticklabels(np.arange(0, height, ticks[0]))
        ax.set_xticks(np.arange(0, width, ticks[1]))
        ax.set_xticklabels(np.arange(0, width, ticks[1]))

    if grid != (0,0):
        ax.set_yticks(np.arange(-.5, height, grid[0]), minor=True)
        ax.set_xticks(np.arange(-.5, width, grid[1]), minor=True)
        plt.grid(visible=True, which="minor", axis="both")

    if saveto != '':
        plt.savefig(saveto, dpi=600, facecolor="white")





#######################################################################################
####################################### PADDING #######################################
#######################################################################################

def circularPadding(f,m):
    if not np.shape(m)[0] % 2 or not np.shape(m)[1] % 2:
        raise Exception("Mask must have odd sizes")
    a = (np.shape(m)[0] - 1) // 2
    b = (np.shape(m)[1] - 1) // 2

    fp = np.pad(f, ((a,a),(b,b),(0,0)) if len(f.shape) == 3 else ((a,a),(b,b)), mode='wrap')
    return fp

def mirrorPadding(f,m):
    if not np.shape(m)[0] % 2 or not np.shape(m)[1] % 2:
        raise Exception("Mask must have odd sizes")
        
    a = (np.shape(m)[0] - 1) // 2
    b = (np.shape(m)[1] - 1) // 2

    fp = np.pad(f, ((a,a),(b,b),(0,0)) if len(f.shape) == 3 else ((a,a),(b,b)), 'symmetric')
    return fp

def replicatePadding(f,m):
    if not np.shape(m)[0] % 2 or not np.shape(m)[1] % 2:
        raise Exception("Mask must have odd sizes")
        
    a = (np.shape(m)[0] - 1) // 2
    b = (np.shape(m)[1] - 1) // 2

    fp = np.pad(f, ((a,a),(b,b),(0,0)) if len(f.shape) == 3 else ((a,a),(b,b)), 'edge')
    return fp

def zeroPadding(f,m):
    if not np.shape(m)[0] % 2 or not np.shape(m)[1] % 2:
        raise Exception("Mask must have odd sizes")
    a = (np.shape(m)[0] - 1) // 2
    b = (np.shape(m)[1] - 1) // 2
    
    if len(np.shape(f)) == 3:   # not a grayscale image
        fp = np.zeros((np.shape(f)[0] + 2 * a, np.shape(f)[1] + 2 * b, np.shape(f)[2]), dtype = f.dtype)
    else:   # grayscale image
        fp = np.zeros((np.shape(f)[0] + 2 * a, np.shape(f)[1] + 2 * b), dtype = f.dtype)

    fp[a:-a,b:-b] = f
    if len(f.shape) == 3 and f.shape[2] == 4:    # Image with alpha -> setting alpha to 1
        fp[0:a,:,3] = 255 if f.dtype == 'uint8' else 1.0
        fp[-a:,:,3] = 255 if f.dtype == 'uint8' else 1.0
        fp[:,0:b,3] = 255 if f.dtype == 'uint8' else 1.0
        fp[:,-b:,3] = 255 if f.dtype == 'uint8' else 1.0
    return fp


def imagePadding(_f, _mask, _type='mirror'):
    if _type == "replicate":
        fp = replicatePadding(_f, _mask)
    elif _type == "mirror":
        fp = mirrorPadding(_f, _mask)
    elif _type == "zero":
        fp = zeroPadding(_f, _mask)
    elif _type == "circular":
        fp = circularPadding(_f, _mask)
    else:
        raise Exception("Unknown padding type")
    return fp


def imageUnpadding(f,mask):
    if not np.shape(mask)[0] % 2 or not np.shape(mask)[1] % 2:
        raise Exception("Mask must have odd sizes")
    b = (np.shape(mask)[0] - 1) // 2
    a = (np.shape(mask)[1] - 1) // 2

    return f[a:-a, b:-b]


#######################################################################################
#################################### Normalization ####################################
#######################################################################################


def imageNormalize(f, min = 0, max = 255):

    originType = f.dtype
    if originType == "float16" or originType == "float32" or originType == "float64":  # float
        max = 1.
    
    f = f.astype("float64")

    imageMin = np.min(f)
    imageMax = np.max(f)

    
    f = ((f - imageMin) * (max - min) / (imageMax - imageMin) + min).astype(originType)
    return f


#######################################################################################
##################################### Convolution #####################################
#######################################################################################


def conv2D(_f,_mask, norm=True):
    a = (np.shape(_mask)[0] - 1) // 2
    b = (np.shape(_mask)[1] - 1) // 2
    m, n = np.shape(_f)

    # padded = imagePadding(_f, _mask)

    img = _f.copy()  # Do not modify inputed image

    if img.dtype == "float32":  # convert to uint8
        img *= MAX_GRAY
        img = img.astype("uint8")

    g = signal.convolve2d(img, _mask, 'same')

    if norm:
        g = imageNormalize(g)

    return g

def conv2DRefined(_f,_mask, norm=True):
    (a,b) = tuple((x-1)//2 for x in np.shape(_mask))
    fp = imagePadding(_f, _mask, _type='zero').astype('double')
    g = np.zeros(np.shape(fp), dtype='double')
    (rows, cols) = np.shape(g)
    
    for r in range(a, rows-a):
        for c in range(b, cols-b):
            g[r, c] = np.sum(_mask*fp[r-a:r+a+1, c-b:c+b+1])
    g = imageUnpadding(g, _mask)
    if(norm):
        return MAX_GRAY * (g - np.min(g))/(np.max(g) - min(g))
    else:
        return g


######################################################################################
##################################### Histograms #####################################
######################################################################################


def computeHisto(f):
    h = np.zeros(NUM_GRAYS, dtype="int")

    img = f.copy()  # Do not modify inputed image

    if img.dtype == "float32":  # convert to uint8
        img *= MAX_GRAY
        img = img.astype("uint8")

    height, width = np.shape(img)
    for y in range(height):
        for x in range(width):
            h[print(f)[y,x]] += 1
    return h


def computeCumulativeHisto(h):
    normalizedCumulativeHisto = np.zeros(np.size(h), dtype=float)
    cumulativeHisto = np.cumsum(h)
    normalizedCumulativeHisto = (MAX_GRAY * cumulativeHisto/cumulativeHisto[-1]).astype('uint8')
    return cumulativeHisto, normalizedCumulativeHisto


######################################################################################
###################################### Contrast ######################################
######################################################################################

# @brief Computes the contrast of an image
# @param f The image
# see https://stackoverflow.com/a/63441306/14722143
def computeGrayScaleContrast(f):
    if len(f.shape) > 2:
        raise Exception("Not a gray scale image")

    img = f.copy()  # Do not modify inputed image

    if img.dtype == "float32": # convert to uint8
        img *= MAX_GRAY
        img = img.astype("uint8")

    hh, ww = img.shape[:2]

    # compute total pixels
    tot = hh * ww

    # compute histogram
    hist = np.histogram(img,bins=256,range=[0,255])[0]

    # compute cumulative histogram
    cum = np.cumsum(hist)

    # normalize histogram to range 0 to 100
    cum = 100 * cum / tot

    # get bins of percentile at 25 and 75 percent in cum histogram
    i = 0
    while cum[i] < 25:
        i = i+1
    B1 = i
    i = 0
    while cum[i] < 75:
        i = i+1
    B3 = i

    # compute min and max graylevel (which are also the min and max bins)
    min = np.amin(img)
    max = np.amax(img)

    # compute contrast
    contrast = (B3-B1)/(max-min)
    return contrast


######################################################################################
##################################### LookUpTable ####################################
######################################################################################


def applyLUT(f,lut):
    g = lut[f]
    return g


######################################################################################
####################################### Kernels ######################################
######################################################################################


# @brief Create a gaussian kernel
# @param N The size of the kernel
# @param sig The sigma of the gaussian
# @see https://stackoverflow.com/a/29731818/14722143
def gaussianKernel(N, sig=1.):
    """\
    creates gaussian kernel with side length `N` and a sigma of `sig`
    """
    x = np.linspace(-sig, sig, N+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

    