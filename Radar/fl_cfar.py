"""
Code (copie partielle et simplifiée) réalisant la détection de cibles au sein des graphes Doppler
lors des prises de vue réalisées dans le cadre du mémoire :

Synchronization of Multimodal Data Flows for Real-Time AI Analysis
Juin 2022, LEDENT François
"""

import numpy as np
import pickle
from scipy.ndimage import binary_dilation
import os
import matplotlib.pyplot as plt



dir_doppler = "/"   # needs to be changed by the user
dir_cfar = "/"      # needs to be changed by the user


def save_file(filename, content):
    with open(filename, "wb+") as f:
        pickle.dump(content, f)


def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# from LELEC2885 - Image Proc. & Comp. Vision
def resize_and_fix_origin(kernel, size):
    pad0, pad1 = size[0]-kernel.shape[0], size[1]-kernel.shape[1]
    shift0, shift1 = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2
    kernel = np.pad(kernel, ((0, pad0), (0, pad1)), mode='constant')
    kernel = np.roll(kernel, (-shift0, -shift1), axis=(0, 1))
    return kernel


# from LELEC2885 - Image Proc. & Comp. Vision
def fast_convolution(image, kernel):
    kernel_resized = resize_and_fix_origin(kernel, image.shape)
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_resized)
    result = np.fft.ifft2(image_fft * kernel_fft)
    return np.real(result)


# from LELEC2885 - Image Proc. & Comp. Vision (MODIFIED)
def get_gaussian_kernel(sigma, n, divX=1, divY=1):
    indices = np.linspace(-n/2, n/2, n)
    [X, Y] = np.meshgrid(indices, indices)
    X, Y = X/divX, Y/divY
    h = np.exp(-(X**2+Y**2)/(2.0*(sigma)**2))
    h /= h.sum()
    return h


def gaussian_filter(file, gausskern):
    magnfilt = load_file(dir_doppler+file+".ema")
    magnconv = fast_convolution(magnfilt, gausskern)
    save_file(dir_doppler+file+".gauss", magnconv)


def corr_kernel(m, n, sigma):
    kernel = np.zeros((2*n+1, 2*n+1))
    kernel[n-m:n+m+1, n-m:n+m+1] = get_gaussian_kernel(sigma, m*2+1, divX=3)
    kernel0 = (kernel == 0)
    kernel[kernel0] = -1 / np.sum(kernel0)
    return kernel

k, l, m = 10, 30, 3
val = 1 / ((2*m+1)*(l-k))

kernelG = np.zeros((2*l+1, 2*l+1))
kernelG[l-m:l+m+1, :l-k] = val

kernelD = np.zeros((2*l+1, 2*l+1))
kernelD[l-m:l+m+1, l+k+1:] = val

kernelH = np.zeros((2*l+1, 2*l+1))
kernelH[:l-k, l-m:l+m+1] = val

kernelB = np.zeros((2*l+1, 2*l+1))
kernelB[l+k+1:, l-m:l+m+1] = val

def CFAR(file_path, output_dir):
    """Calculate the CFAR

    Args:
        file_path (str): The path to the file to be processed. Has to be a .doppler file.
        output_dir (str): The path to the directory where the output files will be saved.
    """    
    filename = os.path.basename(file_path)
    
    signal = load_file(file_path)

    moyG = fast_convolution(signal, kernelG)
    moyD = fast_convolution(signal, kernelD)
    moyH = fast_convolution(signal, kernelH)
    moyB = fast_convolution(signal, kernelB)
    maxmoy = np.maximum.reduce([moyG, moyD, moyH, moyB])

    magncfar = fast_convolution(signal, np.ones((4, 4))/16) - 1.1*maxmoy
    threshcfar = max(15, 1+0.5*(np.max(magncfar)-1))
    loccfar = np.where(magncfar >= threshcfar)
    fn_to_join = filename.split(".")[:-1] + ["cfar"]


    cfar_fname = ".".join(filename.split(".")[:-1] + ["cfar"])
    loccfar_fname = ".".join(filename.split(".")[:-1] + ["loccfar"])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_file(os.path.join(output_dir,cfar_fname), magncfar)
    save_file(os.path.join(output_dir,loccfar_fname), loccfar)
    
    return magncfar, loccfar


def CORR(file, corrkern):
    magnema = load_file(dir_doppler+file+".ema")

    magncorr = fast_convolution(magnema, corrkern)
    magncorr -= magncorr.mean()
    threshcorr = max(7, 0.5*np.max(magncorr))
    loccorr = np.where(magncorr >= threshcorr)
    
    if not os.path.exists(dir_cfar):
        os.makedirs(dir_cfar)

    save_file(dir_cfar+file+".corr", magncorr)
    save_file(dir_cfar+file+".loccorr", loccorr)


def find_shadows(file):
    loc = load_file(dir_cfar+file+".loc")
    matrix = np.zeros((256, 256), dtype=bool)
    matrix[loc] = 1

    res = np.zeros(matrix.shape, dtype=bool)
    start = np.zeros(matrix.shape)
    diff = np.where(res != matrix)
    shadows = []
    while diff[0].size != 0:
        start[:, :] = 0
        start[diff[0][0], diff[1][0]] = 1
        res_iter = binary_dilation(start, mask=matrix, iterations=-1)
        shadows.append(np.where(res_iter != 0))

        res |= res_iter
        diff = np.where(res != matrix)

    find_targets(file, shadows)


def find_targets(file, shadows):
    magncfar = load_file(dir_cfar+file+".cfar")
    targets = []

    for loc in shadows:
        center_point = np.argmax(magncfar[loc])
        x, y = loc[0][center_point], loc[1][center_point]
        targets.append((x, y))

    print(f"{len(shadows)} target(s) found in {file} : {targets}")
    save_file(dir_cfar+file+".targets", targets)
    
    
def plot_cfar(data):
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    
def plot_raw(path):
    data = load_file(path)
    data = np.log(np.abs(data))
    
    # Scale data
    
    data = ((data - data.min()) / (data.max() - data.min())) * 255
    
    plot_cfar(data)
    
def sumFiles(file,scale=False):
    defined = False
    for i in range(3):
        if not defined:
            data = load_file(file+str(i))
            defined = True
        else:
            data += load_file(file+str(i))

    if scale:
        data = np.log(np.abs(data))
    
        # Scale data
    
        data = ((data - data.min()) / (data.max() - data.min())) * 255
    return data

def CFAR_loaded(data):
    """Calculate the CFAR

    Args:
        file_path (str): The path to the file to be processed. Has to be a .doppler file.
        output_dir (str): The path to the directory where the output files will be saved.
    """    
    signal = data

    moyG = fast_convolution(signal, kernelG)
    moyD = fast_convolution(signal, kernelD)
    moyH = fast_convolution(signal, kernelH)
    moyB = fast_convolution(signal, kernelB)
    maxmoy = np.maximum.reduce([moyG, moyD, moyH, moyB])

    magncfar = fast_convolution(signal, np.ones((4, 4))/16) - 1.1*maxmoy
    threshcfar = max(15, 1+0.5*(np.max(magncfar)-1))
    loccfar = np.where(magncfar >= threshcfar)


    
    return magncfar, loccfar


if __name__ == "__main__":
    # input_file_name="1657880881.025508.doppler"
    # FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    
    # input_path = os.path.join(FILE_PATH,"data","graphes", input_file_name)
    # output_dir = os.path.join(FILE_PATH,"data","cfar")
    
    # data = sumFiles(input_path)
    
    
    from Radar.RadarPacketPcapngReader import RadarPacketPcapngReader as RadarPacketReader
    nb_file = "21"
    rdc_file = f"Fusion/data/radar_cube_data_{nb_file}" # Replace with your output file path
    rdc_reader = RadarPacketReader("", rdc_file)
    
    rdc_reader.load()
    radar_cube_data = rdc_reader.radar_cube_datas
    
    data = np.abs(radar_cube_data[100][:,:,0,0])
    
    magncfar, loccfar = CFAR_loaded(data)
    
    #scaling
    magncfar = np.log(np.abs(magncfar))
    magncfar = ((magncfar - magncfar.min()) / (magncfar.max() - magncfar.min())) * 255
    
    plot_cfar(magncfar)
    plot_cfar(loccfar)
    
    # plot_raw(path=input_path)
    
    # cfar_data, loccfar_data =  CFAR(input_path, output_dir)

    # plot_cfar(cfar_data)
    
    
