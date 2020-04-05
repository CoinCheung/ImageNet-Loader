
import cv2
import numpy as np

im = cv2.imread("./example.png", 1)



#  ch = im[:, :, 0]
#  n_bins = 256.
#  hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
#

def equalize_func(img):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256
    def tune_channel(ch, flag=False):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0: return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        if flag:
            flag = False
            print(table)
        return table[ch]
    channels = [tune_channel(ch) for i, ch in enumerate(cv2.split(img))]
    #  channels = []
    #  for i, ch in enumerate(cv2.split(img)):
    #      if i == 0 :
    #          channels.append(tune_channel(ch, False))
    #      else:
    #          channels.append(tune_channel(ch, False))
    out = cv2.merge(channels)
    return out


def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256
    def tune_channel(ch, verbose=False):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max().astype(np.int64), ch.min().astype(np.int64)
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = (-low) * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        if verbose:
            print("table[125]: ", table[125])
            print("cut: ", cut)
            print("high: ", high)
            print("low: ", low)
            print(type(low))
            print(low.dtype)
            #  print(scale.dtype)
            print(offset.dtype)
            print(table)
            print("scale: ", scale)
            print("offset: ", offset)
        return table[ch]
    channels = [tune_channel(ch) for ch in cv2.split(img)]
    #  channels = []
    #  for i, ch in enumerate(cv2.split(img)):
    #      if i == 1 :
    #          channels.append(tune_channel(ch, True))
    #      else:
    #          channels.append(tune_channel(ch, False))
    out = cv2.merge(channels)
    return out


imcpp = np.fromfile('./build/res_cpp.bin', dtype=np.uint8)
imcv = im.transpose((2, 0, 1)).ravel()
print(np.sum(imcv.ravel() - imcpp))
for i in range(10):
    #  hist = equalize_func(im)
    hist = autocontrast_func(im, cutoff=0)
print(np.sum(imcpp - hist.ravel()))
#  print(hist.astype(np.int64).ravel())
#  print(hist.astype(np.int64).ravel().dtype)
#  print(np.bincount(ch.ravel(), minlength=256))

