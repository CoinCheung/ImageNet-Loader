
import cv2
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageOps
import time


impth = "../example.png"



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


def sharpness_func(img, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def color_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Color
    '''
    ## implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )
    #  print(M)
    #  print(factor)
    #  print(M[2,2])
    #  print(M[2,2] * 0.6 + 0.229)
    #  print(np.float32(0.701) * 0.6 + np.float32(0.229))
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out

def contrast_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = np.array([(
        el -74) * factor + 74
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


#  im = cv2.imread(impth, 1)
#  impil = Image.open(impth)

#  t1 = time.time()
#  for i in range(1000):
#      degen_cv2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
#  t2 = time.time()
#  for i in range(1000):
#      degen_pil = np.array(impil.convert("L")).astype(np.float32)
#  t3 = time.time()
#  print(np.max(degen_cv2 - degen_pil))
#  print(np.min(degen_cv2 - degen_pil))
#  print(t2 - t1)
#  print(t3 - t2)
#  factor = 0.2
#  H, W = im.shape[0], im.shape[1]
#  M = np.float32([[1, factor, 0], [0, 1, 0]])
#  out = cv2.warpAffine(im, M, (W, H), borderValue=(128, 128, 128), flags=cv2.INTER_LINEAR).astype(np.uint8)

#  out = color_func(im, 0.6).astype(np.float32)

#  print(im[1:-1, 1:-1, :].shape)
#  sharp_pil = ImageEnhance.Sharpness(impil)
#  out_pil = np.array(sharp_pil.enhance(0.3))[:, :, ::-1].ravel()
#  out = np.array(ImageOps.posterize(impil, 2))[:, :, ::-1]
#  out = np.array(ImageEnhance.Color(impil).enhance(0.6))[:, :, ::-1]
#  out = np.array(ImageEnhance.Color(impil).enhance(0.6))[:, :, ::-1]
#  out = np.array(ImageOps.invert(impil))[:, :, ::-1].astype(np.float32)
#  out = np.array(ImageEnhance.Contrast(impil).enhance(0.6)).astype(np.float32)[:,:,::-1]
#  out = np.array(ImageEnhance.Brightness(impil).enhance(0.6)).astype(np.float32)[:, :, ::-1]
#
#  out = np.array(ImageOps.solarize(impil, 70))[:, :, ::-1].astype(np.float32)
#  #
#  imcpp = np.fromfile('./build/res_cpp.bin', dtype=np.uint8).astype(np.float32)
#  print(np.sum(out.ravel() - imcpp))
#  print(np.max(out.ravel() - imcpp))
#  print(np.min(out.ravel() - imcpp))
#  print('find diff')
#  for i, (e1, e2) in enumerate(zip(out.ravel(), imcpp)):
#      #  if e1 != e2:
#      if e1 != e2:
#          print(i)
#          print(e1)
#          print(e2)
#          break
#  print(im[2, 890])
#  factor = 0.6
#  M = (
#      np.float32([
#          [0.886, -0.114, -0.114],
#          [-0.587, 0.413, -0.587],
#          [-0.299, -0.299, 0.701]]) * factor
#      + np.float32([[0.114], [0.587], [0.299]])
#  )
#  print(np.matmul(im[2, 890], M))
#  print(out.ravel()[27])

#  #  print(np.sum(imcpp - out_pil))
#  kernel = np.ones((3, 3), dtype=np.float32)
#  kernel[1][1] = 5
#  kernel /= 13
#  degenerate = cv2.filter2D(im, -1, kernel)
#  #  degenerate[1:-1, 1:-1, :].tofile('build/degen.bin')
#  degenerate.tofile('build/degen.bin')
#  im[1:-1, 1:-1, :].tofile('build/im.bin')
#  factor = 0.3
#  out = im.copy();
#  out[1:-1, 1:-1, :] = cv2.addWeighted(degenerate[1:-1, 1:-1, :], 1-factor, im[1:-1, 1:-1, :], factor, 0)
#  print(np.sum(out.ravel() - imcpp))

#  a, b = degenerate[1:-1, 1:-1, :].ravel(), im[1:-1, 1:-1, :].ravel()
#  for i in range(1):
#      out = np.array(ImageEnhance.Contrast(impil).enhance(0.6))

#  imcv = im.transpose((2, 0, 1)).ravel()
#  print(np.sum(imcv.ravel() - imcpp))
#  print(np.sum(out_pil - imcpp))
#  for i in range(1):
#      #  hist = equalize_func(im)
#      #  hist = autocontrast_func(im, cutoff=0)
#      sharp_pil = ImageEnhance.Sharpness(impil)
#      out_pil = np.array(sharp_pil.enhance(0.3)).ravel()
#  print(np.sum(imcpp - hist.ravel()))
#  print(hist.astype(np.int64).ravel())
#  print(hist.astype(np.int64).ravel().dtype)
#  print(np.bincount(ch.ravel(), minlength=256))

import dataloader

#  print(dataloader.get_img_by_path('./example.png').shape)
#
#  dl = dataloader.CDataLoader("/data/zzy/imagenet/train/", "grpc/train.txt", 32, [224, 224], True)
#
#  batch = dl.fetch_batch()
#  print(batch.shape)

class CDataLoader(object):

    def __init__(self, imroot, annfile, batchsize, cropsize=(224, 224), shuffle=True, nchw=True, num_workers=4):
        self.shuffle = shuffle
        self.dl = dataloader.CDataLoader(imroot, annfile, batchsize, cropsize, nchw, num_workers)

    def __iter__(self):
        self.dl.restart()
        if self.shuffle: self.dl.shuffle()
        return self

    def __next__(self):
        ##TODO: drop_last
        if self.dl.is_end():
            raise StopIteration
        return self.dl.get_batch()

batchsize = 256
num_workers = 48
dl = CDataLoader("/data/zzy/imagenet/train/", "grpc/train.txt", batchsize, [224, 224], True, num_workers=num_workers)

for e in range(1):
    for i, batch in enumerate(dl):
        print(i, ": ", batch.shape)
        if i == 5: break

print(batch[0, 1, :4, :3])
print(batch[1, 1, :4, :3])
