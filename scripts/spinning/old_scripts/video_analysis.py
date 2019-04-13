import cv2 as cv
import os
import glob
import re
import os
import numpy as np
import matplotlib.pyplot as plt

def key_fun(string):
    return int(re.findall('\d+.bmp', string)[0][:-4])


path = "/data/20180927/videos/75Hz_10s"
video_name = '~/plots/20181018/75Hz_video.mp4'
im_files = glob.glob(path + "/*.bmp")
im_files.sort(key=key_fun)
#frame = cv.imread(im_files[0])
#height, width, layers = frame.shape
#fourcc = cv.VideoWriter_fourcc(*'DIVX')
#video = cv.VideoWriter('output.avi', -1, 20.0, (1000, 1000))

show = True
s0 = 38
s1 = 48
s2 = len(im_files)
imarr = np.zeros((s0, s1, s2))
for i, f in enumerate(im_files):
    im = cv.imread(f, 0)
    imarr[:, :, i] = im
    imS = cv.resize(im, (1000, 1000))
    if show:
        cv.imshow('display', imS)
    cv.waitKey(1)
cv.destroyAllWindows()
fftarr = np.fft.rfft(imarr, axis = -1)

f, axarr = plt.subplots(12, 12)
for i in range(12):
    for j in range(12):
        axarr[i, j].imshow(np.abs(fftarr[:, :, (i+1)*(j+1)-1]))
plt.show()
fft = np.fft.rfft(imarr)

