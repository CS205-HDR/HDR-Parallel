from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im0 = Image.open("orig_0.jpg")
im1 = Image.open("orig_1.jpg")
im2 = Image.open("orig_2.jpg")
im3 = Image.open("orig_3.jpg")

im0.getpixel((0,0))

im4 = Image.blend(im0,im3,alpha=0.5)

id0 = np.array(im0.getdata())
id1 = np.array(im1.getdata())
id2 = np.array(im2.getdata())
id3 = np.array(im3.getdata())

id_comp = (id0+id1+id2+id3)/4

id_comp2 = np.reshape(id_comp, (612,816,3)).astype(np.uint8)

im_comp = Image.fromarray(id_comp2, 'RGB')