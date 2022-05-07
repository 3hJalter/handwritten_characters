from asyncio import futures
import h5py
import pandas
filename = open(r"C:\Users\hhhtn\Downloads\Compressed\archive\handwritten-character-recognition-code\best-trained-model(.h5-file)\model_hand.h5", 'r')
#filename = "model_hand.h5"
h5 = h5py.File(filename, 'r')
futures_data = h5['futures_data']
options_data = h5['options_data']
h5.close()