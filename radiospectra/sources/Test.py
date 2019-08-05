import numpy as np
import os
import sys
sys.path.insert(1,'C:\\Users\\David Sommer\\Documents\\GitHub\\radiospectra')
from radiospectra.sources.callisto import CallistoSpectrogram
import datetime
import matplotlib.pyplot as plt
import time

specs = []

for root, dirs, files in os.walk(r"C:\Users\David Sommer\Desktop\Blen Sort Test", topdown = False):

    for file in files:

        full_name = os.path.join(root, file)

        image = CallistoSpectrogram.read(full_name)

        specs.append(image)

x = CallistoSpectrogram.new_join_many(specs,polarisations=False)

print(x)

