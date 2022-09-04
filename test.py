from radiospectra.sources import CallistoSpectrogram
import datetime
import os
import matplotlib.pyplot as plt

def save(spectro, filename):
    if os.path.exists(filename):
        os.remove(filename)

    spectro.save(filename)

def load_and_display(filename):
    s = CallistoSpectrogram.read(filename)
    mean = s.data.mean()
    s.peek(vmin=0, vmax=20, cmap=plt.get_cmap('plasma'))

def prettify(spec):
    return spec.elimwrongchannels().subtract_bg().denoise(full=True)

base_dir = "/Volumes/Daten/bursts"
filename = "/Volumes/Daten/combined.fit.gz"
filename = f"{base_dir}/AUSTRIA-OE3FLB_20220901_093003_57.fit.gz"
spec = CallistoSpectrogram.read(filename)
s = spec.in_interval("09:35", "09:45")
print(s.header.tostring(sep="\n"))
s.save(f"{base_dir}/test.fit.gz")

test = CallistoSpectrogram.read(f"{base_dir}/test.fit.gz")
test.peek()