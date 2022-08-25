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


start = datetime.datetime(2022, 8, 15, 14, 32)
end = datetime.datetime(2022, 8, 15, 14, 36)
spec = CallistoSpectrogram.from_range("SWISS-Landschlacht", start.isoformat(), end.isoformat(), exact=True)
# spec = CallistoSpectrogram.from_url("http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/2022/08/15/SWISS-Landschlacht_20220815_143000_62.fit.gz")
# inter = spec.in_interval(start, end)
pretty = prettify(spec)
mean = pretty.data.mean()
fig = plt.figure(figsize=(6,4))
pretty.plot(fig, vmin=-5*mean, vmax=10*mean)
plt.show()
plt.close(fig)

save(pretty, "pretty.fit.gz")
#load_and_display("nocorrections.fit.gz")
load_and_display("pretty.fit.gz")