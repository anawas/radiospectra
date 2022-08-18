from radiospectra.sources import CallistoSpectrogram
from datetime import datetime

if __name__ == '__main__':
    # spec = CallistoSpectrogram.from_range(
    #     'ALASKA-HAARP', datetime(2020, 11, 12, 18, 51), datetime(2020, 11, 12, 18, 56), exact=True
    # ).elimwrongchannels().subtract_bg().denoise()
    spec = CallistoSpectrogram.read(
        'Z:\\radio\\2002-20yy_Callisto\\2020\\11\\12\\ALASKA-HAARP_20201112_184503_01.fit.gz'
    ).elimwrongchannels().subtract_bg().denoise()
    spec.peek()



