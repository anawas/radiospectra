from pathlib import Path
from datetime import datetime
from unittest import mock

import numpy as np

import astropy.units as u
from astropy.time import Time
from sunpy.net import attrs as a

from radiospectra.spectrogram2 import Spectrogram
from radiospectra.spectrogram2.sources import CALISTOSpectrogram


@mock.patch('radiospectra.spectrogram2.spectrogram.parse_path')
def test_callisto(parse_path_moc):
    start_time = Time('2011-06-07 06:24:00.213')
    meta = {
        'fits_meta': {
            'OBS_LAC': 'N',
            'OBS_LAT': 53.0941390991211,
            'OBS_LOC': 'E',
            'OBS_LON': 7.92012977600098,
            'OBS_ALT': 416.5
        },
        'detector': 'e-CALLISTO',
        'instrument': 'e-CALLISTO',
        'observatory': 'BIR',
        'start_time': Time('2011-06-07 06:24:00.213'),
        'end_time': Time('2011-06-07 06:39:00.000'),
        'wavelength': a.Wavelength(20000.0*u.kHz, 91813.00*u.kHz),
        'times': start_time + np.arange(3600) * 0.25 * u.s,
        'freqs': [91.81300354003906, 91.25, 91.06300354003906, 90.625, 90.43800354003906, 89.75,
                  89.68800354003906, 89.0, 88.625, 88.25, 88.06300354003906, 87.56300354003906,
                  87.43800354003906, 87.06300354003906, 86.5, 86.06300354003906, 85.875,
                  85.56300354003906, 84.875, 84.68800354003906, 84.31300354003906, 83.875,
                  83.68800354003906, 83.0, 82.75, 82.43800354003906, 81.875, 81.75,
                  81.18800354003906, 80.75, 80.625, 80.25, 79.68800354003906, 79.25, 79.125,
                  78.68800354003906, 78.43800354003906, 78.06300354003906, 77.43800354003906, 77.0,
                  76.625, 76.56300354003906, 76.0, 75.56300354003906, 75.125, 75.0,
                  74.68800354003906, 74.31300354003906, 73.68800354003906, 73.31300354003906,
                  72.875, 72.625, 72.125, 71.75, 71.56300354003906, 71.0, 70.93800354003906, 70.25,
                  70.18800354003906, 69.68800354003906, 69.43800354003906, 69.06300354003906,
                  68.43800354003906, 68.06300354003906, 67.93800354003906, 67.31300354003906,
                  66.93800354003906, 66.81300354003906, 66.125, 66.06300354003906,
                  65.43800354003906, 65.0, 64.875, 64.25, 64.06300354003906, 63.8129997253418,
                  63.3129997253418, 62.75, 62.5, 62.0, 61.9379997253418, 61.5629997253418, 60.875,
                  60.75, 60.3129997253418, 59.9379997253418, 59.6879997253418, 59.0, 58.625, 58.25,
                  58.1879997253418, 57.5, 57.125, 56.9379997253418, 56.5, 56.3129997253418,
                  55.9379997253418, 55.4379997253418, 54.9379997253418, 54.8129997253418,
                  54.4379997253418, 53.9379997253418, 53.4379997253418, 53.125, 52.9379997253418,
                  52.5629997253418, 51.875, 51.5629997253418, 51.1879997253418, 50.8129997253418,
                  50.5629997253418, 50.0629997253418, 49.6879997253418, 49.3129997253418,
                  48.9379997253418, 48.8129997253418, 48.125, 48.0629997253418, 47.4379997253418,
                  47.25, 46.9379997253418, 46.25, 45.875, 45.8129997253418, 45.3129997253418,
                  45.0629997253418, 44.375, 44.1879997253418, 43.8129997253418, 43.3129997253418,
                  43.1879997253418, 42.5, 42.125, 42.0629997253418, 41.5629997253418,
                  41.1879997253418, 40.6879997253418, 40.4379997253418, 39.875, 39.8129997253418,
                  39.4379997253418, 38.75, 38.375, 38.0629997253418, 37.6879997253418,
                  37.5629997253418, 36.875, 36.8129997253418, 36.25, 36.0629997253418,
                  35.6879997253418, 35.0629997253418, 34.875, 34.5, 34.125, 33.8129997253418,
                  33.125, 33.0, 32.5629997253418, 32.0629997253418, 31.937999725341797, 31.25,
                  30.875, 30.562999725341797, 30.125, 30.062999725341797, 29.375,
                  29.312999725341797, 28.875, 28.25, 28.187999725341797, 27.562999725341797, 27.125,
                  26.75, 26.687999725341797, 26.125, 25.75, 25.375, 25.0, 24.812999725341797,
                  24.125, 23.812999725341797, 23.437999725341797, 23.125, 22.812999725341797,
                  22.437999725341797, 21.875, 21.812999725341797, 21.125, 20.75, 20.375, 20.0, 20.0,
                  20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0] * u.MHz
    }
    array = np.zeros((200, 3600))
    parse_path_moc.return_value = [(array, meta)]
    file = Path('fake.fit.gz')
    spec = Spectrogram(file)
    assert isinstance(spec, CALISTOSpectrogram)
    assert spec.observatory == 'BIR'
    assert spec.instrument == 'E-CALLISTO'
    assert spec.detector == 'E-CALLISTO'
    assert spec.start_time.datetime == datetime(2011, 6, 7, 6, 24, 0, 213000)
    assert spec.end_time.datetime == datetime(2011, 6, 7, 6, 39)
    assert spec.wavelength.min.to(u.MHz) == 20 * u.MHz
    assert spec.wavelength.max.to(u.MHz).round(1) == 91.8 * u.MHz
    assert str(spec.observatory_location) == '(3801942.21260148, 528924.60367802, 5077174.56861812) m'
