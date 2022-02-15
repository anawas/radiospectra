import urllib
import datetime
from collections import defaultdict
import time

import os
import ntpath
from distutils.version import LooseVersion

import numpy as np
from bs4 import BeautifulSoup
from scipy.ndimage import gaussian_filter1d

from astropy.io import fits
from astropy.nddata.ccddata import CCDData
from scipy.optimize import leastsq

from sunpy import __version__
from sunpy.time import parse_time
from sunpy.util.net import download_file

from radiospectra.spectrogram import REFERENCE, LinearTimeSpectrogram, _union
from radiospectra.util import ConditionalDispatch, minimal_pairs, run_cls

__all__ = ['CallistoSpectrogram']

SUNPY_LT_1 = LooseVersion(__version__) < LooseVersion('1.0')
TIME_STR = "%Y%m%d%H%M%S"
DEFAULT_URL = 'http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/'
_DAY = datetime.timedelta(days=1)

DATA_SIZE = datetime.timedelta(seconds=15 * 60)


def parse_filename(href):
    name = href.split('.')[0]
    try:
        inst, date, time, no = name.rsplit('_')
        dstart = datetime.datetime.strptime(date + time, TIME_STR)
    except ValueError:
        # If the split fails, the file name does not match out
        # format,so we skip it and continue to the next
        # iteration of the loop.
        return None
    return inst, no, dstart


PARSERS = [
    # Everything starts with ""
    ("", parse_filename)
]


def query(start, end, instruments=None, url=DEFAULT_URL):
    """
    Get URLs for callisto data from instruments between start and end.

    Parameters
    ----------
    start : `~sunpy.time.parse_time` compatible
    end : `~sunpy.time.parse_time` compatible
    instruments : sequence
        Sequence of instruments whose data is requested.
    url : str
        Base URL for the request.
    """
    day = datetime.datetime(start.year, start.month, start.day)
    while day <= end:
        directory = url + day.strftime('%Y/%m/%d/')
        opn = urllib.request.urlopen(directory)
        try:
            soup = BeautifulSoup(opn, 'lxml')
            for link in soup.find_all("a"):
                href = link.get("href")
                for prefix, parser in PARSERS:
                    if href.startswith(prefix):
                        break

                result = parser(href)
                if result is None:
                    continue
                inst, no, dstart = result

                if (instruments is not None and
                    inst not in instruments and
                        (inst, int(no)) not in instruments):
                    continue

                dend = dstart + DATA_SIZE
                if dend > start and dstart < end:
                    yield directory + href
        finally:
            opn.close()
        day += _DAY


def download(urls, directory):
    """
    Download files from urls into directory.

    Parameters
    ----------
    urls : list of str
        urls of the files to retrieve
    directory : str
        directory to save them in
    """
    return [download_file(url, directory) for url in urls]


def _parse_header_time(date, time):
    """
    Returns `~datetime.datetime` object from date and time fields of header.
    """
    if time is not None:
        date = date + 'T' + time

    if SUNPY_LT_1:
        time = parse_time(date)
    else:
        time = parse_time(date).datetime

    return time


class CallistoSpectrogram(LinearTimeSpectrogram):
    """
    Class used for dynamic spectra coming from the Callisto network.

    Attributes
    ----------
    header : `~astropy.io.fits.Header`
        main header of the FITS file
    axes_header : `~astropy.io.fits.Header`
        header for the axes table
    swapped : bool
        flag that specifies whether originally in the file the x-axis was
        frequency
    """
    # XXX: Determine those from the data.
    SIGMA_SUM = 75
    SIGMA_DELTA_SUM = 20
    _create = ConditionalDispatch.from_existing(LinearTimeSpectrogram._create)
    create = classmethod(_create.wrapper())
    # Contrary to what pylint may think, this is not an old-style class.
    # pylint: disable=E1002,W0142,R0902

    # This needs to list all attributes that need to be
    # copied to maintain the object and how to handle them.
    COPY_PROPERTIES = LinearTimeSpectrogram.COPY_PROPERTIES + [
        ('header', REFERENCE),
        ('swapped', REFERENCE),
        ('axes_header', REFERENCE)
    ]

    # List of instruments retrieved in July 2012 from
    # http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/
    INSTRUMENTS = {
        'ALASKA', 'ALMATY', 'BIR', 'DARO', 'HB9SCT', 'HUMAIN',
        'HURBANOVO', 'KASI', 'KENYA', 'KRIM', 'MALAYSIA', 'MRT1',
        'MRT2', 'OOTY', 'OSRA', 'SWMC', 'TRIEST', 'UNAM'
    }

    ARRAY_TYPE = np.float_
    MISSING_VALUE = np.nan

    def save(self, filepath=None):
        """
        Save modified spectrogram back to filepath.

        Parameters
        ----------
        filepath : str
            path to save the spectrogram to
        """

        if filepath is None:
            filepath = self.filename

        main_header = self.get_header()
        data = CCDData(data=self.data, header=main_header, unit='Sun')
        # XXX: Update axes header.

        data.header.append(card=('BZERO', 0, 'scaling offset'))
        data.header.append(card=('BSCALE', 1, 'scaling factor'))
        data.header['NAXIS1'] = (data.header['NAXIS1'], 'length of data axis 1')
        data.header['NAXIS2'] = (data.header['NAXIS2'], 'length of data axis 2')

        freq_col = fits.Column(
            name="FREQUENCY",
            format=f"{len(self.freq_axis)}D8.3",
            array=np.reshape(np.array(self.freq_axis), (1, len(self.freq_axis)))
        )
        time_col = fits.Column(
            name="TIME",
            format=f"{len(self.time_axis)}D8.3",
            array=np.reshape(np.array(self.time_axis), (1, len(self.time_axis)))
        )

        cols = fits.ColDefs([time_col, freq_col])
        table = fits.BinTableHDU.from_columns(cols, header=self.axes_header, name='AXES')

        table.header['TTYPE1'] = (table.header['TTYPE1'], 'label for field   1')
        table.header['TFORM1'] = (table.header['TFORM1'], 'data format of field: 8-byte DOUBLE')
        table.header['TTYPE2'] = (table.header['TTYPE2'], 'label for field   2')
        table.header['TFORM2'] = (table.header['TFORM2'], 'data format of field: 8-byte DOUBLE')

        table.header['TSCAL1'] = 1
        table.header['TZERO1'] = 0
        table.header['TSCAL2'] = 1
        table.header['TZERO2'] = 0

        hdulist = data.to_hdu()
        hdulist.insert(1, table)

        if self.rfi_freq_axis.size != 0:
            # Putting information, which frequencies channels were removed into the header
            freq_col = fits.Column(
                name="RFI_FREQUENCY",
                format=f"{len(self.rfi_freq_axis)}D8.3",
                array=np.reshape(np.array(self.rfi_freq_axis), (1, len(self.rfi_freq_axis)))
            )
            rfi_freq_col = fits.ColDefs([freq_col])
            rfi_table = fits.BinTableHDU.from_columns(rfi_freq_col, header=self.axes_header, name='RFI_FREQ')
            hdulist.insert(2, rfi_table)

        if not os.path.exists(filepath):
            hdulist.writeto(filepath)
            return filepath
        else:
            i = 0
            split = filepath.split('.')
            new_filepath = split[0] + f' ({i})' + f'{"." if len(split[1:]) > 0 else ""}' + '.'.join(split[1:])
            while os.path.exists(new_filepath):
                i += 1
                new_filepath = split[0] + f' ({i})' + f'{"." if len(split[1:]) > 0 else ""}' + '.'.join(split[1:])
            hdulist.writeto(new_filepath)
            return new_filepath

    def get_header(self):
        """
        Returns the updated header.
        """
        header = self.header.copy()

        if self.swapped:
            header['NAXIS2'] = self.shape[1]  # pylint: disable=E1101
            header['NAXIS1'] = self.shape[0]  # pylint: disable=E1101
        else:
            header['NAXIS1'] = self.shape[1]  # pylint: disable=E1101
            header['NAXIS2'] = self.shape[0]  # pylint: disable=E1101
        return header

    @classmethod
    def read(cls, filename: str, **kwargs):
        """
        Reads in FITS file and return a new CallistoSpectrogram.
        Any unknown (i.e. any except filename) keyword arguments get
        passed to fits.open.

        Parameters
        ----------
        filename : str
            path of the file to read
        """
        fl = fits.open(filename, **kwargs)
        data = np.ma.array(fl[0].data, dtype=cls.ARRAY_TYPE, mask=np.full(fl[0].data.shape, False))
        data.mask[np.where(np.isnan(data.data))] = True

        header = fl[0].header

        # not every fits file does provide axes in their header file
        axes = None
        if len(fl) != 1:
            axes = fl[1]

        modified = False
        # simple patch for TIME-END issue
        if header['TIME-END'][6:] == '60':
            header['TIME-END'] = (header['TIME-END'][:6] + '59', header.comments['TIME-END'])
            modified = True

        if header['TIME-END'][:2] == '24':
            header['TIME-END'] = ('00' + header['TIME-END'][2:], header.comments['TIME-END'])
            modified = True

        # since there are fits files where both cases of the '60' and '24' changes occurs,
        # this makes sure only one time '[modified]' is written in the comments section
        if modified:
            header['TIME-END'] = (header['TIME-END'], header.comments['TIME-END'] + ' [modified]')

        start = _parse_header_time(
            header['DATE-OBS'], header.get('TIME-OBS', header.get('TIME$_OBS'))
        )

        end = _parse_header_time(
            header['DATE-END'], header.get('TIME-END', header.get('TIME$_END'))
        )

        swapped = "time" not in header["CTYPE1"].lower()

        # Swap dimensions so x-axis is always time.
        if swapped:
            t_delt = header["CDELT2"]
            t_init = header["CRVAL2"] - t_delt * header["CRPIX2"]
            t_label = header["CTYPE2"]

            f_delt = header["CDELT1"]
            f_init = header["CRVAL1"] - t_delt * header["CRPIX1"]
            f_label = header["CTYPE1"]
            data = data.transpose()
        else:
            t_delt = header["CDELT1"]
            t_init = header["CRVAL1"] - t_delt * header["CRPIX1"]
            t_label = header["CTYPE1"]

            f_delt = header["CDELT2"]
            f_init = header["CRVAL2"] - t_delt * header["CRPIX2"]
            f_label = header["CTYPE2"]

        # Table may contain the axes data. If it does, the other way of doing
        # it might be very wrong.
        tm = None
        fq = None
        if axes is not None:
            try:
                # It's not my fault. Neither supports __contains__ nor .get
                tm = axes.data['TIME']
            except KeyError:
                tm = None
            try:
                fq = axes.data['FREQUENCY']
            except KeyError:
                fq = None

        if tm is not None:
            # Fix dimensions (whyever they are (1, x) in the first place)
            time_axis = np.squeeze(tm)
        else:
            # Otherwise, assume it's linear.
            time_axis = np.linspace(0, data.shape[1] - 1, num=data.shape[1]) * t_delt  # pylint: disable=E1101

        if fq is not None:
            freq_axis = np.squeeze(fq)
        else:
            freq_axis = np.linspace(0, data.shape[0] - 1, num=data.shape[0]) * f_delt + f_init  # pylint: disable=E1101

        # Remove duplicate entries on the borders
        left = 1
        while freq_axis[left] == freq_axis[0]:
            left += 1
        right = data.shape[0] - 1
        while freq_axis[right] == freq_axis[-1]:
            right -= 1

        c_left = left - 1
        c_right = right + 2

        if c_left > 0:
            data.data[:c_left, :] = cls.MISSING_VALUE
            data.mask[:c_left, :] = True
        if c_right < (len(freq_axis) - 1):
            data.data[c_right:, :] = cls.MISSING_VALUE
            data.mask[c_right:, :] = True

        # mask the zigzag pattern
        zigzag = [*range(255), *range(255)]
        for row_index in range(data.shape[0]):
            if np.array_equal(data[row_index, :len(zigzag)], zigzag):
                data.data[row_index] = cls.MISSING_VALUE
                data.mask[row_index] = True

        content = header["CONTENT"]
        instruments = {header["INSTRUME"]}

        axes_header = None
        if axes:
            axes_header = axes.header

        fl.close()
        return cls(
            data, time_axis, freq_axis, start, end, t_init, t_delt,
            t_label, f_label, content, instruments,
            header, axes_header, swapped, filename
        )

    def __init__(self, data, time_axis, freq_axis, start, end,
                 t_init=None, t_delt=None, t_label="Time", f_label="Frequency",
                 content="", instruments=None, header=None, axes_header=None,
                 swapped=False, filename=None):
        # Because of how object creation works, there is no avoiding
        # unused arguments in this case.
        # pylint: disable=W0613

        if not isinstance(data, np.ma.MaskedArray):
            data = np.ma.array(data, dtype=self.ARRAY_TYPE, mask=False)

        super(CallistoSpectrogram, self).__init__(
            data, time_axis, freq_axis, start, end,
            t_init, t_delt, t_label, f_label,
            content, instruments
        )

        self.header = header
        self.axes_header = axes_header
        self.swapped = swapped
        if filename:
            head, tail = ntpath.split(filename)
            self.filename = tail or ntpath.basename(head)
        else:
            self.filename = None

    @classmethod
    def is_datasource_for(cls, header):
        """
        Check if class supports data from the given FITS file.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            main header of the FITS file
        """
        return header.get('instrume', '').strip() in cls.INSTRUMENTS

    def remove_border(self):
        """
        Remove duplicate entries on the borders.
        """
        left = 0
        while self.freq_axis[left] == self.freq_axis[0]:
            left += 1
        right = self.shape[0] - 1
        while self.freq_axis[right] == self.freq_axis[-1]:
            right -= 1
        return self[left - 1:right + 2, :]

    @classmethod
    def read_many(cls, filenames, sort_by=None):
        """
        Returns a list of CallistoSpectrogram objects read from filenames.

        Parameters
        ----------
        filenames : list of str
            list of paths to read from
        sort_by : str
            optional attribute of the resulting objects to sort from, e.g.
            start to sort by starting time.
        """
        objs = list(map(cls.read, filenames))
        if sort_by is not None:
            objs.sort(key=lambda x: getattr(x, sort_by))
        return objs

    @classmethod
    def from_range(cls, instrument, start, end, exact: bool = False, distribution: int = 0):
        """
        Automatically download data from instrument between start and end and
        join it together.

        Parameters
        ----------
        instrument : str
            instrument to retrieve the data from
        start : `~sunpy.time.parse_time` compatible
            start of the measurement
        end : `~sunpy.time.parse_time` compatible
            end of the measurement
        exact : bool
            return data only for the requested time interval
        distribution : int
            extend data start-end so, that the data can be changed in such a way that n pieces (minutes)
            with the same duration distribution will be obtained. Only apply if 'exact' parameter is True.
        """

        def missing_nan_data(_start, _end, freq_axis_len, time_steps=4):
            """
            Create a numpy array whit missing_nan_data for fill in CallistoSpectrogram data.

            Parameters
            ----------
            _start : `~datetime` compatible
                start of the interval
            _end : `~datetime` compatible
                end of the interval
            freq_axis_len : int
                len of the freq_axis of the CallistoSpectrogram object
            time_steps: int
                spec time_axis property step count inside 1 min.
            """
            diference = int(math.ceil(
                ((_end - _start).total_seconds() / DATA_SIZE.total_seconds())
            ) * DATA_SIZE.total_seconds())
            nan_shape = (freq_axis_len, diference * time_steps)
            return np.full(nan_shape, np.nan)

        if SUNPY_LT_1:
            start = parse_time(start)
            end = parse_time(end)
        else:
            start = parse_time(start).datetime
            end = parse_time(end).datetime

        urls = query(start, end, [instrument])

        # Do it using next
        urls = list(urls)

        if len(urls) > 0:
            url = urls[0]
            inst, no, dstart = parse_filename(url.split('/')[-1])

            if dstart.replace(microsecond=0) > start and start.hour < 1 and start.minute < 15:
                # If start hour:min < 00:15 start from 15 min less because many times 00:00 start from midnight - 1 min
                # So, update url list files with the previous file
                aux_start = start - datetime.timedelta(minutes=15)
                new_urls = list(query(aux_start, end, [instrument]))
                urls = new_urls + urls if len(new_urls) > 0 else urls

        data = list(map(cls.from_url, urls))
        freq_buckets = defaultdict(list)
        for elem in data:
            freq_buckets[tuple(elem.freq_axis)].append(elem)
        try:
            spec = cls.combine_frequencies(
                [cls.new_join_many(elem) for elem in freq_buckets.values()]
            )

            spec.start = spec.start.replace(microsecond=0)
            spec.end = spec.end.replace(microsecond=0)

            if exact or spec.start > start or spec.end < end:
                step = spec.time_axis[1]

            if spec.start > start:
                # There isn't data between spec.start and the requested start datetime
                # Fill whit np.nan values and update spec attributes
                nan_data = missing_nan_data(start, spec.start, len(spec.freq_axis), int(1 / step))
                spec.data = np.concatenate((nan_data, spec.data), axis=1)
                spec.start = start
                spec.time_axis = np.arange(
                    start=0.0, stop=len(spec.time_axis) + nan_data.shape[-1], step=step
                )

            if spec.end < end:
                # There isn't data between spec.end and the requested end datetime
                # Fill whit np.nan values and update spec attributes
                nan_data = missing_nan_data(spec.end, end, len(spec.freq_axis), int(1 / step))
                spec.data = np.concatenate((spec.data, nan_data), axis=1)
                spec.end = end
                spec.time_axis = np.arange(
                    start=0.0, stop=len(spec.time_axis) + nan_data.shape[-1], step=step
                )

            if exact:
                spec.start = start
                spec.end = end

                start_difference = (start - spec.start).total_seconds()
                end_difference = (end - start).total_seconds()
                from_start = int(start_difference * 1 / step)

                if distribution > 1:
                    remainder, quotient = math.modf(end_difference / DATA_SIZE.total_seconds())

                    if quotient == 0:
                        end_difference = DATA_SIZE.total_seconds()
                    else:
                        end_difference = quotient * DATA_SIZE.total_seconds()
                        if remainder > 0:
                            end_difference += DATA_SIZE.total_seconds()

                    spec.end = spec.start + datetime.timedelta(seconds=end_difference)

                to_end = int(from_start + end_difference * 1 / step)

                spec.time_axis = spec.time_axis[from_start: to_end]
                spec.data = spec.data[:, from_start: to_end]

            return spec

        except ValueError:
            raise ValueError("No data found.")

    def _overlap(self, other):
        """
        Find frequency and time overlap of two spectrograms.
        """
        one, two = self.intersect_time([self, other])
        ovl = one.freq_overlap(two)
        return one.clip_freq(*ovl), two.clip_freq(*ovl)

    @staticmethod
    def _to_minimize(a, b):
        """
        Function to be minimized for matching to frequency channels.
        """

        def _fun(p):
            if p[0] <= 0.2 or abs(p[1]) >= a.max():
                return float("inf")
            return a - (p[0] * b + p[1])

        return _fun

    def _homogenize_params(self, other, maxdiff=1):
        """
        Return triple with a tuple of indices (in self and other,
        respectively), factors and constants at these frequencies.

        Parameters
        ----------
        other : `radiospectra.CallistoSpectrogram`
            Spectrogram to be homogenized with the current one.
        maxdiff : float
            Threshold for which frequencies are considered equal.
        """

        pairs_indices = [
            (x, y) for x, y, d in minimal_pairs(self.freq_axis, other.freq_axis)
            if d <= maxdiff
        ]

        pairs_data = [
            (self[n_one, :], other[n_two, :]) for n_one, n_two in pairs_indices
        ]

        # XXX: Maybe unnecessary.
        pairs_data_gaussian = [
            (gaussian_filter1d(a, 15), gaussian_filter1d(b, 15))
            for a, b in pairs_data
        ]

        # If we used integer arithmetic, we would accept more invalid
        # values.
        pairs_data_gaussian64 = np.float64(pairs_data_gaussian)
        least = [
            leastsq(self._to_minimize(a, b), [1, 0])[0]
            for a, b in pairs_data_gaussian64
        ]

        factors = [x for x, y in least]
        constants = [y for x, y in least]

        return pairs_indices, factors, constants

    def homogenize(self, other, maxdiff=1):
        """
        Return overlapping part of self and other as (self, other) tuple.
        Homogenize intensities so that the images can be used with
        combine_frequencies. Note that this works best when most of the picture
        is signal, so use :meth:`in_interval` to select the subset of your
        image before applying this method.

        Parameters
        ----------
        other : `radiospectra.CallistoSpectrogram`
            Spectrogram to be homogenized with the current one.
        maxdiff : float
            Threshold for which frequencies are considered equal.
        """
        one, two = self._overlap(other)
        pairs_indices, factors, constants = one._homogenize_params(
            two, maxdiff
        )
        # XXX: Maybe (xd.freq_axis[x] + yd.freq_axis[y]) / 2.
        pairs_freqs = [one.freq_axis[x] for x, y in pairs_indices]

        # XXX: Extrapolation does not work this way.
        # XXX: Improve.
        f1 = np.polyfit(pairs_freqs, factors, 3)
        f2 = np.polyfit(pairs_freqs, constants, 3)

        return (one,
                two * np.polyval(f1, two.freq_axis)[:, np.newaxis] +
                np.polyval(f2, two.freq_axis)[:, np.newaxis])

    def extend(self, minutes=15, **kwargs):
        """
        Requests subsequent files from the server.

        If minutes is negative, retrieve preceding files.
        """
        if len(self.instruments) != 1:
            raise ValueError

        instrument = next(iter(self.instruments))
        if minutes > 0:
            data = CallistoSpectrogram.load_from_range(
                instrument,
                self.end, self.end + datetime.timedelta(minutes=minutes)
            )
        elif minutes < 0:
            data = CallistoSpectrogram.load_from_range(
                instrument,
                self.start - datetime.timedelta(minutes=-minutes), self.start
            )
        else:
            return self

        filtered_data = list()
        for current_data in data:
            if current_data.header["PWM_VAL"] == self.header["PWM_VAL"]:
                filtered_data.append(current_data)

        return CallistoSpectrogram.new_join_many([self] + filtered_data, **kwargs)

    @classmethod
    def from_url(cls, url):
        """
        Returns CallistoSpectrogram read from URL.

        Parameters
        ----------
        url : str
            URL to retrieve the data from

        Returns
        -------
        newSpectrogram : `radiospectra.CallistoSpectrogram`
        """
        return cls.read(url)

    @classmethod
    def load_from_range(cls, instrument, start, end, **kwargs):
        """
        Automatically download data from instrument between start and
        end.

        Parameters
        ----------
        instrument : str
            instrument to retrieve the data from
        start : `~sunpy.time.parse_time` compatible
            start of the measurement
        end : `~sunpy.time.parse_time` compatible
            end of the measurement
        """
        kw = {
            'maxgap': None,
            'fill': cls.JOIN_REPEAT,
        }

        kw.update(kwargs)
        if SUNPY_LT_1:
            start = parse_time(start)
            end = parse_time(end)
        else:
            start = parse_time(start).datetime
            end = parse_time(end).datetime
        urls = query(start, end, [instrument])
        specs = list(map(cls.from_url, urls))

        return specs

    @classmethod
    def new_join_many(cls, specs: ['CallistoSpectrogram'], polarisations=False):
        """
        Produce new Spectrogram that contains spectrograms
        joined together in time and frequency.

        Parameters
        ----------
        specs : list
            List of CallistoSpectrogram's to join together in time.
        polarisations : bool
            TODO
        """
        # checks
        if not specs:
            raise ValueError("Need at least one spectrogram.")

        if len(specs) == 1:
            return specs[0]

        # sorts specs
        specs = sorted(specs, key=lambda x: x.start)

        # initiates combine_polarisations or gives separated spectrograms back
        if polarisations:
            sorting_dict = cls.detect_and_combine_polarisations(specs)
        else:
            pwm_val_list = set(
                map(lambda x: "{}_{}".format(x.header['PWM_VAL'], x.filename.split('.')[0].split('_')[-1]), specs))
            sorting_dict = dict(map(lambda x: (x, []), pwm_val_list))
            for spec in specs:
                sorting_dict["{}_{}".format(spec.header['PWM_VAL'], spec.filename.split('.')[0].split('_')[-1])].append(
                    spec)
            map(lambda x: x.start, list(sorting_dict.values()))

        if len(specs) == 1:
            return specs[0]

        joined_spec_list = []
        for PWM, sorted_specs in sorting_dict.items():
            if not isinstance(sorted_specs[0], CallistoSpectrogram):
                raise ValueError("Can only combine CallistoSpectrogram's.")
            instr = sorted_specs[0].header['INSTRUME']
            delta = sorted_specs[0].header['CDELT1']

            first_time_point = sorted_specs[0].start + datetime.timedelta(seconds=sorted_specs[0].time_axis[0])
            last_time_point = sorted_specs[0].start + datetime.timedelta(seconds=sorted_specs[0].time_axis[-1])

            for spec in sorted_specs[1:]:
                if not isinstance(spec, CallistoSpectrogram):
                    raise ValueError("Can only combine CallistoSpectrogram's.")
                if spec.header['INSTRUME'] != instr:
                    raise ValueError("Can only combine spectrogram's from the same instrument.")
                if spec.header['CDELT1'] != delta:
                    raise ValueError("Can only combine spectrogram's with the same time delta (CDELT1).")

                cur_start = spec.start + datetime.timedelta(seconds=spec.time_axis[0])
                cur_end = spec.start + datetime.timedelta(seconds=spec.time_axis[-1])

                if cur_start < first_time_point:
                    first_time_point = cur_start
                if cur_end > last_time_point:
                    last_time_point = cur_end

            new_header = sorted_specs[0].get_header()
            new_axes_header = sorted_specs[0].axes_header

            borderless_specs = [sp.mark_border() for sp in sorted_specs]

            new_freq_axis = np.array(sorted(_union(set(sp.freq_axis) for sp in borderless_specs), key=lambda y: -y))
            new_time_axis = np.arange(0, (last_time_point - first_time_point).total_seconds() + 0.00001, delta)

            for spec in sorted_specs:
                curr_start = spec.start + datetime.timedelta(seconds=spec.time_axis[0])
                diff = (curr_start - first_time_point).total_seconds()
                diff_index = int(diff / delta)
                new_time_axis[diff_index:diff_index + spec.time_axis.shape[0]] = (spec.time_axis + diff)

            nan_arr = np.empty((len(new_freq_axis), len(new_time_axis)), dtype=cls.ARRAY_TYPE)
            nan_arr[:] = cls.MISSING_VALUE
            new_data = np.ma.array(nan_arr, mask=True)

            # fill new data array
            for sp in borderless_specs:
                if np.array_equal(new_freq_axis, sp.freq_axis):
                    c_start_time = (
                        (sp.start + datetime.timedelta(seconds=sp.time_axis[0])) - first_time_point).total_seconds()
                    temp_pos_time = np.where(new_time_axis == c_start_time)
                    if len(temp_pos_time[0]) == 1:
                        new_pos_time = temp_pos_time[0][0]

                        new_data[:, new_pos_time:new_pos_time + sp.shape[1]] = sp.data[:, :]
                        new_data.mask[:, new_pos_time:new_pos_time + sp.shape[1]] = sp.data.mask[:, :]
                else:
                    for pos_freq in range(sp.shape[0]):
                        c_freq = sp.freq_axis[pos_freq]
                        new_pos_freq = np.where(new_freq_axis == c_freq)[0][0]

                        c_start_time = ((sp.start + datetime.timedelta(
                            seconds=sp.time_axis[0])) - first_time_point).total_seconds()

                        temp_pos_time = np.where(new_time_axis == c_start_time)
                        if len(temp_pos_time[0]) == 1:
                            new_pos_time = temp_pos_time[0][0]

                            new_data[new_pos_freq, new_pos_time:new_pos_time + sp.shape[1]] = sp.data[pos_freq, :]
                            new_data.mask[new_pos_freq, new_pos_time:new_pos_time + sp.shape[1]] = sp.data.mask[
                                                                                                   pos_freq, :]

            ftp_time = first_time_point.time()
            second_of_day = ftp_time.hour * 3600 + ftp_time.minute * 60 + ftp_time.second

            params = {
                'time_axis': new_time_axis,
                'freq_axis': new_freq_axis,
                'start': first_time_point,
                'end': last_time_point,
                't_delt': delta,
                't_init': second_of_day,
                't_label': sorted_specs[0].t_label,
                'f_label': sorted_specs[0].f_label,
                'content': sorted_specs[0].content,
                'instruments': _union(spec.instruments for spec in sorted_specs),
                'filename': f'JOINED_{sorted_specs[0].filename}'
            }

            new_header['DATE-OBS'] = min([x.get_header()['DATE-OBS'] for x in sorted_specs])
            new_header['TIME-OBS'] = min([x.get_header()['TIME-OBS'] for x in sorted_specs])
            new_header['DATE-END'] = max([x.get_header()['DATE-END'] for x in sorted_specs])
            new_header['TIME-END'] = max([x.get_header()['TIME-END'] for x in sorted_specs])
            new_header['DATAMIN'] = min([x.get_header()['DATAMIN'] for x in sorted_specs])
            new_header['DATAMAX'] = max([x.get_header()['DATAMAX'] for x in sorted_specs])
            new_header['CRVAL1'] = second_of_day
            new_header['CRVAL2'] = len(new_freq_axis)

            new_axes_header['NAXIS1'] = int(new_axes_header['BITPIX']) * (len(new_time_axis) + len(new_freq_axis))
            new_axes_header['TFORM1'] = str(len(new_time_axis)) + "D8.3"
            new_axes_header['TFORM2'] = str(len(new_freq_axis)) + "D8.3"

            joined_spec = CallistoSpectrogram(new_data, header=new_header, axes_header=new_axes_header, **params)
            joined_spec.adjust_header()
            joined_spec_list.append(joined_spec)

        if len(joined_spec_list) == 1:
            return joined_spec_list[0]
        else:
            return joined_spec_list

    @classmethod
    def detect_and_combine_polarisations(cls, specs: ['CallistoSpectrogram']) -> ['CallistoSpectrogram']:
        """Takes a list of spectrogram and evaluates and processes all polarisations

        Parameters
        ----------
        specs:
            A list of spectrograms
        return:
            A dictionary of all spectrograms sorted by there PWM_VAL and lists for all combined polarisations
        """
        pwm_val_list = set(map(lambda x: x.header['PWM_VAL'], specs))
        sorted_work_dict = dict(map(lambda x: (x, []), pwm_val_list))
        pwm_val_list = set(
            map(lambda x: "{}_{}".format(x.header['PWM_VAL'], x.filename.split('.')[0].split('_')[-1]), specs))
        sorting_dict = dict(map(lambda x: (x, []), pwm_val_list))
        pol_val_list = set(map(lambda x: "{}_{}".format(x.header['PWM_VAL'], "Pol"), specs))
        sorting_dict.update(map(lambda x: (x, []), pol_val_list))
        for spec in specs:
            sorting_dict["{}_{}".format(spec.header['PWM_VAL'], spec.filename.split('.')[0].split('_')[-1])].append(
                spec)
            sorted_work_dict[spec.header['PWM_VAL']].append(spec)
        map(lambda x: x.start, list(sorting_dict.values()))
        for PWM, specs_list in sorted_work_dict.items():
            for index, spec1 in enumerate(specs_list[:-1]):
                spec2 = specs_list[index + 1]
                delta1 = float(spec1.header['CDELT1'])
                delta2 = float(spec1.header['CDELT1'])
                if abs(delta1 - delta2) > 0.000001:
                    continue
                if spec1.header['INSTRUME'] != spec2.header['INSTRUME']:
                    continue
                if spec1.shape != spec2.shape:
                    continue
                if abs((spec1.start - spec2.start).total_seconds()) > delta1:
                    continue
                if not np.array_equal(spec1.freq_axis, spec2.freq_axis):
                    continue
                if not np.array_equal(spec1.time_axis, spec2.time_axis):
                    continue
                merged_spec = CallistoSpectrogram.combine_polarisation(specs[index], specs[index + 1])
                sorting_dict["{}_{}".format(merged_spec.header['PWM_VAL'], "Pol")].append(merged_spec)
        del_list = []
        for PWM, specs_list in sorting_dict.items():
            if not specs_list:
                del_list.append(PWM)
        for val in del_list:
            del sorting_dict[val]
        return sorting_dict

    @classmethod
    def combine_polarisation(cls, spec1: 'CallistoSpectrogram', spec2: 'CallistoSpectrogram') -> 'CallistoSpectrogram':
        """
        Compares and combines two spectrograms that are polarisations of the same event

        Parameters
        ----------
        spec1 : CallistoSpectrogram
            The first polarized spectrogram.
        spec2 : CallistoSpectrogram
            The second polarized spectrogram.
        """

        def pythagoras_combine(a, b):
            return (a ** 2 + b ** 2) ** 0.5

        # checks
        delta1 = float(spec1.header['CDELT1'])
        delta2 = float(spec1.header['CDELT1'])
        if abs(delta1 - delta2) > 0.000001:
            raise ValueError('CDELT1 of spectrograms are not the same')
        if spec1.header['INSTRUME'] != spec2.header['INSTRUME']:
            raise ValueError('Instruments of spectrograms are not the same')
        if spec1.shape != spec2.shape:
            raise ValueError('Shapes of spectrograms not the same')
        if abs((spec1.start - spec2.start).total_seconds()) > delta1:
            raise ValueError('Start times of spectrograms are too far from each other')
        if not np.array_equal(spec1.freq_axis, spec2.freq_axis):
            raise ValueError('Frequency axes of spectrograms are not the same')
        if not np.array_equal(spec1.time_axis, spec2.time_axis):
            raise ValueError('Time axes of spectrograms are not the same')

        merge_funk = np.vectorize(pythagoras_combine, excluded=[cls.MISSING_VALUE])
        merged_matrix = merge_funk(spec1.data, spec2.data)
        x = np.logical_or(spec1.data.mask, spec2.data.mask)
        merged_matrix = np.ma.array(merged_matrix, mask=x)

        merged_spec = spec1._with_data(merged_matrix)
        merged_spec.header["DATAMIN"] = merged_matrix.min()
        merged_spec.header["DATAMAX"] = merged_matrix.max()
        return merged_spec

    def remove_single_freq_rfi(self, threshold=17, row_window_height=3):
        """
        Detects rfi in single frequencies and masks the data

        Parameters
        ----------
        threshold : Num
            Minimal Intensity difference between the row and the mean of the surrounding rows to be flagged as rfi.
        row_window_height : int
            The amount of rows before and after should be considered for the surrounding rows.
        """
        rfi_mask = np.full(self.shape, False)

        for row_index in range(self.shape[0]):
            aff_row = self.data[row_index]
            rows_before = self.data[max(row_index - row_window_height, 0):row_index]
            rows_after = self.data[row_index + 1:min(row_index + 1 + row_window_height, self.shape[0])]
            window_rows = np.concatenate([rows_before, rows_after], axis=0)

            window_mean = np.mean(window_rows, axis=0)
            diff = abs(aff_row - window_mean)

            rfi_mask[row_index, np.where(diff >= threshold)] = True

        new_data = self.data.copy()
        new_data.mask[np.where(rfi_mask)] = True
        new_data.data[np.where(rfi_mask)] = self.MISSING_VALUE

        return self._with_data(new_data)

    def mark_border(self):
        """
        Mark duplicate entries on the borders.
        """
        left = 1
        while self.freq_axis[left] == self.freq_axis[0]:
            left += 1
        right = self.data.shape[0] - 1
        while self.freq_axis[right] == self.freq_axis[-1]:
            right -= 1

        c_left = left - 1
        c_right = right + 2

        if c_left > 0:
            self.data.data[:c_left, :] = self.MISSING_VALUE
            self.data.mask[:c_left, :] = True
        if c_right < (len(self.freq_axis) - 1):
            self.data.data[c_right:, :] = self.MISSING_VALUE
            self.data.mask[c_right:, :] = True

        return self

    def adjust_header(self, date_obs=None, time_obs=None, date_end=None, time_end=None):
        # data header
        new_header = self.get_header()

        if date_obs is not None:
            new_header['DATE-OBS'] = date_obs
        if time_obs is not None:
            new_header['TIME-OBS'] = time_obs
        if date_end is not None:
            new_header['DATE-END'] = date_end
        if time_end is not None:
            new_header['TIME-END'] = time_end

        data_min = np.amin(self.data)
        data_max = np.amax(self.data)

        new_header['DATAMIN'] = int(data_min) if not np.isnan(data_min) else 0
        new_header['DATAMAX'] = int(data_max) if not np.isnan(data_max) else 0

        self.header = new_header

        # axes header
        new_axes_header = self.axes_header

        new_axes_header['NAXIS1'] = int(new_axes_header['BITPIX']) * (len(self.time_axis) + len(self.freq_axis))
        new_axes_header['TFORM1'] = str(len(self.time_axis)) + "D8.3"
        new_axes_header['TFORM2'] = str(len(self.freq_axis)) + "D8.3"

        self.axes_header = new_axes_header


CallistoSpectrogram._create.add(
    run_cls('from_range'),
    lambda cls, instrument, start, end: True,
    check=False
)

try:
    CallistoSpectrogram.create.__func__.__doc__ = (
        """ Create CallistoSpectrogram from given input dispatching to the
        appropriate from_* function.

    Possible signatures:

    """ + CallistoSpectrogram._create.generate_docs())
except AttributeError:
    CallistoSpectrogram.create.__func__.__doc__ = (
        """ Create CallistoSpectrogram from given input dispatching to the
        appropriate from_* function.

    Possible signatures:

    """ + CallistoSpectrogram._create.generate_docs())
