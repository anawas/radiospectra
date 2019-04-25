import re
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from os.path import relpath
from astropy.io import fits
from .sources import CallistoSpectrogram
from .sources.callisto import query

from sunpy.util.net import download_file
from sunpy.time import parse_time


def preprocessing_txt(data):
    """
    Preprocessing dataframe with the Callisto Flare Notation

    Args:
      data: Pandas dataframe
    Returns:
      Preprocessed dataframe
    """
    data['date'] = data['date'].apply(lambda x: '{0:0>8}'.format(x))
    data['end'] = data['end'].apply(lambda x: '{0:0>6}'.format(x))
    data['start'] = data['start'].apply(lambda x: '{0:0>6}'.format(x))
    data['lower'] = data['lower'].astype(str).map(lambda x: x.rstrip('xX'))
    data['upper'] = data['upper'].astype(str).map(lambda x: x.rstrip('xX'))
    # Sanity preserver
    # data['lower'] = data['lower'].astype(str).str.replace('\D-', '')
    # data['upper'] = data['upper'].astype(str).str.replace('\D+', '')
    data['remarks'] = data['remarks'].astype(str)
    return data


def date_cleaner(date):
    """
    Cleans date string after modifing previos years
    """
    date = str(date)
    long = len(date)-2
    new = ''
    for x in range(long): new = new + date[x]
    return new


#It may not work if the given frequencies are out of range
def creator_instrument(lower, upper):
    """
    Generates the aproximated instrument string based in the frequencies, to use directly with CallistoSpectrogram

    Args:
        Lower: Lower intensity of the flare
        Upper: Upper intensity of the flare

    Returns:
        Name of the Instrument based on the frequencies analysed
    """

    lower = int(lower)
    upper = int(upper)
    if lower>=1200 and upper<=1800 : return "BLEN5M"
    if lower>=110 and upper<=870 : return "BLEN7M"
    #The Upper value is set to 110 in order to download a bigger wide of flares
    if lower>=20 and upper<=110 : return "BLENSW"

def creator_date(date):
    """
    Creates the date format to use directly with CallistoSpectrogram

    Args:
        date: date from dataframe
    Returns:
        Modified date to use with standard Time Libraries
    """
    date = date_cleaner(date)
    date = re.sub(r'((?:(?=(1|.))\2){2})(?!$)', r'\1/', date)
    if int(date[0] + date[1]) > 50:
        date = '19' + date
    else:
        date = '20' + date
    return date


def creator_time(time):
    """
    Creates the time format to use directly with CallistoSpectrogram

    Args:
        time: time from dataframe
    Returns:
        Modified time to use with standard Time Libraries
    """
    time = str(time)
    microT = ':0' + time[5]
    long = len(time) - 2
    new = ''
    for x in range(long): new = new + time[x]
    return re.sub(r'((?:(?=(1|.))\2){2})(?!$)', r'\1:', new) + microT


# Sets the previous methods together
def range_Generator(row_num, dataframe):
    """
    Generates the required strings to work with CallistoSpectrogram class

    Args:
        row_num: index of the row
        dataframe:  pandas dataframe
    Returns:
        Modified time to use with standard Time Libraries
    """
    row = dataframe.loc[row_num]
    instrument = creator_instrument(row['lower'], row['upper'])
    year = creator_date(row['date'])
    start = creator_time(row['start'])
    end = creator_time(row['end'])
    return instrument, year, start, end


# Peek a flare from Callisto Database
def Callisto_flare(row_num, dataframe, show_url=False):
    """
    Peek a flare from a row of a given dataframe

    Args:
        row_num: index of the row
        dataframe: pandas dataframe
    Returns:
        CallistoSpectrogram object
    """
    row = dataframe.loc[row_num]
    instrument, year, start, end = range_Generator(row_num, dataframe)
    print(instrument)
    print('  ' + row['lower'], row['upper'])
    print(creator_date(row['date']))
    print(start)
    print(end)
    if show_url:
        startQ = parse_time(year + ' ' + start)
        endQ = parse_time(year + ' ' + end)
        urls = query(startQ, endQ, [instrument])
        for url in urls:
            print(url)

    Spectra = CallistoSpectrogram.from_range(instrument, year + ' ' + start, year + ' ' + end)
    Spectra.peek()
    return Spectra


# Directory methods
def directorySubtypeGenerator(folder, flareType, subtype):
    """
    Generates Directories based in the subtype and type of flares

    Args:
        folder: root directory
        flareType: type of flare from dataframe
        subtype: subtype of flare from dataframe
    Returns:
        path to new flare directory
    """
    if os.path.isdir('./{}/{}/{}'.format(folder, flareType, subtype)) == False:
        os.makedirs('./{}/{}/{}'.format(folder, flareType, subtype))
        return os.path.realpath('./{}/{}/{}'.format(folder, flareType, subtype))
    else:
        return os.path.realpath('./{}/{}/{}'.format(folder, flareType, subtype))


def directoryFlaretype(folder, flareType):
    """
    Generates Directories based ONLY in the type of flares

    Args:
        folder: root directory
        flareType: type of flare from dataframe
    Returns:
        path to new flares directory
    """
    if os.path.isdir('./{}/{}'.format(folder, flareType)) == False:
        os.makedirs('./{}/{}'.format(folder, flareType))
        return os.path.realpath('./{}/{}'.format(folder, flareType))
    else:
        return os.path.realpath('./{}/{}'.format(folder, flareType))



def dir_Gen(row_num, dataframe):
    """
    Gets the directory of the data from the remarks column

    Args:
        row_num: index of the row
        dataframe: pandas dataframe
    Returns:
        directions of files saved in the remarks column
    """

    row = dataframe.loc[row_num]
    directions = row['remarks']

    directionsList = [x.strip() for x in directions.split(',')[:-1]]

    return directionsList


def Callisto_dir_flare(row_num, dataframe, show_details=False, show_urls=False):
    """
    Peek a CallistoSpectrogram from a row of the Already Downloaded dataframe

    Args:
        row_num: index of the row
        dataframe: pandas dataframe
        show_details: A boolean to decide if show flare details
        show_urls: A boolean to decide if show flare urls
    Returns:
        List of CallistoSpectrogram directory paths

    """
    if show_details:
        row = dataframe.loc[row_num]
        instrument, year, start, end = range_Generator(row_num, dataframe)
        print(instrument)
        print('  ' + row['lower'], row['upper'])
        print(creator_date(row['date']))
        print(start)
        print(end)

    if show_urls:
        startQ = parse_time(year + ' ' + start)
        endQ = parse_time(year + ' ' + end)
        urls = query(startQ, endQ, [instrument])
        for url in urls:
            print(url)

    Gen = dir_Gen(row_num, dataframe)
    print("----------------plots----------------")
    for elem in Gen:
        CallistoSpectrogram.read(elem).peek()
    return Gen


# Downloader Core Methods
def e_Callisto_exceptionSeeker(row_num, dataframe, new_frame, exceptions_fr, folder, sort=False):
    """
    Returns new_frame and exceptions_fr also download the files of the new frame
    """
    try:

        instrument, year, start, end = range_Generator(row_num, dataframe)
        start = parse_time(year + ' ' + start)
        end = parse_time(year + ' ' + end)
        urls = query(start, end, [instrument])

        if instrument == None:
            raise Exception

        row = dataframe.loc[row_num]
        flareType = row['class']
        subtype = row['sub']

        if sort == True:
            directory = directorySubtypeGenerator(folder, flareType, subtype)
        else:
            #directory = os.path.realpath('./{}'.format(folder))
            directory = directoryFlaretype(folder, flareType)

        dirlist = ''
        for url in urls:
            dire = download_file(url, directory)
            dirlist = dirlist + relpath(dire) + ','

        new_frame = new_frame.append(dataframe.loc[row_num])
        new_frame.at[row_num, 'remarks'] = dirlist
        return new_frame, exceptions_fr
    except:
        exceptions_fr = exceptions_fr.append(dataframe.loc[row_num])
        return new_frame, exceptions_fr

def remarks_Cleaners(row_num, dataframe, new_frame, exceptions_fr):
    """
    Cleans remarks column from an already downloaded dataframe
    """
    row = dataframe.loc[row_num]
    directions = row['remarks']

    if directions != '':
        new_frame = new_frame.append(dataframe.loc[row_num])
        return new_frame, exceptions_fr
    else:
        exceptions_fr = exceptions_fr.append(dataframe.loc[row_num])
        return new_frame, exceptions_fr

def iter_remarks_Cleaners(data):
    """Iterates over a dataframe using remarks_Cleaners"""
    clean_directions = pd.DataFrame(columns = data.columns)
    exceptions_frame = pd.DataFrame(columns = data.columns)
    for index, row in data.iterrows():
        clean_directions, exceptions_frame = remarks_Cleaners(index, data, clean_directions, exceptions_frame)
    return clean_directions, exceptions_frame


#Main Method
def e_Callisto_burst_downloader(data, sort=False, folder="e-Callisto_Flares", exist=False):
    """
    Download a set of burst based on a dataframe with the Callisto-Notation

    Args:
        data: pandas dataframe
        sort: Python boolean. If 'True', it creates a subset of folders for the flares subtypes
        folder: name or path of the folder where the flares will be downloaded
        exist = Python boolean. If 'True' overwrites the path (if already exist)
    Returns:
        rclean: Pandas dataframe. Contains the information of all the already downloaded flares,
            as well as the paths of their respective FITS files.
        exceptions_frame: Pandas dataframe. Contains information about files that could not be downloaded
    """
    start = time.time()
    data = preprocessing_txt(data)
    os.makedirs('./{}'.format(folder), exist_ok=exist)
    clean_directions = pd.DataFrame(columns=data.columns)
    exceptions_frame = pd.DataFrame(columns=data.columns)
    for index, row in data.iterrows():
        clean_directions, exceptions_frame = e_Callisto_exceptionSeeker(index, data, clean_directions, exceptions_frame,
                                                                        folder, sort)
    rclean_test, rexcept_test = iter_remarks_Cleaners(clean_directions)
    exceptions_frame = exceptions_frame.append(rexcept_test)
    end = time.time()
    print("Download completed in ----- " + str(end - start) + " secs")
    return rclean_test, exceptions_frame


#Joining methods
def e_Callisto_Burst_simplifier(dataframe, folder, sort=False):
    """
    Returns a new data frame with the direction of the joined data set
    """
    start = time.time()
    os.makedirs('./{}'.format(folder))
    joined = pd.DataFrame(columns=dataframe.columns)

    for index, elem in dataframe.iterrows():

        directions = dir_Gen(index, dataframe)
        name = os.path.basename(directions[0])

        bursts_here = CallistoSpectrogram.from_files(directions)

        row = dataframe.loc[index]

        if sort == True:
            flareType = row['class']
            subtype = row['sub']
            directory = directorySubtypeGenerator(folder, flareType, subtype)
        else:
            flareType = row['class']
            directory = directoryFlaretype(folder, flareType)

        path = directory + '\\{}'.format(name)
        CallistoSpectrogram.join_many(bursts_here).save(relpath(path))
        joined = joined.append(dataframe.loc[index])
        joined.at[index, 'remarks'] = relpath(path)
    end = time.time()
    print("Joined after----- " + str(end - start) + " secs")
    return joined

def Callisto_simple_flare(index, dataframe):
    Spectra = CallistoSpectrogram.read(dataframe.loc[index]['remarks'])
    Spectra.peek()
    return Spectra

def preview(dataframe, show_details = True):
    for index, elem in dataframe.iterrows():
        row = dataframe.loc[index]
        instrument, year, start, end = range_Generator(index, dataframe)
        if show_details:
            print("Type "+str(row['class']))
            print('  Range ' + row['lower'], row['upper'])
            print(start)
            print(end)
            print(creator_date(row['date']))
        Callisto_simple_flare(index, dataframe)
