import re
import os
import time

import pandas as pd
from os.path import relpath

from .sources import CallistoSpectrogram
from .sources.callisto import query

from sunpy.util.net import download_file
from sunpy.time import parse_time


def preprocessing_txt(data):
    """
    Preprocessing dataframe with the Callisto Flare Notation
    """

    data['date'] = data['date'].apply(lambda x: '{0:0>8}'.format(x))
    data['end'] = data['end'].apply(lambda x: '{0:0>6}'.format(x))
    data['start'] = data['start'].apply(lambda x: '{0:0>6}'.format(x))
    data['lower'] = data['lower'].astype(str).map(lambda x: x.rstrip('xX'))
    data['upper'] = data['upper'].astype(str).map(lambda x: x.rstrip('xX'))
    data['remarks'] = data['remarks'].astype(str)
    return data


def microseconds_clean(data):
    """
    Second step for preprocessing the dataframe
    """

    for index, elemen in data.iterrows():
        if data.loc[index]['start'] == '000nan':
            continue

        string = data.loc[index]['start']
        new_start = string[0:4] + str(int(float(string[4:6]) * 60))
        data['start'].at[index] = new_start

        string = data.loc[index]['end']
        new_start = string[0:4] + str(int(float(string[4:6]) * 60))
        data['end'].at[index] = new_start

    return data


def date_cleaner(date):
    """
    Cleans date string after modifying previous years
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
    """

    time = str(time)
    long = len(time)
    new = ''
    for x in range(long): new = new + time[x]
    return re.sub(r'((?:(?=(1|.))\2){2})(?!$)', r'\1:', new)


# Sets the previous methods together
def range_Generator(row_num, dataframe):
    """
    Generates the required strings to work with CallistoSpectrogram class
    """

    row = dataframe.loc[row_num]
    instrument = creator_instrument(row['lower'], row['upper'])
    year = creator_date(row['date'])
    start = creator_time(row['start'])
    end = creator_time(row['end'])
    return instrument, year, start, end

def range_iGenerator(row_num, dataframe):
    """
    Generates the required strings to work with CallistoSpectrogram class, uses iloc instead of loc

    Args:
        row_num: index of the row
        dataframe:  pandas dataframe
    Returns:
        Modified time to use with standard Time Libraries
    """
    row = dataframe.iloc[row_num]
    instrument = creator_instrument(row['lower'], row['upper'])
    year = creator_date(row['date'])
    start = creator_time(row['start'])
    end = creator_time(row['end'])
    return instrument, year, start, end


# Directory methods
def directorySubtypeGenerator(folder, flareType, subtype):
    """
    Generates Directories based in the subtype and type of flares
    """

    if os.path.isdir('./{}/{}/{}'.format(folder, flareType, subtype)) == False:
        os.makedirs('./{}/{}/{}'.format(folder, flareType, subtype))
        return os.path.realpath('./{}/{}/{}'.format(folder, flareType, subtype))
    else:
        return os.path.realpath('./{}/{}/{}'.format(folder, flareType, subtype))


def directoryFlaretype(folder, flareType):
    """
    Generates Directories based ONLY in the type of flares
    """

    if os.path.isdir('./{}/{}'.format(folder, flareType)) == False:
        os.makedirs('./{}/{}'.format(folder, flareType))
        return os.path.realpath('./{}/{}'.format(folder, flareType))
    else:
        return os.path.realpath('./{}/{}'.format(folder, flareType))



def dir_Gen(row_num, dataframe):
    """
    Gets the directory of the data from the remarks column
    """

    row = dataframe.loc[row_num]
    directions = row['remarks']

    directionsList = [x.strip() for x in directions.split(',')[:-1]]

    return directionsList

def dir_iGen(row_num, dataframe):
    """
    Gets the directory of the data using the actual position of the flare in the list

    Args:
        row_num: index of the row
        dataframe: pandas dataframe
    Returns:
        directions of files saved in the remarks column
    """

    row = dataframe.iloc[row_num]
    directions = row['remarks']

    directionsList = [x.strip() for x in directions.split(',')[:-1]]

    return directionsList


def Callisto_dir_flare(row_num, dataframe, show_details=False, show_urls=False):
    """
    Peek a CallistoSpectrogram from a row of the Already Downloaded dataframe
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

def Callisto_idir_flare(row_num, dataframe, show_details=False, show_urls=False):
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
        row = dataframe.iloc[row_num]
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
    """
    Iterates over a dataframe using remarks_Cleaners
    """

    clean_directions = pd.DataFrame(columns = data.columns)
    exceptions_frame = pd.DataFrame(columns = data.columns)
    for index, row in data.iterrows():
        clean_directions, exceptions_frame = remarks_Cleaners(index, data, clean_directions, exceptions_frame)
    return clean_directions, exceptions_frame


#Main Method
def e_Callisto_burst_downloader(data, sort=False, folder="e-Callisto_Flares", exist=False):
    """
    Download a set of burst based on a dataframe with the Callisto-Notation
    """

    start = time.time()
    data = preprocessing_txt(data)
    data = microseconds_clean(data)
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
    Joins the slice data into time axis per Flare
    """

    start = time.time()
    os.makedirs('./{}'.format(folder))
    joined = pd.DataFrame(columns=dataframe.columns)
    special = pd.DataFrame(columns=dataframe.columns)

    for index, elem in dataframe.iterrows():

        directions = dir_Gen(index, dataframe)
        name = os.path.basename(directions[0])
        try:
            bursts_here = CallistoSpectrogram.from_files(directions)
        except ValueError:
            print("Damage file found at \n" + str(dataframe.loc[index]))
            special = special.append(dataframe.loc[index])
            continue
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
    print("\nJoined after----- " + str(end - start) + " secs\n")
    return joined, special


# Peek a flare from Callisto Database
def Callisto_flare(row_num, dataframe, show_url=False):
    """
    Peek a flare from a row of a given dataframe
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

def Callisto_simple_flare(index, dataframe):
    """
    Peeks a spectrogram

    Args:
        index: position of the elem in the dataframe
        dataframe: pandas dataframe
    Returns:
        Callistopectrogram obejct
    """
    Spectra = CallistoSpectrogram.read(dataframe.loc[index]['remarks'])
    Spectra.peek()
    return Spectra

def Callisto_simple_iflare(index, dataframe):
    """
    Peeks a spectrogram

    Args:
        index: INDEX of the elem in the dataframe
        dataframe: pandas dataframe
    Returns:
        Callistopectrogram obejct
    """
    Spectra = CallistoSpectrogram.read(dataframe.iloc[index]['remarks'])
    Spectra.peek()
    return Spectra

def preview(dataframe, show_details = True):
    """
    Show a preview of a whole dataframe, to limit the amount of examples, slice the dataframe
      Args:
          dataframe: pandas dataframe
          show_details: Boolean, if true shows information about the flare
      Returns:
          CallistoSpectrogram objects
      """
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
