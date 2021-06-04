#! /usr/bin/python

######################################
#                                    #
# Program for collecting information #
# from a set of SLHA files           #
#                                    #
######################################
#
# Usage:
# ------
#     python harvest_slha_ew.py <output file> <root search dir> <file tag>
#
#
# Details:
# --------
# The program will only include files that have <file tag> as part of the filename.
# If <file tag> is set to '', all files in the search directory are included.
#
#

import os
import sys
import time
from modules import pyslha
from collections import OrderedDict
import Messages_and_functions_for_harvest as mfh

###########################
#  What data to collect?  #
###########################

# Should this really be ordered? Try to change
datadict = OrderedDict([])

# The element stored in the dictionary is defined by a pair (key,index).
# The key can itself be a tuple (n1,n2,...) if the SLHA block has multiple
# indices. The index is the column number of the sought entry on the line of the
# SLHA file counting from 0 and *including* the key entries.

##############################
#  Initial setup and checks  #
##############################

# set output prefix
outpref = sys.argv[0] + ' : '

# check input arguments:

global abs_p
global abs_m
if 'abs_p' in os.environ:
    abs_p = os.environ['abs_p']
    abs_p = mfh.str2bool(abs_p)
else:
    abs_p = False

if 'abs_m' in os.environ:
    abs_m = os.environ['abs_m']
    abs_m = mfh.str2bool(abs_m)
else:
    abs_m = False

if 'include' in os.environ:
    include = os.environ['include']
    include = include.split()
else:
    include = None
    print('Including all available elements for given parameters\n')

if include is not None:
    possible = ['P:', 'M:', 'O:', 'X:', 'CS:']
    if len(include) <= 1:
        mfh.include_fail()
        include = None

    i = 0
    cond = True
    while cond:
        if possible[i] in include:
            cond = False
        else:
            i += 1

        if i == len(possible):
            mfh.include_fail()
            include = None
            cond = False

if len(sys.argv) not in {4}:
    sys.stdout.write("%s Wrong number of input arguments.\n" % (outpref))
    sys.stdout.write("%s Usage:\n" % (outpref))
    sys.stdout.write(
        "%s   python harvest_slha_nimbus.py <output file> <root search dir> <file tag> \n" % (
            outpref))
    sys.exit()

# assign input arguments to variables
outfile = sys.argv[1]
searchdir = sys.argv[2]
filetag = sys.argv[3]


# outfile decides what parameters to harvest. To harvest all parameters write EwOnly_PMON_CS
# P: parameters
# M: Main masses, chargino and neutralino
# O: Other relevant masses
# X: Neutralino mixing matrix
# CS: Cross section data

#############################
#   Initialize dictionary   #
#############################

# Parameters for EWonly
def parameters_dict():
    abs = abs_p
    datadict_temp = OrderedDict([])
    datadict_temp['tanb'] = {'block': 'MINPAR', 'element': (3, 1), 'abs': abs}
    datadict_temp['M_1(MX)'] = {'block': 'EXTPAR', 'element': (1, 1), 'abs': abs}
    datadict_temp['M_2(MX)'] = {'block': 'EXTPAR', 'element': (2, 1), 'abs': abs}
    datadict_temp['mu(MX)'] = {'block': 'EXTPAR', 'element': (23, 1), 'abs': abs}

    # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered.
    mfh.filter_dict('P:', include, datadict, datadict_temp)
    return


# Chargino and neutralino masses
# TODO: check if we really want abs
def masses_char_neu_dict():
    abs = abs_m
    datadict_temp = OrderedDict([])
    datadict_temp['m1000022'] = {'block': 'MASS', 'element': (1000022, 1), 'abs': abs}
    datadict_temp['m1000023'] = {'block': 'MASS', 'element': (1000023, 1), 'abs': abs}
    datadict_temp['m1000024'] = {'block': 'MASS', 'element': (1000024, 1), 'abs': abs}
    datadict_temp['m1000025'] = {'block': 'MASS', 'element': (1000025, 1), 'abs': abs}
    datadict_temp['m1000035'] = {'block': 'MASS', 'element': (1000035, 1), 'abs': abs}
    datadict_temp['m1000037'] = {'block': 'MASS', 'element': (1000037, 1), 'abs': abs}

    # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered/and or new masses are fetched.
    mfh.filter_dict('M:', include, datadict, datadict_temp)
    return


# Other relevant masses
def masses_others_dict():
    abs = abs_m
    datadict_temp = OrderedDict([])
    datadict_temp['m1000021'] = {'block': 'MASS', 'element': (1000021, 1), 'abs': abs}
    datadict_temp['m1000004'] = {'block': 'MASS', 'element': (1000004, 1), 'abs': abs}
    datadict_temp['m1000003'] = {'block': 'MASS', 'element': (1000003, 1), 'abs': abs}
    datadict_temp['m1000001'] = {'block': 'MASS', 'element': (1000001, 1), 'abs': abs}
    datadict_temp['m1000002'] = {'block': 'MASS', 'element': (1000002, 1), 'abs': abs}
    datadict_temp['m2000002'] = {'block': 'MASS', 'element': (2000002, 1), 'abs': abs}
    datadict_temp['m2000001'] = {'block': 'MASS', 'element': (2000001, 1), 'abs': abs}
    datadict_temp['m2000003'] = {'block': 'MASS', 'element': (2000003, 1), 'abs': abs}
    datadict_temp['m2000004'] = {'block': 'MASS', 'element': (2000004, 1), 'abs': abs}

    # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered.
    mfh.filter_dict('O:', include, datadict, datadict_temp)
    return


# Neutralino mixing matrix
def mixing_neu_dict():
    datadict_temp = OrderedDict([])
    datadict_temp['nmix11'] = {'block': 'NMIX', 'element': ((1, 1), 2)}
    datadict_temp['nmix12'] = {'block': 'NMIX', 'element': ((1, 2), 2)}
    datadict_temp['nmix13'] = {'block': 'NMIX', 'element': ((1, 3), 2)}
    datadict_temp['nmix14'] = {'block': 'NMIX', 'element': ((1, 4), 2)}
    datadict_temp['nmix21'] = {'block': 'NMIX', 'element': ((2, 1), 2)}
    datadict_temp['nmix22'] = {'block': 'NMIX', 'element': ((2, 2), 2)}
    datadict_temp['nmix23'] = {'block': 'NMIX', 'element': ((2, 3), 2)}
    datadict_temp['nmix24'] = {'block': 'NMIX', 'element': ((2, 4), 2)}

    # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered.
    mfh.filter_dict('X:', include, datadict, datadict_temp)
    return

def mixing_u_dict():
    datadict_temp = OrderedDict([])
    datadict_temp['umix11'] = {'block': 'UMIX', 'element': ((1, 1), 2)}
    datadict_temp['umix12'] = {'block': 'UMIX', 'element': ((1, 2), 2)}
    datadict_temp['umix21'] = {'block': 'UMIX', 'element': ((2, 1), 2)}
    datadict_temp['umix22'] = {'block': 'UMIX', 'element': ((2, 2), 2)}

    # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered.
    mfh.filter_dict('U:', include, datadict, datadict_temp)
    return

def mixing_v_dict():
    datadict_temp = OrderedDict([])
    datadict_temp['vmix11'] = {'block': 'VMIX', 'element': ((1, 1), 2)}
    datadict_temp['vmix12'] = {'block': 'VMIX', 'element': ((1, 2), 2)}
    datadict_temp['vmix21'] = {'block': 'VMIX', 'element': ((2, 1), 2)}
    datadict_temp['vmix22'] = {'block': 'VMIX', 'element': ((2, 2), 2)}

    # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered.
    mfh.filter_dict('V:', include, datadict, datadict_temp)
    return

# Chargino mixing matrix

# Reading Prospino output

def cross_sections_dict():
    # com_energies = [7000, 8000, 13000, 14000]
    com_energies = [13000]
    for en in com_energies:
        try:
            datadict_temp = OrderedDict([])
            # chi01-chi01 cross section and prospino numerical error
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_1_relerr'] = {'block': 'PROSPINO_OUTPUT',
                                                                             'element': ((1000022, 1000022, en), 6)}
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_1'] = {'block': 'PROSPINO_OUTPUT',
                                                                      'element': ((1000022, 1000022, en), 7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_05'] = {'block': 'PROSPINO_OUTPUT',
                                                                       'element': ((1000022, 1000022, en), 8)}
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_2'] = {'block': 'PROSPINO_OUTPUT',
                                                                      'element': ((1000022, 1000022, en), 9)}
            # PDF variation
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_pdf'] = {'block': 'PROSPINO_OUTPUT',
                                                                        'element': ((1000022, 1000022, en), 10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_aup'] = {'block': 'PROSPINO_OUTPUT',
                                                                        'element': ((1000022, 1000022, en), 11)}
            datadict_temp['1000022_1000022_' + str(en) + '_NLO_adn'] = {'block': 'PROSPINO_OUTPUT',
                                                                        'element': ((1000022, 1000022, en), 12)}

            # chi02-chi02 cross section and prospino numerical error
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_1_relerr'] = {'block': 'PROSPINO_OUTPUT',
                                                                             'element': ((1000023, 1000023, en), 6)}
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_1'] = {'block': 'PROSPINO_OUTPUT',
                                                                      'element': ((1000023, 1000023, en), 7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_05'] = {'block': 'PROSPINO_OUTPUT',
                                                                       'element': ((1000023, 1000023, en), 8)}
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_2'] = {'block': 'PROSPINO_OUTPUT',
                                                                      'element': ((1000023, 1000023, en), 9)}
            # PDF variation
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_pdf'] = {'block': 'PROSPINO_OUTPUT',
                                                                        'element': ((1000023, 1000023, en), 10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_aup'] = {'block': 'PROSPINO_OUTPUT',
                                                                        'element': ((1000023, 1000023, en), 11)}
            datadict_temp['1000023_1000023_' + str(en) + '_NLO_adn'] = {'block': 'PROSPINO_OUTPUT',
                                                                        'element': ((1000023, 1000023, en), 12)}

            # Appends datadict_temp to the global datadict. If "include" is specified datadict_temp is filtered.
            mfh.filter_dict('CS:', include, datadict, datadict_temp, str(en))
        except:
            pass

    return


def datadict_init():
    print('')
    outfile_list = outfile.split("_")
    global include

    if len(outfile_list) in {2, 3}:
        keys = list(outfile_list[1])

    else:
        sys.stdout.write(
            "%s Input variables not specified in filename, underscore not used before input variables or more than two underscores used. Loading all available to file> \n" % (
                outpref))
        parameters_dict()
        masses_char_neu_dict()
        masses_others_dict()
        mixing_neu_dict()
        mixing_v_dict()
        mixing_u_dict()
        cross_sections_dict()
        return

    # removing duplicates
    keys = list(dict.fromkeys(keys))
    for i, key in enumerate(keys):
        if key == 'P':  # Parameters
            sys.stdout.write("%s %s recognized. Harvesting 'parameters'. \n" % (outpref, key))
            parameters_dict()

        elif key == 'M':  # Masses of neutralino and chargino
            sys.stdout.write("%s %s recognized. Harvesting 'neutralino and chargino masses'. \n" % (outpref, key))
            masses_char_neu_dict()

        elif key == 'O':  # Other relevant masses
            sys.stdout.write("%s %s recognized. Harvesting other relevant masses. \n" % (outpref, key))
            masses_others_dict()

        elif key == 'X':  # Neutralino mixing angle
            sys.stdout.write("%s %s recognized. Harvesting 'neutralino mixing'. \n" % (outpref, key))
            mixing_neu_dict()

        elif key == 'U':  # Chargino mixing angle
            sys.stdout.write("%s %s recognized. Harvesting 'chargino mixing'. \n" % (outpref, key))
            mixing_u_dict()

        elif key == 'V':  # Chargino mixing angle
            sys.stdout.write("%s %s recognized. Harvesting 'chargino mixing'. \n" % (outpref, key))
            mixing_v_dict()

        elif key == 'C' and keys[i + 1] == 'S':  # Cross section data
            sys.stdout.write("%s %s recognized. Harvesting 'cross section data'.\n" % (outpref, 'CS'))
            cross_sections_dict()

        elif key == 'S':
            pass

        else:
            sys.stdout.write("%s %s not recognized. \n" % (outpref, key))
    return


datadict_init()

#####################################
#  File search and data collection  #
#####################################


inputfiles = []
for root, dirnames, filenames in os.walk(searchdir):
    # for filename in fnmatch.filter(filenames, filetag):
    for filename in filenames:

        if filetag not in filename:
            continue

        inputfiles.append(os.path.join(root, filename))

# print file count
n_files_total = len(inputfiles)
sys.stdout.write("%s Found %d input files.\n" % (outpref, n_files_total))

# sort the file list
inputfiles.sort()

# print info
sys.stdout.write("%s Collecting data...\n" % (outpref))

# open outfile for output
f = open(outfile, "w")

# add tag for filename columns

max_path_lenght = len(max(filenames))
tagline = ''
tag = 'file'
tagline += (tag + ' ' * (max_path_lenght - 21))

for i, tag in enumerate(datadict.keys()):  # IH: Remove the tag for masses and xsections
    n = i + 2
    complete_tag = tag  # IH
    # complete_tag = '%i.%s' % (n,tag)
    tagline += (complete_tag + ' ' * (max(1, 14 - len(complete_tag))))

tagline += '\n'

f.write(tagline)

# collect data from each file and write to outfile
lines = ''
for count, filepath in enumerate(inputfiles, 1):
    slha_dict = pyslha.readSLHAFile(filepath)

    datalist = []

    accepted_file = True
    number_of_unaccepted_sxections = 0  # IH
    for key in datadict.keys():
        if 'block' in datadict[key].keys():
            if not datadict[key]['block'] in slha_dict.keys():
                accepted_file = False
                sys.stdout.write("%s Problem encountered when looking for block %s in file %s. File ignored.\n" % (
                    outpref, datadict[key]['block'], filepath))
                break
            # print(datadict[key]['block'])
            if ('abs' in datadict[key].keys()) and (datadict[key]['abs'] == True):
                datalist.append(abs(slha_dict[datadict[key]['block']].getElement(*datadict[key]['element'])))
            else:
                try:
                    slha_dict[datadict[key]['block']].getElement(*datadict[key]['element'])  # IH
                    datalist.append(slha_dict[datadict[key]['block']].getElement(*datadict[key]['element']))
                except:
                    number_of_unaccepted_sxections += 1
                    datalist.append(-1)

        elif 'decay' in datadict[key].keys():
            if not datadict[key]['decay'] in slha_dict.keys():
                accepted_file = False
                sys.stdout.write("%s Problem encountered when looking for decay %s in file %s. File ignored.\n" % (
                    outpref, datadict[key]['decay'], filepath))
                break

            datalist.append(slha_dict[datadict[key]['decay']].getBR(list(datadict[key]['element'])))

        # got_it = True
        # try:
        #     datalist.append( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) )

        # except Exception as err:
        #     sys.stdout.write("%s %s \n" % (outpref, err.message))
        #     sys.stdout.write("%s Problem encountered when harvesting data from file %s. File ignored.\n" % (outpref, filepath))
        #     continue

    if not accepted_file:
        continue

    datatuple = tuple(datalist)

    # Chop filepath for printing
    file = filepath.split('/')[-1]

    # Make lines for printing
    lines += ('%s' + ' ' * (max_path_lenght - len(filepath) + 2)) % file
    lines += ((' ' + '% .5e' + ' ' * 2) * len(datatuple) + '\n') % datatuple
    # added ' ' + in front because filetag and -1 got merged.

    # write to file once per 1000 files read
    if count % 1000 == 0:
        sys.stdout.write("%s %d of %d files read\n" % (outpref, count, n_files_total))
        f.write(lines)
        lines = ''

# Remove final endline and write remaining lines
lines = lines.rstrip('\n')
f.write(lines)

##############
#  Finalise  #
##############

# output info
sys.stdout.write("%s ...done!\n" % (outpref))

# close the outfile
f.close()

# print some output
sys.stdout.write("%s Summary written to the file %s \n" % (outpref, outfile))
