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
#     python harvest_slha_strong.py <output file> <root search dir> <file tag>
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
from modules import pyslha
from collections import OrderedDict


###########################
#  What data to collect?  #
###########################


### SETTINGS #############################################
#com_energies = [7000, 8000, 13000, 14000]
com_energies = [13000]
processes = ['gg','ss','sg','sb','bb','tb']
# processes = ['bb', 'tb']
processes = ['ss']
##########################################################

# Initialize dictionary specifying the data to be gathered from each SLHA file
datadict = OrderedDict([])

# The element stored in the dictionary is defined by a pair (key,index).
# The key can itself be a tuple (n1,n2,...) if the SLHA block has multiple
# indices. The index is the column number of the sought entry on the line of the
# SLHA file counting from 0 and *including* the key entries.

# Sparticle masses
datadict["m1000021"] = {"block": "MASS", "element": (1000021, 1), "abs": True}
datadict["m1000004"] = {"block": "MASS", "element": (1000004, 1), "abs": True}
datadict["m1000003"] = {"block": "MASS", "element": (1000003, 1), "abs": True}
datadict["m1000001"] = {"block": "MASS", "element": (1000001, 1), "abs": True}
datadict["m1000002"] = {"block": "MASS", "element": (1000002, 1), "abs": True}
datadict["m2000002"] = {"block": "MASS", "element": (2000002, 1), "abs": True}
datadict["m2000001"] = {"block": "MASS", "element": (2000001, 1), "abs": True}
datadict["m2000003"] = {"block": "MASS", "element": (2000003, 1), "abs": True}
datadict["m2000004"] = {"block": "MASS", "element": (2000004, 1), "abs": True}
datadict["m1000005"] = {"block": "MASS", "element": (1000005, 1), "abs": True}
datadict["m2000005"] = {"block": "MASS", "element": (2000005, 1), "abs": True}
datadict["m1000006"] = {"block": "MASS", "element": (1000006, 1), "abs": True}
datadict["m2000006"] = {"block": "MASS", "element": (2000006, 1), "abs": True}

# Sbottom and stop mixing angles
datadict["sbotmix11"] = {"block": "SBOTMIX", "element": ((1, 1), 2)}
datadict["stopmix11"] = {"block": "STOPMIX", "element": ((1, 1), 2)}


# squarks = ['2000004', '2000003', '2000002', '2000001', '1000004', '1000003', '1000002', '1000001']
squarks = ["1000004", "1000003", "1000001", "1000002", "2000002", "2000001", "2000003", "2000004"]
gluino = "1000021"


# Read Prospino output as a loop over wanted energies and processes
for en in com_energies:
    for proc in processes:
        # gluino-gluino cross section
        if proc == 'gg':
            namebase = "1000021_1000021_"
            datadict["1000021_1000021_" + str(en) + "_NLO_1_relerr"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 6)}
            datadict["1000021_1000021_" + str(en) + "_NLO_1"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict["1000021_1000021_" + str(en) + "_NLO_05"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 8)}
            datadict["1000021_1000021_" + str(en) + "_NLO_2"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 9)}
            # PDF variation
            datadict["1000021_1000021_" + str(en) + "_NLO_pdf"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict["1000021_1000021_" + str(en) + "_NLO_aup"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 11)}
            datadict["1000021_1000021_" + str(en) + "_NLO_adn"] = {"block": "PROSPINO_OUTPUT", "element": ((1000021, 1000021, en), 12)}

        # squark-squark cross sections
        if proc == 'ss':
            for i in range(len(squarks)):
                for j in range(i, len(squarks)):

                    # Put the squark with the lowest PID at the beginning
                    if squarks[i] >= squarks[j]:
                        namebase = str(squarks[j]) + "_" + str(squarks[i])
                    elif squarks[i] < squarks[j]:
                        namebase = str(squarks[i]) + "_" + str(squarks[j])

                    name = namebase + "_" + str(en) + "_NLO_1"
                    name_err = namebase + "_" + str(en) + "_NLO_1_relerr"
                    name_05 = namebase + "_" + str(en) + "_NLO_05"
                    name_2 = namebase + "_" + str(en) + "_NLO_2"
                    name_pdf = namebase + "_" + str(en) + "_NLO_pdf"
                    name_aup = namebase + "_" + str(en) + "_NLO_aup"
                    name_adn = namebase + "_" + str(en) + "_NLO_adn"

                    datadict[name_err] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 6)}
                    datadict[name] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 7)}
                    datadict[name_05] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 8)}
                    datadict[name_2] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 9)}
                    datadict[name_pdf] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 10)}
                    datadict[name_aup] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 11)}
                    datadict[name_adn] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), int(squarks[j]), en), 12)}

        # squark-gluino cross sections
        if proc == 'sg':
            for i in range(len(squarks)):

                if squarks[i] < gluino:
                    namebase = str(squarks[i]) + "_" + gluino
                else:
                    namebase = gluino + "_" + str(squarks[i])

                name = namebase + "_" + str(en) + "_NLO_1"
                name_err = namebase + "_" + str(en) + "_NLO_1_relerr"
                name_05 = namebase + "_" + str(en) + "_NLO_05"
                name_2 = namebase + "_" + str(en) + "_NLO_2"
                name_pdf = namebase + "_" + str(en) + "_NLO_pdf"
                name_aup = namebase + "_" + str(en) + "_NLO_aup"
                name_adn = namebase + "_" + str(en) + "_NLO_adn"

                datadict[name_err] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 6)}
                datadict[name] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 7)}
                datadict[name_05] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 8)}
                datadict[name_2] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 9)}
                datadict[name_pdf] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 10)}
                datadict[name_aup] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 11)}
                datadict[name_adn] = {"block": "PROSPINO_OUTPUT", "element": ((int(1000021), int(squarks[i]), en), 12)}

        # squark-antisquark cross sections
        if proc == 'sb':
            for i in range(len(squarks)):
                for j in range(i, len(squarks)):

                    if squarks[i] >= squarks[j]:
                        namebase = "-" + str(squarks[i]) + "_" + str(squarks[j])
                    elif squarks[i] < squarks[j]:
                        namebase = "-" + str(squarks[j]) + "_" + str(squarks[i])

                    name = namebase + "_" + str(en) + "_NLO_1"
                    name_err = namebase + "_" + str(en) + "_NLO_1_relerr"
                    name_05 = namebase + "_" + str(en) + "_NLO_05"
                    name_2 = namebase + "_" + str(en) + "_NLO_2"
                    name_pdf = namebase + "_" + str(en) + "_NLO_pdf"
                    name_aup = namebase + "_" + str(en) + "_NLO_aup"
                    name_adn = namebase + "_" + str(en) + "_NLO_adn"

                    datadict[name_err] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 6)}
                    datadict[name] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 7)}
                    datadict[name_05] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 8)}
                    datadict[name_2] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 9)}
                    datadict[name_pdf] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 10)}
                    datadict[name_aup] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 11)}
                    datadict[name_adn] = {"block": "PROSPINO_OUTPUT", "element": ((int(squarks[i]), -int(squarks[j]), en), 12)}

        # sbottom pair production cross sections
        if proc == 'bb':
            # Main result
            datadict["-1000005_1000005_" + str(en) + "_NLO_1_relerr"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 6)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_1_relerr"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 6)}
            datadict["-1000005_1000005_" + str(en) + "_NLO_1"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 7)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_1"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict["-1000005_1000005_" + str(en) + "_NLO_05"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 8)}
            datadict["-1000005_1000005_" + str(en) + "_NLO_2"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 9)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_05"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 8)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_2"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 9)}
            # PDF variation
            datadict["-1000005_1000005_" + str(en) + "_NLO_pdf"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 10)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_pdf"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict["-1000005_1000005_" + str(en) + "_NLO_aup"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 11)}
            datadict["-1000005_1000005_" + str(en) + "_NLO_adn"] = {"block": "PROSPINO_OUTPUT", "element": ((1000005, -1000005, en), 12)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_aup"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 11)}
            datadict["-2000005_2000005_" + str(en) + "_NLO_adn"] = {"block": "PROSPINO_OUTPUT", "element": ((2000005, -2000005, en), 12)}

        # stop pair production cross sections
        if proc == 'tb':
            # Main result
            datadict["-1000006_1000006_" + str(en) + "_NLO_1_relerr"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 6)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_1_relerr"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 6)}
            datadict["-1000006_1000006_" + str(en) + "_NLO_1"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 7)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_1"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict["-1000006_1000006_" + str(en) + "_NLO_05"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 8)}
            datadict["-1000006_1000006_" + str(en) + "_NLO_2"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 9)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_05"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 8)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_2"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 9)}
            # PDF variation
            datadict["-1000006_1000006_" + str(en) + "_NLO_pdf"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 10)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_pdf"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict["-1000006_1000006_" + str(en) + "_NLO_aup"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 11)}
            datadict["-1000006_1000006_" + str(en) + "_NLO_adn"] = {"block": "PROSPINO_OUTPUT", "element": ((1000006, -1000006, en), 12)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_aup"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 11)}
            datadict["-2000006_2000006_" + str(en) + "_NLO_adn"] = {"block": "PROSPINO_OUTPUT", "element": ((2000006, -2000006, en), 12)}

##############################
#  Initial setup and checks  #
##############################

# set output prefix
outpref = sys.argv[0] + " : "

# check input arguments:
if len(sys.argv) != 4:
    sys.stdout.write("%s Wrong number of input arguments.\n" % (outpref))
    sys.stdout.write("%s Usage:\n" % (outpref))
    sys.stdout.write("%s   python harvest_slha_nimbus.py <output file> <root search dir> <file tag>\n" % (outpref))
    sys.exit()

# assign input arguments to variables
outfile = sys.argv[1]
searchdir = sys.argv[2]
filetag = sys.argv[3]


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
max_path_length = len(max(filenames))
tagline = ""
tag = "file"
tagline += tag + " " * (max_path_length - 4)

for i, tag in enumerate(datadict.keys()):  # IH: Remove the tag for masses and xsections
    n = i + 2
    complete_tag = tag  # IH
    # complete_tag = '%i.%s' % (n,tag)
    tagline += complete_tag + " " * (max(1, 14 - len(complete_tag)))


tagline += "\n"

f.write(tagline)

# collect data from each file and write to outfile
lines = ""
for count, filepath in enumerate(inputfiles, 1):

    slha_dict = pyslha.readSLHAFile(filepath)

    datalist = []

    accepted_file = True
    number_of_unaccepted_xsections = 0  # IH

    for key in datadict.keys():

        if "block" in datadict[key].keys():

            if not datadict[key]["block"] in slha_dict.keys():
                accepted_file = False
                sys.stdout.write(
                    "%s Problem encountered when looking for block %s in file %s. File ignored.\n" % (
                        outpref, datadict[key]["block"], filepath)
                )
                break

            if ("abs" in datadict[key].keys()) and (datadict[key]["abs"] == True):

                datalist.append(abs(slha_dict[datadict[key]["block"]].getElement(*datadict[key]["element"])))
            else:

                try:
                    slha_dict[datadict[key]["block"]].getElement(*datadict[key]["element"])  # IH
                    # print slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] )
                    datalist.append(slha_dict[datadict[key]["block"]].getElement(*datadict[key]["element"]))
                except:
                    number_of_unaccepted_xsections += 1
                    # Add -1 as table entry for missing xsections
                    datalist.append(-1)

        elif "decay" in datadict[key].keys():

            if not datadict[key]["decay"] in slha_dict.keys():
                accepted_file = False
                sys.stdout.write(
                    "%s Problem encountered when looking for decay %s in file %s. File ignored.\n" % (
                        outpref, datadict[key]["decay"], filepath)
                )
                break

            datalist.append(slha_dict[datadict[key]["decay"]].getBR(list(datadict[key]["element"])))

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
    file = filepath.split("/")[-1]

    # Make lines for printing
    lines += ("%s" + " " * (max_path_length - len(filepath) + 2)) % file
    lines += (("% .5e" + " " * 2) * len(datatuple) + "\n") % datatuple

    # write to file once per 1000 files read
    if count % 1000 == 0:
        sys.stdout.write("%s %d of %d files read\n" % (outpref, count, n_files_total))
        f.write(lines)
        lines = ""


# Remove final endline and write remaining lines
lines = lines.rstrip("\n")
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
