import time

all_masses = ['m1000021', 'm1000022', 'm1000023', 'm1000024', 'm1000025', 'm1000035', 'm1000037', 'm1000001',
              'm1000002', 'm1000003', 'm1000004', 'm1000005', 'm1000006', 'm1000011', 'm1000012', 'm1000013',
              'm1000014', 'm1000015', 'm1000016',
              'm2000001', 'm2000002', 'm2000003', 'm2000004', 'm2000005', 'm2000006', 'm2000011', 'm2000013',
              'm2000015']
all_cs = ['m1000022m1000022', 'm1000022m1000023', 'm1000022m1000025', 'm1000022m1000035', 'm1000022m1000024',
          'm1000022m1000037', 'm1000022-m1000024', 'm1000022-m1000037',
          'm1000023m1000023', 'm1000023m1000025', 'm1000023m1000035', 'm1000023m1000024', 'm1000023m1000037',
          'm1000023-m1000024', 'm1000023-m1000037', 'm1000025m1000025',
          'm1000025m1000035', 'm1000025m1000024', 'm1000025m1000037', 'm1000025-m1000024', 'm1000025-m1000037',
          'm1000035m1000035', 'm1000035m1000024', 'm1000035m1000037',
          'm1000035-m1000024', 'm1000035-m1000037', 'm1000024-m1000024', 'm1000024-m1000037', 'm1000037-m1000024',
          'm1000037-m1000037']

prospino_elements = ['LO', 'LO_relerr', 'NLO', 'NLO_1_relerr', 'NLO_1', 'NLO_05', 'NLO_2', 'NLO_pdf', 'NLO_aup',
                     'NLO_adn']

def filter_dict(s, include, dict_glob, dict_loc, en=13000):
    if include is None:
        dict_glob.update(dict_loc)
    if s in include:
        i = include.index(s) + 1
        j = 0
        while True:
            if i == len(include):
                break
            elif ':' in include[i]:
                break
            else:
                key = include[i]
                j = harvest_and_filter_by_include(s, key, dict_glob, dict_loc, en)
            i += 1
        if j == 0:
            dict_glob.update(dict_loc)
            print(
                '\nCANT recognize ANY of input parameters in "%s"\nHarvesting available dictionary-data in current block instead\n' % s)
        else:
            print('')
    else:
        dict_glob.update(dict_loc)
    return


def fix_mass_name(input_name, en):
    new_name = ""
    PID = ""
    order = 'NLO_1'
    mass_ls = input_name.split('_')

    if len(mass_ls) == 4:
        mass_ls[2] = mass_ls[2] + '_' + mass_ls[3]
        mass_ls.pop()

    for i, m in enumerate(mass_ls):
        if m[0] in {'M', 'm', '-'} and i <= 1:
            mass = m.lower()
            default_size = 8

            if len(mass_ls) > 1 and mass[0] == 'm':
                mass = mass.replace('ø', '0' * (default_size - (len(mass) - 1)))
                mass = mass[1:]
            elif len(mass_ls) > 1 and mass[0] == '-':
                mass = mass.replace('ø', '0' * (default_size - (len(mass) - 2)))
                mass = '-' + mass[2:]
            else:
                mass = mass.replace('ø', '0' * (default_size - (len(mass) - 1)))

            if i == 1:
                new_name += "_"
            new_name += mass
            if i == 1:
                PID = new_name

        elif any(x in m for x in {'LO', 'lo', 'NLO', 'nlo'}) and i == 2:
            order = m.upper()
            new_name += "_" + en + "_" + order
        else:
            print('\nInput is not correct.')
    if len(mass_ls) == 2:
        order = 'NLO_1'
        new_name += "_" + en + "_" + order
    return new_name, PID, order


def harvest_and_filter_by_include(s, key, dict_glob, dict_loc, en):
    j = 0
    if key[0] == 'm' and s in {'M:', 'O:'}:
        if key[1] == '-':
            key = key[0] + key[2:]
        key, _, _ = fix_mass_name(key, en)
        value = dict_loc.get(key)
        if key not in dict_glob.keys():
            if value is not None:
                dict_glob[key] = value
                print('*** %s added to dictionary' % key)
                j += 1
            elif key in all_masses:
                print('*** %s added to dictionary(was not in dictionary %s, but found in the data)' % (key, s))
                key_ = int(key[1:])
                dict_glob[key] = {'block': 'MASS', 'element': (key_, 1), 'abs': abs}
                j += 1
            else:
                print('*** %s NOT added to dictionary(not recognized or found)' % key)
        else:
            j += 1

    elif key[0] == 'm' and s in {'CS:'}:
        key, PID, order = fix_mass_name(key, en)
        value = dict_loc.get(key)
        PIDs = PID.split('_')
        if PIDs[0][0] == '-' and PIDs[1][0] != '-':
            # PIDs[0] = PIDs[0][1:]
            PID = '-m' + PIDs[0][1:] + 'm' + PIDs[1]
        elif PIDs[1][0] == '-' and PIDs[0][0] != '-':
            # PIDs[1] = PIDs[1][1:]
            PID = 'm' + PIDs[0] + '-m' + PIDs[1][1:]
        else:
            PID = 'm' + PIDs[0] + 'm' + PIDs[1]

        if value is not None:
            dict_glob[key] = value
            print('*** %s added to dictionary' % key)
            j += 1
        elif PID in all_cs:
            # ordernr = 7 is NLO_1
            ordernr = 7

            if order in prospino_elements:
                ordernr = prospino_elements.index(order)+3
            else:
                print('Input order not recognized or found. Order set to %s' % (prospino_elements[4]))
            PID1 = int(PIDs[0])
            PID2 = int(PIDs[1])
            dict_glob[key] = {'block': 'PROSPINO_OUTPUT', 'element': ((PID1, PID2, int(en)), ordernr)}
            print('*** %s added to dictionary(was not in dictionary %s, but found in the data)' % (key, s))
            j += 1
        else:
            print('*** %s NOT added to dictionary(not recognized or found)' % key)
    return j


def str2bool(v):
    if v in ("yes", "y", "true", "True", "1"):
        return True
    elif v in ("no", "no", "false", "False", "0"):
        return False

def include_fail():
    cont = True
    while cont:
        time.sleep(1)
        print('ERRONEOUS INPUT IN "include"')
        x = input("Press enter for instructions, or 's' to skip ")
        if x == 's':
            break
        print("\n")
        time.sleep(1)
        print("READ FOR INSTRUCTION: ")
        time.sleep(4)
        print('For input variable "include", please use format "<parameter key>: <element name>"')
        time.sleep(5)
        print(
            'Note that zeros can be excluded from name by using a "Ø" or "ø" instead. For cross section data use underscore "_" between masses and to specify order.')
        time.sleep(8)
        print('Example: "P: tanb M: m1Ø22 m1Ø23 CS: m1Ø22_m1Ø23_NLO1"\n')
        time.sleep(10)
        print(".")
        time.sleep(0.5)
        print(".")
        time.sleep(0.5)
        print(".")
        time.sleep(0.5)
        print(".")
        time.sleep(0.5)
        print(".")
        time.sleep(0.5)
        print(".")
        time.sleep(0.5)
        cont = False
    print('\nIncluding all available elements for given parameters\n')
    time.sleep(4)
    print("\n\n\n")
    return
