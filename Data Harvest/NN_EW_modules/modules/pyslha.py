#! /usr/bin/env python
"""
Modified and simplified Python SLHA parser.

(Based on pySLHA by Andy Buckley. Modified by Anders Kvellestad.)
"""

# ToDo:
# -----
# 
# - write doc-strings
# - test
# - extend?

from collections import OrderedDict


#
# Block key formats:
# - If a block is not listed here, its Block.keyformat value is set to 1 by defualt
#
block_key_formats = { 'ALPHA'   : 0,
                      'NMIX'    : 2, 
                      'UMIX'    : 2, 
                      'VMIX'    : 2, 
                      'STOPMIX' : 2, 
                      'SBOTMIX' : 2, 
                      'STAUMIX' : 2,
                      'YE'      : 2,
                      'YU'      : 2,
                      'YD'      : 2,
                      'PROSPINO_OUTPUT': 3 }



#
# Some general methods:
#
def _autotype(var):
    """Automatically convert strings to numerical types if possible."""
    if type(var) is not str:
        return var
    if var.isdigit() or (var.startswith("-") and var[1:].isdigit()):
        return int(var)
    try:
        f = float(var)
        return f
    except ValueError:
        return var

def _autostr(var, precision=8):
    """Automatically numerical types to the right sort of string."""
    if type(var) is float:
        return ("%." + str(precision) + "e") % var
    return str(var)

def extractElements(inputstring):

  # check bracket balance
  if inputstring.count('|') % 2 != 0:
    raise Exception("delimiters '|' are not balanced in string: '%s'" % inputstring)

  withinbars = False
  lastsplit  = 0
  lastbar    = 0
  res = []
 
  # scan input string
  for i,c in enumerate(inputstring):

    # if normal whitespace outside delimiters, do normal split
    # -- ignore blanks
    if (c == " ") and (withinbars == False):
      appendstring = inputstring[lastsplit:i].strip()
      if appendstring != "":
        res.append( appendstring )
      lastsplit = i

    # when the first of a set of bars is encountered,
    # set the flags to appropriate values
    elif (c == "|") and (withinbars == False):
      lastbar    = i
      withinbars = True

    # when the last of a set of bars is encountered,
    # append the content within bars to the result list
    elif (c == "|") and (withinbars == True):
      res.append( inputstring[lastbar+1:i] )
      lastsplit  = i+1
      withinbars = False

    # when the end of the string is reaced,
    # append the final entry to the result list
    elif (i == len(inputstring)-1) and (withinbars == False):
      res.append( inputstring[lastsplit:].strip() )

  return res


#
# Error classes
#
class ParseError(Exception):
    "Exception object to be raised when a spectrum file/string is malformed"
    def __init__(self, errmsg):
        self.msg = errmsg
    def __str__(self):
        return self.msg

class NotFoundError(Exception):
    "Exception object to be raised when a requested key/entry/element is not found"
    def __init__(self, errmsg):
        self.msg = errmsg
    def __str__(self):
        return self.msg


#
# The Block class
#
class Block(object):
    """
    docstring for Block class...
    """
    def __init__(self, name, q=None):
        self.name = name.upper()
        self.entries = OrderedDict()
        self.q = _autotype(q)
        self.comments = {'title': '', 1: ''}
        if self.name in block_key_formats:
          self.keyformat = block_key_formats[self.name]
        else:
          self.keyformat = 1
        
        
    def newEntry(self, entry, comment=''):
        if type(entry) != list:
            raise TypeError("block entries must be of type 'list'")
        key = self.constructKey(entry)
        entry = list(map(_autotype, entry))
        self.entries[key] = entry

        ## Add comment
        self.addComment(key, comment)
        # self.comments[key] = comment


    def setEntry(self, key, newvalue, comment=''):
        if type(newvalue) != list:
            raise TypeError("block entries must be of type 'list'")
        #if type(key) != int:
        #    print self.name, key, newvalue
        #    raise TypeError("block keys must be of type 'int'")
        self.entries[key] = newvalue

        ## Add comment
        self.addComment(key, comment)
        # self.comments[key] = comment


    def getKey(self, lookup_values):
        if type(lookup_values) != list:
            raise TypeError("search values must be given in a list")
        for key, blockentry in list(self.entries.items()):
            is_found = True
            if len(lookup_values) > len(blockentry):
                is_found = False
                continue
            for index,lookup_val in enumerate(lookup_values):
                if lookup_val == blockentry[index]:
                    continue
                else:
                    is_found = False
            if is_found == True:
                return key
            else:
                continue
        raise NotFoundError("value(s) '%s' not found in block '%s'" % (str(lookup_values), self.name))


    def getEntry(self,key):
        if key in self.entries:
            return self.entries[key]
        else:
            raise NotFoundError("no entry with key '%s' found in block '%s'" % (str(key), self.name))


    def getEntryByVal(self,lookup_values):
        try:
            key = self.getKey(lookup_values)
            return self.entries[key]
        except Exception as err:
            raise err


    def getElement(self,key,col):
        try:
            entry = self.getEntry(key)
            #print entry, col, entry[col]
            return entry[col]
        except Exception as err:
            raise err
        

    def getElementByVal(self,lookup_values,col):
        try:
            key = self.getKey(lookup_values)
            return self.getElement(key,col)
        except Exception as err:
            raise err
        

    def constructKey(self, entry):
        if self.keyformat == 0:
          key = 1
        elif self.keyformat == 1:
          key = entry[0]
        elif self.keyformat > 1:
          key = tuple(entry[0:self.keyformat])
        return key
          

    def addComment(self, key, comment):
        keylist = list(self.entries.keys())
        if (key not in keylist) and (key != 'title'):
            print(('key: ', key, '\n'))
            raise Exception("Comment key must be identical to an existing block entry key, or set to 'title'.")
        if type(comment) != str:
            raise Exception("Comment must be of type string")
        comment = ' '.join(comment.split())
        if (len(comment) > 0) and (comment[0] == '#'):
            comment = comment[1:]
            comment = ' '.join(comment.split())
        self.comments[key] = '# ' + comment

    def __cmp__(self, other):
        return cmp(self.name, other.name)

    def __str__(self):
        s = self.name
        if self.q is not None:
            s += " (Q=%s)" % self.q
        s += "\n"
        s += str(self.entries)
        return s

    def __repr__(self):
        return self.__str__()


#
# The Decay class
#
class Decay(object):
    """
    Object representing a decay entry on a particle decribed by the SLHA file.
    'Decay' objects are not a direct representation of a DECAY block in an SLHA
    file... that role, somewhat confusingly, is taken by the Particle class.

    Decay objects have three properties: a branching ratio, br, an nda number
    (number of daughters == len(ids)), and a tuple of PDG PIDs to which the
    decay occurs. The PDG ID of the particle whose decay this represents may
    also be stored, but this is normally known via the Particle in which the
    decay is stored.
    """
    def __init__(self, br, nda, ids, parentid=None):
        self.parentid = parentid
        self.br = br
        self.nda = nda
        self.ids = ids
        assert(self.nda == len(self.ids))

    def __cmp__(self, other):
        return cmp(other.br, self.br)

    def __str__(self):
        return "%.8e %s" % (self.br, self.ids)

    def __repr__(self):
        return self.__str__()

#
# The Particle class
#
class Particle(object):
    """
    Representation of a single, specific particle, decay block from an SLHA
    file.  These objects are not themselves called 'Decay', since that concept
    applies more naturally to the various decays found inside this
    object. Particle classes store the PDG ID (pid) of the particle being
    represented, and optionally the mass (mass) and total decay width
    (totalwidth) of that particle in the SLHA scenario. Masses may also be found
    via the MASS block, from which the Particle.mass property is filled, if at
    all. They also store a list of Decay objects (decays) which are probably the
    item of most interest.
    """
    def __init__(self, pid, totalwidth=None, mass=None):
        self.pid = pid
        self.totalwidth = totalwidth
        self.mass = mass
        self.decays = []
        self.comments = {'title': ''}
        
    def addDecay(self, br, nda, ids):
        self.decays.append(Decay(br, nda, ids))
        self.decays.sort()
        ## Add an empty comment
        self.comments[str(ids)] = ''

    def getBR(self, ids):
        if type(ids) != list:
            raise Exception("Argument must be a list of PDG IDs of the decay products. (e.g. [-5,5])")

        found_decay = False
        return_br   = 0.0
        for decay in self.decays:
            if sorted(decay.ids) == sorted(ids):
                found_decay = True
                return_br = decay.br
                break
        # if found_decay == False:
        #   raise Exception('Branching ratio for decay to PDGs ' + str(ids) + ' not found')
        return return_br
    
    def updateMass(self,SLHAcontent):
        try:
            self.mass = SLHAcontent['MASS'].getElement_byval([self.pid],2)
        except:
            raise ParseError("No MASS block entry found for particle %d -- cannot set mass value" % self.pid)

		
    def addComment(self, key, comment):
        keylist = []
        key = str(key)
        for i in range(len(self.decays)):
            keylist.append( str(self.decays[i].ids) )
        if (key not in keylist) and (key != 'title'):
            print(('key: ', key, '\n'))
            raise Exception("Comment key must be identical to an existing decay id (e.g. [-5,5]), or set to 'title'.")
        if type(comment) != str:
            raise Exception("Comment must be of type string")
        comment = ' '.join(comment.split())
        if (len(comment) > 0) and (comment[0] == '#'):
            comment = comment[1:]
            comment = ' '.join(comment.split())
        self.comments[key] = '# ' + comment

    def __cmp__(self, other):
        if abs(self.pid) == abs(other.pid):
            return cmp(self.pid, other.pid)
        return cmp(abs(self.pid), abs(other.pid))

    def __str__(self):
        s = str(self.pid)
        if self.mass is not None:
            s += " : mass = %.8e GeV" % self.mass
        if self.totalwidth is not None:
            s += " : total width = %.8e GeV" % self.totalwidth
        for d in self.decays:
            if d.br > 0.0:
                s += "\n  %s" % d
        return s

    def __repr__(self):
        return self.__str__()


#
# Method definitions
#
def readSLHA(spcstr):
    """
    Read an SLHA definition from a string, returning an OrderedDict
    containing Block objects, Particle objects and Full-line comments.
    
    In-line comments are stored in dicts as: 
                    [Block object].comments
                    [Particle object].comments
    
    Full-line comments are stored as elements in the OrderedDict, 
    with keys 'COMMENT1', 'COMMENT2', etc.
    """

    SLHAcontent = OrderedDict()

    comm_i = 0      ## Counter for number of full-line comments

    import re
    currentblockname = None
    currentdecaypdg  = None
    for line in spcstr.splitlines():

        ## Handle full-line comments
        if line.startswith(r"#"):
            comm_i = comm_i + 1
            comm_name = 'COMMENT' + str(comm_i)
            SLHAcontent[comm_name] = line 
            continue

        ## Handle in-line comments (set position of '#' symbol)
        if "#" in line:
            comm_pos = line.index("#")
            line_nocomment = line[:comm_pos]
            line_comment   = line[comm_pos:]
        else:
            line_nocomment = line
            line_comment   = ""
            
        ## Handle BLOCK start lines
        if line.upper().startswith("BLOCK"):
            #print line
            match = re.match(r"BLOCK\s+(\w+)(\s+Q\s*=\s*.+)?", line_nocomment.upper())
            if not match:
                continue
            blockname = match.group(1)
            qstr = match.group(2)
            if qstr is not None:
                qstr = qstr[qstr.find("=")+1:].strip()
            currentblockname = blockname
            currentdecaypdg  = None

            block = Block(blockname, q=qstr)
            block.addComment('title', line_comment)
            SLHAcontent[blockname] = block

        ## Handle DECAY start lines
        elif line.upper().startswith("DECAY"):
            match = re.match(r"DECAY\s+(\d+)\s+([\d\.E+-]+).*", line_nocomment.upper())
            if not match:
                continue
            pdgid = int(match.group(1))
            width = float(match.group(2))
            currentblockname = "DECAY"
            currentdecaypdg = pdgid
            SLHAcontent[pdgid] = Particle(pdgid, width)
            SLHAcontent[pdgid].addComment('title', line_comment)

        ## Handle in-block lines
        else:
            if currentblockname is not None:
                items_strlist = extractElements(line_nocomment)
                #if '{' in line_nocomment:
                #  items_strlist = extractElements(line_nocomment)
                #else:
                #  items_strlist = line_nocomment.split()
                ## If line is empty, continue
                if len(items_strlist) == 0:
                    continue
                ## If line is in a BLOCK
                if currentblockname != "DECAY":
                    items = list(map(_autotype,items_strlist))
                    block = SLHAcontent[currentblockname]
                    currentkey = block.constructKey(items)
                    block.setEntry(currentkey,items)
                    block.addComment(currentkey,line_comment)
                ## If line is in a DECAY
                else:
                    br  = float(items_strlist[0])
                    nda = int(items_strlist[1])
                    ids = list(map(int, items_strlist[2:]))
                    SLHAcontent[currentdecaypdg].addDecay(br, nda, ids)
                    SLHAcontent[currentdecaypdg].addComment(ids, line_comment)

    return SLHAcontent



def writeSLHA(SLHAcontent, precision=8):
    """
    Returns a string with the content supplied in the OrderedDict (SLHAcontent)
    containing Block objects, Particle objects and Full-line comments
    """
    fmte = "% ." + str(precision) + "e"
    sep = "   "
    out = ""
    comm_pos = 46

   
    def formatLineBeforeComment(lineelements, floatformat, commentposition, titletag=""):
        nelems = len(lineelements)
        line = ""
        if titletag != "":
            line += titletag
        for i in range(nelems):
            if type(lineelements[i]) == float:
                elemstr = (floatformat) % lineelements[i]
                elemlength = len(elemstr)
                spaces = max(4, 6 - elemlength) 
                line += (" "*spaces + floatformat) % lineelements[i]
            elif type(lineelements[i]) == int:
                elemlength = len( str(lineelements[i]) )
                spaces = max(1, 6 - elemlength) 
                line += (" "*spaces + "%d") % lineelements[i]                                                    
            else:
                elemlength = len( str(lineelements[i]) )
                spaces = max(1, 6 - elemlength) 
                line += (" %s") % lineelements[i]
        line_length = len(line)
        spaces = max(4, comm_pos - line_length)
        line = line + " "*spaces
        return line


    ## Use the SLHAcontent OrderedDict
    for ident, elem in list(SLHAcontent.items()):

        ## If BLOCK:
        if type(elem) == Block:
            namestr = elem.name
            if elem.q is not None:
                namestr += (" Q= " + fmte) % float(elem.q)
            titleelements = [namestr]
            titletag = "BLOCK"
            title = formatLineBeforeComment(titleelements, fmte, comm_pos, titletag)
            out += (title + "%s\n") % (SLHAcontent[ident].comments['title'])

            ## Move through the substructure of the BLOCK
            d = elem.entries  # <-- should be an OrderedDict
            for key, val_list in list(d.items()): 
                val_str = list(map(str,val_list))
                s = ' '.join(val_str)
                sitems = list(map(_autotype, s.split()))
                line = formatLineBeforeComment(sitems, fmte, comm_pos)
                out += (line + "%s\n") % (SLHAcontent[ident].comments[key])
                        
        # If DECAY:
        elif type(elem) == Particle:
            titleelements = [elem.pid, (elem.totalwidth or -1)]
            titletag = "DECAY"
            title = formatLineBeforeComment(titleelements, fmte, comm_pos, titletag)
            out += (title + "%s\n") % (SLHAcontent[ident].comments['title'])

            ## Move through the substructure of the DECAY
            for d in elem.decays:
                lineelements = [d.br, len(d.ids)] + d.ids 
                line = formatLineBeforeComment(lineelements, fmte, comm_pos)
                comm_id = str(d.ids)
                out += (line + "%s\n") % (SLHAcontent[ident].comments[comm_id])

        # If full-line comment:
        elif type(elem) == str:
            out += elem
            out += '\n'
        # If unknown entry:
        else:
            print('Unknown entry!')
            print('')
    return out

 

def readSLHAFile(spcfilename):
    """
    Read an SLHA file, returning an OrderedDict of Block objects, 
    Particle objects and Full-line comments
    """
    f = open(spcfilename, "r")
    rtn = readSLHA(f.read())
    f.close()
    return rtn



def writeSLHAFile(spcfilename, SLHAcontent, precision=8):
    """
    Write an SLHA file from the supplied OrderedDict ('SLHAcontent'), 
    which contains:
                    - Full-line comments
                    - Block objects (for BLOCKs)
                    - Particle objects (for DECAYs)
                    
    Other keyword parameters are passed to writeSLHA.
    """
    f = open(spcfilename, "w")
    f.write(writeSLHA(SLHAcontent, precision))
    f.close()


def updateAllMasses(SLHAcontent):
    for dictval in list(SLHAcontent.values()):
        if isinstance(dictval, Particle):
            try:
                dictval.updateMass(SLHAcontent)
            except:
                raise ParseError("No MASS block entry found for particle %d -- cannot set mass value" % dictval.pid)

