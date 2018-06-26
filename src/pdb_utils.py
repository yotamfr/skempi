AA_dict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASX": "B",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLX": "Z",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


amino_acids = "ARNDCQEGHILKMFPSTWYV"


class Atom(object):

    def __init__(self, name, x, y, z):
        self.name = name
        self._coord = (x, y, z)

    @property
    def coord(self):
        return self._coord


class Residue(object):

    def __init__(self, name, atoms=[]):
        self.atoms = atoms
        self.name = name

    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def __getitem__(self, i):
        return self.atoms[i]

    def __str__(self):
        return "<Residue %s>" % self.name


class Chain(object):

    def __init__(self, pdb, chain_id, residues=[]):
        self.residues = residues
        self.chain_id = chain_id
        self.pdb = pdb

    def __iter__(self):
        for res in self.residues:
            yield res

    def __getitem__(self, i):
        return self.residues[i]

    def __str__(self):
        return "<Chain %s %d>" % (self.id, len(self))

    @property
    def seq(self):
        return ''.join([res.name for res in self.residues])

    @property
    def id(self):
        return "%s_%s" % (self.pdb, self.chain_id)

    def __len__(self):
        return len(self.seq)


class PDB(object):

    def __init__(self, pdb, chains_dict={}):
        self.chains = chains_dict
        self.id = pdb

    def __iter__(self):
        for chain in self.chains.values():
            yield chain

    def __getitem__(self, id):
        return self.chains[id]


def _handle_line(line, residues, chains, pdb, chain_id=''):

    typ = line[:4].strip()

    if not typ:
        pass

    elif typ == 'ATOM':
        # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        l_line = line[6:11], line[12:16], line[17:20], line[21], line[22:26], \
                 line[30:38], line[38:46], line[46:54]

        l_line = [s.strip() for s in l_line]

        atom_num, atom_name, res_name, chain_id, res_num, x, y, z = l_line

        x, y, z = float(x), float(y), float(z)
        res_name = AA_dict[res_name[-3:]]
        atom_num, res_num = int(atom_num), int(res_num)
        atom = Atom(atom_name, x, y, z)

        if res_num - 1 == len(residues):
            res = Residue(res_name, [atom])
            residues.append(res)
        else:
            res = residues[-1]
            res.atoms.append(atom)

    elif typ == 'TER':
        assert chain_id
        chain = Chain(pdb, chain_id, residues)
        chains[chain_id] = chain
        residues = []

    else:
        raise ValueError("Unidentified row type: %s" % typ)

    return residues, chains, chain_id


def parse_pdb(pdb, fd):

    line = fd.readline()

    residues, chains, chain_id = _handle_line(line, [], {}, pdb)

    while line:

        line = fd.readline()

        residues, chains, chain_id = _handle_line(line, residues, chains, pdb, chain_id)

    return PDB(pdb, chains)
