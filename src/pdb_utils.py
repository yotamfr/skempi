import os
import os.path as osp
import requests

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

AAA = {val:key for key, val in AA_dict.iteritems()}

UNKNOWN = 'UNK'


UNHANDLED = {'HETSYN', 'COMPND', 'SOURCE', 'KEYWDS', 'EXPDTA', 'AUTHOR', 'REVDAT', 'JRNL', 'SIGATM',
             'REMARK', 'DBREF', 'SEQADV', 'HET', 'HETATM', 'HETNAM', 'FORMUL', 'HELIX', 'CAVEAT',
             'SHEET', 'SITE', 'LINK', 'CISPEP', 'SITE', 'CRYST1', 'MTRIX1', 'MTRIX2', 'MTRIX3',
             'ORIGX1', 'ORIGX2', 'ORIGX3', 'SSBOND', 'MODRES', 'SPRSDE', 'ANISOU', 'ANISOU10000',
             'HETATM13247', 'HETATM13248', 'HETATM13249', 'HETATM13250', 'HETATM13256', 'HETATM13258',
             'HETATM13251', 'HETATM13252', 'HETATM13253', 'HETATM13254', 'HETATM13255', 'HETATM13257',
             'SCALE1', 'SCALE2', 'SCALE3', 'SEQRES', 'CONECT', 'MASTER', 'END'}


amino_acids = "ARNDCQEGHILKMFPSTWYV"


def download_pdb(pdb, outdir="../data/pdbs_n"):
    assert osp.exists(outdir)
    print("downloading %s->%s" % (pdb, outdir))
    req = requests.get('http://files.rcsb.org/download/%s.pdb' % (pdb,))
    ext = "pdb"
    if req.status_code == 404: # then assume it's a .cif
        req = requests.get('http://files.rcsb.org/download/%s.cif' % (pdb,))
        ext = "cif"
    if req.status_code != 200:   # then assume it's a .cif
        raise requests.HTTPError('HTTP Error %s' % req.status_code)
    with open(osp.join(outdir, "%s.%s" % (pdb, ext)), 'w+') as f:
        f.write(req.content)


class Atom(object):

    def __init__(self, name, res_num, x, y, z, temp):
        self.name = name
        self.temp = temp
        self.res_num = res_num
        self._coord = (x, y, z)

    @property
    def type(self):
        return self.name[0]

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
        self._id = pdb

    @property
    def pdb(self):
        return self._id

    def __iter__(self):
        for chain in self.chains.values():
            yield chain

    def __getitem__(self, chain_id):
        return self.chains[chain_id]


def _handle_line(line, residues, chains, pdb, chain_id=''):

    typ = line.strip().split(' ')[0]

    if not line:
        return residues, chains, chain_id

    if typ in UNHANDLED:
        pass
    elif typ == 'HEADER':
        pass
    elif typ == 'TITLE':
        pass
    elif 'HETATM' in typ:
        pass
    elif 'CONECT' in typ:
        pass
    elif 'ANISOU' in typ:
        pass

    elif typ == 'ATOM':

        # https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
        l_line = line[6:11], line[12:16], line[17:20], line[21], line[22:26], \
                 line[30:38], line[38:46], line[46:54], \
                 line[54:60], line[60:66], line[72:76], line[76:78]
        l_line = [s.strip() for s in l_line]
        atom_num, atom_name, res_name, chain_id, res_num, x, y, z, occup, temp, ident, sym = l_line

        try:
            occup, temp = float(occup), float(temp)
        except ValueError as e:
            occup, temp = -1, -1
        x, y, z = float(x), float(y), float(z)
        try:
            res_name = AA_dict[res_name[-3:]]
        except KeyError:
            return residues, chains, chain_id

        atom_num, res_num = int(atom_num), int(res_num)
        atom = Atom(atom_name, res_num, x, y, z, temp)

        if len(residues) == 0:
            res = Residue(res_name, [atom])
            residues.append(res)
        elif residues[-1].atoms[-1].res_num != res_num:
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


def parse_pdb2(pdb, path):
    return parse_pdb(pdb, open(path, "r"))


if __name__ == "__main__":
    import os.path as osp
    pdb = "1tm1"
    PDB_PATH = "../data/"
    fd = open(osp.join(PDB_PATH, "%s.pdb" % pdb), 'r')
    struct = parse_pdb(pdb, fd)
    print(struct.id)
