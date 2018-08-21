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

    def __init__(self, name, num, residue, x, y, z, occup, temp):
        self.num = num
        self.name = name
        self.temp = temp
        self.occup = occup
        self.res = residue
        self._coord = (x, y, z)

    @property
    def res_name(self):
        return self.res.name

    @property
    def res_num(self):
        return self.res.num

    @property
    def chain_id(self):
        return self.res.chain.chain_id

    @property
    def type(self):
        return self.name[0]

    @property
    def coord(self):
        return self._coord

    @property
    def x(self):
        return self._coord[0]

    @property
    def y(self):
        return self._coord[1]

    @property
    def z(self):
        return self._coord[2]

    def __str__(self):
        atom_num = self.num
        atom_name = self.name
        res_num = self.res_num
        aa = self.res_name
        chain_id = self.chain_id
        occup = self.occup
        temp = self.temp
        (x, y, z) = self._coord
        # return '{typ: <6}{0: >5}  {1: <4}{2: <4}{3: <1}{4: >4}    ' \
        #        '{5:8.3f}{6:8.3f}{7:8.3f}'\
        #     .format(atom_num, atom_name, AAA[aa], chain_id, res_num, x, y, z, typ='ATOM')
        return '{typ: <6}{0: >5}  {1: <4}{2: <4}{3: <1}{4: >4}    ' \
               '{5:8.3f}{6:8.3f}{7:8.3f}{8:6.2f}{9:6.2f}          ' \
               ' {10: <4}'\
            .format(atom_num, atom_name, AAA[aa], chain_id, res_num, x, y, z, occup, temp, self.type, typ='ATOM')


class Residue(object):

    def __init__(self, name, num, chain, atoms=[]):
        self.atoms = atoms
        self.name = name
        self.chain = chain
        self.num = num

    @property
    def ca(self):
        return self.get_atom_by_name("CA")

    def get_atom_by_name(self, name):
        atom = [a for a in self.atoms if a.name == name]
        assert len(atom) >= 1
        return atom[0]

    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def __getitem__(self, i):
        return self.atoms[i]

    def __str__(self):
        return "<Residue %s>" % AAA[self.name]


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
    def atoms(self):
        return reduce(lambda atoms, res: atoms + res.atoms, self.residues, [])

    @property
    def seq(self):
        return ''.join([res.name for res in self.residues])

    @property
    def id(self):
        return "%s_%s" % (self.pdb, self.chain_id)

    def __len__(self):
        return len(self.seq)


class PDB(object):

    def __init__(self, modelname, chains=[], chain_dict={}):
        self._id_to_ix = {}
        self._id = modelname
        self._chains = []
        for _, c in enumerate(chains):
            if len(chain_dict) > 0:
                if c.chain_id not in chain_dict:
                    continue
                cid = chain_dict[c.chain_id]
                c.chain_id = cid
            else:
                cid = c.chain_id
            self._id_to_ix[cid] = len(self._chains)
            self._chains.append(c)

    @property
    def chains(self):
        return {c.chain_id: c for c in self._chains}

    @property
    def pdb(self):
        return self._id

    def __iter__(self):
        for chain in self._chains:
            yield chain

    def __getitem__(self, chain_id):
        return self._chains[self._id_to_ix[chain_id]]

    def to_pdb(self, path):
        lines = []
        for chain in self._chains:
            for atom in chain.atoms:
                lines.append("%s\n" % atom)
            lines.append("TER\n")
        with open(path, "w+") as f:
            f.writelines(lines)


def _handle_line(line, residues, chains, pdb, chain_id='', residue_num=0):

    typ = line.strip().split(' ')[0]

    if not line:
        return residues, chains, chain_id, residue_num

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
        atom_num, res_num = int(atom_num), int(res_num)

        try:
            occup, temp = float(occup), float(temp)
        except ValueError as e:
            occup, temp = 1.00, 00.00
        x, y, z = float(x), float(y), float(z)
        try:
            res_name = AA_dict[res_name[-3:]]
        except KeyError:
            return residues, chains, chain_id, res_num

        atom = Atom(atom_name, atom_num, None, x, y, z, occup, temp)

        if len(residues) == 0 or residue_num != res_num:
            res = Residue(res_name, len(residues) + 1, None, [atom])
            residues.append(res)
        else:
            res = residues[-1]
            res.atoms.append(atom)

        atom.res = res
        residue_num = res_num

    elif typ == 'TER':
        assert chain_id
        chain = Chain(pdb, chain_id, residues)
        for res in residues: res.chain = chain
        chains.append(chain)
        residues = []

    else:
        raise ValueError("Unidentified row type: %s" % typ)

    return residues, chains, chain_id, residue_num


def parse_pdb(pdb, fd, chain_dict={}):

    line = fd.readline()

    residues, chains, chain_id, res_num = _handle_line(line, [], [], pdb)

    while line:

        line = fd.readline()

        residues, chains, chain_id, res_num = _handle_line(line, residues, chains, pdb, chain_id, res_num)

    return PDB(pdb, chains, chain_dict)


def parse_pdb2(pdb, path):
    return parse_pdb(pdb, open(path, "r"))


if __name__ == "__main__":
    import os.path as osp
    pdb = "1A4Y"
    PDB_PATH = "../data/pdbs"
    fd = open(osp.join(PDB_PATH, "%s.pdb" % pdb), 'r')
    struct = parse_pdb(pdb, fd)
    struct.to_pdb("../data/1a4y.pdb")
    print(struct.pdb)
