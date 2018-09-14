import os
import sys
import shutil
from os import path as osp
from glob import glob

from pdb_utils import *

from skempi_consts import *

fmt_modeller = """
# Comparative modeling by the automodel class
#
# Demonstrates how to build multi-chain models, and symmetry restraints
#
from modeller import *
from modeller.automodel import *    # Load the automodel class

log.verbose()

env = environ()
# directories for input atom files
env.io.atom_files_directory = ['.', '../atom_files']

a = automodel(env,
            alnfile  = 'complex.ali' ,    # alignment filename
            knowns   = '%s',              # codes of the templates
            sequence = '%s')              # code of the target

a.starting_model= 1                # index of the first model
a.ending_model  = 1                # index of the last model
                                   # (determines how many models to calculate)
a.make()                           # do comparative modeling

"""

MODELLER_CHAINS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def create_modeller_workspace(template, mutant, workspace):
    ali = Ali(template, mutant)
    with open(osp.join(workspace, "mutate-model.py"), 'w+') as f:
        f.write(fmt_modeller % (template.name, mutant.name))
    with open(osp.join(workspace, "complex.ali"), 'w+') as f:
        f.write(str(ali))


def ali_template(model):
    header = ">P1;%s\n" % model.name
    sequences = model.sequences
    seqs = [k for k, _ in sequences]
    chains_ids = "\t".join("::%s" % (k,) for k in [seqs[0], seqs[-1]])
    structure = "%s:%s\t%s::::\n" % (model.type, model.name, chains_ids)
    seqs = [''.join(v) for _, v in sequences]
    seqs = "/\n".join(seqs) + "*"
    return header + structure + seqs + "\n"


class Template(object):

    def __init__(self, struct):
        self._struct = struct

    @property
    def type(self):
        return "structureX"

    @property
    def name(self):
        return self._struct.pdb

    @property
    def sequences(self):
        seqs = []
        for c in self._struct._chains:
            seqs.append((c.chain_id, [res.name for res in c]))
        return seqs

    def __str__(self):
        return ali_template(self)


class Mutant(object):

    def __init__(self, struct, mutations):
        self._struct = struct
        self._mutations = mutations

    @property
    def type(self):
        return "sequence"

    @property
    def name(self):
        return "%s_%s" % (self._struct.pdb, "_".join([str(mut) for mut in self._mutations]))

    @property
    def sequences(self):
        seqs = {}
        chains = self._struct._chains
        for c in chains:
            seqs[c.chain_id] = [res.name for res in c]
        for mut in self._mutations:
            assert seqs[mut.chain_id][mut.i] == mut.w
            seqs[mut.chain_id][mut.i] = mut.m
        return [(c, seqs[c.chain_id]) for c in chains]

    def __str__(self):
        return ali_template(self)


class Ali(object):

    def __init__(self, template, mutant):
        self.template = template
        self.mutant = mutant

    def __str__(self):
        return "%s\n%s" % (self.template, self.mutant)


def apply_modeller(pdb_struct, mutations):
    tmpl = Template(pdb_struct)
    mutant = Mutant(pdb_struct, mutations)
    ws = "modeller/%s" % mutant.name
    dst1 = osp.join(ws, "%s.pdb" % tmpl.name)
    if not osp.exists(ws):
        os.makedirs(ws)
    if not osp.exists(dst1):
        pdb_struct.to_pdb(dst1)
    create_modeller_workspace(tmpl, mutant, ws)
    dst2 = osp.join(ws, "%s.pdb" % mutant.name)
    if osp.exists(dst2):
        return mutant.name, ws
    cline = "cd %s; %s mutate-model.py" % (ws, sys.executable)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    src2 = glob(osp.join(ws, "%s*.pdb" % mutant.name))[0]
    with open(src2, "r") as f:
        ids = [c.chain_id for c in pdb_struct]
        chain_dict = dict(zip(MODELLER_CHAINS, ids))
        struct = parse_pdb(mutant.name, f, chain_dict)
        struct.to_pdb(dst2)
    return mutant.name, ws
