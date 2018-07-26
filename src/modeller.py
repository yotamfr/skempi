import os
from os import path as osp
from glob import glob
import shutil

from pdb_utils import *

from skempi_utils import *


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


def create_modeller_workspace(template, mutant, workspace):
    ali = Ali(template, mutant)
    with open(osp.join(workspace, "mutate-model.py"), 'w+') as f:
        f.write(fmt_modeller % (template.name, mutant.name))
    with open(osp.join(workspace, "complex.ali"), 'w+') as f:
        f.write(str(ali))


def ali_template(model):
    header = ">P1;%s\n" % model.name
    chains = model.chains
    chains = sorted(chains.iteritems(), key=lambda kv: kv[0])
    chains_ids = "\t".join("::%s" % (k,) for k, _ in chains)
    structure = "%s:%s\t%s::::\n" % (model.type, model.name, chains_ids)
    seqs = [''.join(v) for _, v in chains]
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
    def chains(self):
        seqs = {}
        for c, chain in self._struct.chains.iteritems():
            seqs[c] = [res.name for res in chain]
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
    def chains(self):
        seqs = {}
        for c, chain in self._struct.chains.iteritems():
            seqs[c] = [res.name for res in chain]
        for mut in self._mutations:
            assert seqs[mut.chain_id][mut.i] == mut.w
            seqs[mut.chain_id][mut.i] = mut.m
        return seqs

    def __str__(self):
        return ali_template(self)


class Ali(object):

    def __init__(self, template, mutant):
        self.template = template
        self.mutant = mutant

    def __str__(self):
        return "%s\n%s" % (self.template, self.mutant)


def apply_modeller(skempi_struct, mutations):
    tmpl = Template(skempi_struct)
    mutant = Mutant(skempi_struct, mutations)
    ws = "modeller/%s" % mutant.name
    if not osp.exists(ws): os.makedirs(ws)
    indir = "../data/pdbs"
    src = osp.join(indir, "%s.pdb" % tmpl.name)
    dst = osp.join(ws, "%s.pdb" % tmpl.name)
    shutil.copy(src, dst)
    create_modeller_workspace(tmpl, mutant, ws)
    os.system("cd %s; python mutate-model.py" % ws)
    src = glob(osp.join(ws, "%s*.pdb" % mutant.name))[0]
    dst = osp.join(ws, "%s.pdb" % mutant.name)
    shutil.move(src, dst)
    return mutant.name, ws
