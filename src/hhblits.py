import os
import sys
import io
import glob

from datetime import datetime
from datetime import timedelta

from numpy import unique

from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import parse as parse_fasta

from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

from consts import *

import argparse


out_dir = "./hhblits"
prefix_hhsuite = "/usr/share/hhsuite"
prefix_blast = "/usr/bin"
# hhblits_dbname = "pdb70"
hhblits_dbname = "uniprot20_2016_02"

batch_size = 2
num_cpu = 2
max_filter = 20000
coverage = 0
mact = 0.9

GAP = '-'
NOW = datetime.utcnow()
IGNORE = [aa for aa in map(str.lower, AA.aa2index.keys())] + [GAP]  # ignore deletions + insertions


# def prepare_uniprot20():
#     if not os.path.exists("dbs/uniprot20_2016_02"):
#         os.system("tar -xvzf dbs/uniprot20_2016_02.tgz -C dbs")


#    CREDIT TO SPIDER2
def read_pssm(pssm_file):
        # this function reads the pssm file given as input, and returns a LEN x 20 matrix of pssm values.

        # index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
        # idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)

        # open the two files, read in their data and then close them
        if pssm_file == 'STDIN': fp = sys.stdin
        else: fp = open(pssm_file, 'r')
        lines = fp.readlines()
        fp.close()

        # declare the empty dictionary with each of the entries
        aa = []
        pssm = []

        # iterate over the pssm file and get the needed information out
        for line in lines:
                split_line = line.split()
                # valid lines should have 32 points of data.
                # any line starting with a # is ignored
                try: int(split_line[0])
                except: continue

                if line[0] == '#': continue

                aa_temp = split_line[1]
                aa.append(aa_temp)
                if len(split_line) in (44, 22):
                        pssm_temp = [float(i) for i in split_line[2:22]]
                elif len(line) > 70:  # in case double digits of pssm
                        pssm_temp = [float(line[k*3+9: k*3+12]) for k in range(20)]
                        pass
                else: continue
                pssm.append({AA.index2aa[k]: pssm_temp[k] for k in range(20)})

        return aa, pssm


def _set_unique_ids(input_file, output_file):
    with open(input_file, "rt") as fin:
        with open(output_file, "wt") as fout:
            for j, line in enumerate(fin):
                prefix = line.split()[0]
                if j > 0: line = line.replace(prefix, "0" * (20 - len(str(j))) + "%d" % j)
                fout.write(line)


def _run_hhblits_batched(sequences, collection):

    records = [SeqRecord(Seq(seq), seqid) for (seqid, seq) in sequences]
    pbar = tqdm(range(len(records)), desc="sequences processed")

    while records:
        batch = []
        while (len(batch) < batch_size) and (len(records) > 0):
            seq = records.pop()
            pbar.update(1)
            batch.append(seq)

        pwd = os.getcwd()
        os.chdir(out_dir)

        if len(glob.glob('*.seq')): os.system("rm *.seq")

        sequences_fasta = 'batch.fasta'
        SeqIO.write(batch, open(sequences_fasta, 'w+'), "fasta")
        cline = "%s/scripts/splitfasta.pl %s 1>/dev/null 2>&1" \
                % (prefix_hhsuite, sequences_fasta)
        assert os.WEXITSTATUS(os.system(cline)) == 0

        if len(glob.glob('*.a3m')): os.system("rm *.a3m")

        hhblits_cmd = "%s/bin/hhblits -i $file -d ../dbs/%s/%s -oa3m $name.a3m -n 2 -maxfilt %d -mact %s"\
                      % (prefix_hhsuite, hhblits_dbname, hhblits_dbname, max_filter, mact)
        cline = "%s/scripts/multithread.pl \'*.seq\' \'%s\' -cpu %d 1>/dev/null 2>&1" \
                % (prefix_hhsuite, hhblits_cmd, num_cpu)
        assert os.WEXITSTATUS(os.system(cline)) == 0

        if coverage > 0:
            if len(glob.glob('*.fil')): os.system("rm *.fil")
            hhfilter_cmd = "%s/bin/hhfilter -i $file -o $name.fil -cov %d" \
                           % (prefix_hhsuite, coverage)
            cline = "%s/scripts/multithread.pl \'*.a3m\' \'%s\' -cpu %d 1>/dev/null 2>&1" \
                    % (prefix_hhsuite, hhfilter_cmd, num_cpu)
            assert os.WEXITSTATUS(os.system(cline)) == 0

        suffix = 'fil' if coverage > 0 else 'a3m'

        if output_fasta:
            reformat_cmd = "%s/scripts/reformat.pl -r a3m fas $file $name.fas" % prefix_hhsuite
            cline = "%s/scripts/multithread.pl \'*.%s\' \'%s\' -cpu %d 1>/dev/null 2>&1" \
                    % (prefix_hhsuite, suffix, reformat_cmd, num_cpu)
            assert os.WEXITSTATUS(os.system(cline)) == 0

        reformat_cmd = "%s/scripts/reformat.pl -r a3m psi $file $name.psi" % prefix_hhsuite
        cline = "%s/scripts/multithread.pl \'*.%s\' \'%s\' -cpu %d 1>/dev/null 2>&1"\
                % (prefix_hhsuite, suffix, reformat_cmd, num_cpu)
        assert os.WEXITSTATUS(os.system(cline)) == 0

        e = ThreadPoolExecutor(num_cpu)
        for (seq, aln) in e.map(_get_aln, [seq for seq in batch if os.path.exists("%s.a3m" % seq.id)]):
            collection.update_one({
                "_id": seq.id}, {
                '$set': {"alignment": aln,
                         "seq": str(seq.seq),
                         "length": len(seq.seq),
                         "updated_at": datetime.utcnow()}
            }, upsert=True)

        if cleanup: os.system("rm ./*")

        os.chdir(pwd)

    pbar.close()


def _get_profile_func(method="pssm"):
    func = _get_pssm if method == "pssm" else _get_profile

    def _comp_profile_batched(sequences, collection):

        records = [SeqRecord(Seq(seq), seqid) for (seqid, seq) in sequences]
        pbar = tqdm(range(len(records)), desc="sequences processed")

        while records:
            batch = []
            while (len(batch) < batch_size) and (len(records) > 0):
                seq = records.pop()
                pbar.update(1)
                batch.append(seq)

            pwd = os.getcwd()
            os.chdir(out_dir)

            e = ThreadPoolExecutor(num_cpu)
            for (seq, attr) in e.map(func, [seq for seq in batch]):
                if not attr: continue
                collection.update_one({
                    "_id": seq.id}, {
                    '$set': {method: attr,
                             "seq": str(seq.seq),
                             "length": len(seq.seq),
                             "updated_at": datetime.utcnow()}
                }, upsert=True)

            os.chdir(pwd)

        pbar.close()

    return _comp_profile_batched


def _read_a3m(seq):
    return seq, open("%s.a3m" % str(seq.id), 'r').read()


# MUST BE RUN AFTER HHBLITS FINISHED
def _get_aln(seq):

    # cline = "%s/scripts/addss.pl %s.a3m" % (prefix_hhsuite, seq.id)
    # assert os.WEXITSTATUS(os.system(cline)) == 0

    # cline = "%s/scripts/reformat.pl -r a3m psi %s.a3m %s.psi" \
    #         % (prefix_hhsuite, seq.id, seq.id)
    # assert os.WEXITSTATUS(os.system(cline)) == 0

    _set_unique_ids("%s.psi" % seq.id, "%s.msa" % seq.id)   # can assume there is a file like that

    aln = []
    with open("%s.psi" % seq.id, 'rt') as f:
        for line in f.readlines():
            r = line.strip().split()
            assert len(r) == 2
            aln.append(r)

    return seq, aln


def _get_pssm(seq):

    doc = db.skempi.find_one({"_id": seq.id})
    if not doc or not "alignment" in doc:
        return None, None
    aln = doc["alignment"]
    lines = [' '.join([uid, homo]) for uid, homo in aln]
    with open("%s.msa" % seq.id, 'w+') as f:
        f.writelines(lines)

    SeqIO.write(seq, open("%s.seq" % seq.id, 'w+'), "fasta")

    cline = "%s/psiblast -subject %s.seq -in_msa %s.msa -out_ascii_pssm %s.pssm 1>psiblast.out 2>&1" \
            % (prefix_blast, seq.id, seq.id, seq.id)
    assert os.WEXITSTATUS(os.system(cline)) == 0

    # aln = list(AlignIO.parse(open("%s.fas" % seq.id, 'r'), "fasta"))
    # pssm = SummaryInfo(aln[0]).pos_specific_score_matrix(chars_to_ignore=IGNORE)
    aa, pssm = read_pssm("%s.pssm" % seq.id)

    return seq, pssm


def _get_profile(seq):

    doc = db.skempi.find_one({"_id": seq.id})
    if not doc or not "alignment" in doc:
        return None, None
    aln = doc["alignment"]

    profile = []
    for pos in range(len(seq)):
        total_count = 0
        pos_profile = {AA.index2aa[ix]: 0. for ix in range(20)}
        for _, homo in aln:
            aa = homo[pos]
            if aa == GAP: continue
            if AA.aa2index[aa] < 20:
                pos_profile[aa] += 1
            elif aa == 'X':
                for k in pos_profile.keys():
                    pos_profile[k] += 0.05
            elif aa == 'B':
                pos_profile['N'] += 0.5
                pos_profile['D'] += 0.5
            elif aa == 'Z':
                pos_profile['Q'] += 0.5
                pos_profile['E'] += 0.5
            elif aa == 'O':
                pos_profile['K'] += 1.0
            elif aa == 'U':
                pos_profile['C'] += 1.0
            else: continue
            total_count += 1
        profile.append({k: v/total_count for k, v in pos_profile.items()})

    return seq, profile


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--prefix_hhsuite", type=str, default='/usr/share/hhsuite',
                        help="Specify where you installed hhsuite.")
    parser.add_argument("--prefix_blast", type=str, default='/usr/bin',
                        help="Specify where you installed ncbi blast.")
    parser.add_argument("--limit", type=int, default=None,
                        help="How many sequences for PSSM computation.")
    parser.add_argument("--max_filter", type=int, default=20000,
                        help="How many sequences to include in the MSA for PSSM computation.")
    parser.add_argument("--num_cpu", type=int, default=2,
                        help="How many cpus for computing PSSM (when running in batched mode).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="How many sequences in batch (when running in parallel mode).")
    parser.add_argument("--coverage", type=int, default=0,
                        help="The desired coverage (for the alignment algorithm).")
    parser.add_argument("--mact", type=float, default=0.9,
                        help="Set the Max ACC (mact) threshold (for the alignment algorithm).")
    parser.add_argument('--keep_files', action='store_true', default=False,
                        help="Whether to keep intermediate files.")
    parser.add_argument('--dd', type=int, default=180,
                        help="How many days before records are considered obsolete?")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Supply an input file in FASTA format.")
    parser.add_argument('--out_dir', default='./hhblits', type=str, metavar='PATH',
                        help='Where to save the output/log files?')
    parser.add_argument("--comp", type=str, required=True, choices=['hhblits', 'profile', 'pssm'],
                        help="The name of the computation that you want run.")
    parser.add_argument("--db_name", type=str, default='prot2vec', choices=['prot2vec', 'prot2vec2'],
                        help="The name of the DB to which to write the data.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    num_cpu = args.num_cpu
    batch_size = args.batch_size
    max_filter = args.max_filter
    coverage = args.coverage
    mact = args.mact
    prefix_blast = args.prefix_blast
    prefix_hhsuite = args.prefix_hhsuite

    cleanup = not args.keep_files
    output_fasta = args.keep_files

    os.environ['HHLIB'] = prefix_hhsuite

    client = MongoClient(args.mongo_url)
    db = client[args.db_name]
    lim = args.limit

    func = _run_hhblits_batched if args.comp == 'hhblits' else _get_profile_func(args.comp)

    fasta_fname = args.input_file
    fasta_src = parse_fasta(open(fasta_fname, 'r'), 'fasta')
    seqs = [(r.id, str(r.seq)) for r in fasta_src]
    db.skempi.create_index("updated_at")
    func(seqs, db.skempi)
