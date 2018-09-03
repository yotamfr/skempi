import os.path as osp
import pandas as pd

# PDB_PATH = "../data/pdbs_n"
PDB_PATH = "../data/pdbs"

try:
    skempi_df = pd.read_excel(osp.join('../data', 'SKEMPI_1.1.xlsx'))
    skempi2_df = pd.read_csv(osp.join('../data', 'skempi_v2.csv'))
except IOError as e:
    print("warning: %s" % e)
    skempi_df = None
    skempi2_df = None

from pymongo import MongoClient

mongo_url = "mongodb://localhost:27017/"
db_name = "prot2vec"
client = MongoClient(mongo_url)
db = client[db_name]
collection_msa = db.skempi_uniprot20
# collection_msa = db.skempi_scop90
# collection_msa = db.skempi_uniclust30

NUM_GROUPS = 5

MODELLER_CHAINS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

G1 = [
    "%s_%s_%s" % (s[:4], s[4], s[5].strip()) for s in
    """
    1CSOEI 1CT0EI 1CT2EI 1CT4EI 1SGDEI 1SGEEI 1SGNEI 1SGPEI 1SGQEI 1SGYEI 2NU0EI
    2NU1EI 2NU2EI 2NU4EI 2SGPEI 2SGQEI 3SGBEI 1IARAB 1XD3AB 1F47AB 1ACBEI
    1H9DAB 2HRKAB 3BP8AC 2OOBAB
    """.split(' ') if s.strip()]

G2 = [
    "%s_%s_%s" % (s[:4], s[4], s[5].strip()) for s in
    """
    1JTGAB 1S0WAC 2G2UAB 2G2WAB 1A4YAB 1Z7XWX 2GOXAB 2NOJAB 3D5RAC
    3D5SAC 1KACAB 1P69AB 1P6AAB 3BK3AC 1JCKAB 1SBBAB 4CPAAI 1S1QAB 2B42AB
    1E96AB 2I26NL
    """.split(' ') if s.strip()]


G3 = [
    "%s_%s_%s" % (s[:4], s[4], s[5].strip()) for s in
    """
    1PPFEI 1CSEEI 1SBNEI 1TM1EI 1TM3EI 1TM4EI 1TM5EI 1TM7EI 1TMGEI 1TO1EI 1Y1KEI
    1Y33EI 1Y34EI 1Y3BEI 1Y4AEI 1GC1GC 2SICEI 2O3BAB 1FC2CD 2BTFAP 1EFNAB
    2A9KAB
    """.split(' ') if s.strip()]

G4 = [
    "%s_%s_%s" % (s[:4], s[4], s[5].strip()) for s in
    """
    1R0REI 1EAWAB 2FTLEI 3BTDEI 3BTEEI 3BTFEI 3BTGEI 3BTHEI 3BTMEI 3BTQEI
    3BTTEI 3BTWEI 1AK4AD 1M9EAD 2J0TAD 1FFWAB 1MAHAF 1UUZAD 1SMFEI 2AJFAE
    2J1KCT
    """.split(' ') if s.strip()]

G5 = [
    "%s_%s_%s" % (s[:4], s[4], s[5].strip()) for s in
    """
    1B2SAD 1B2UAD 1B3SAD 1BRSAD 1X1XAD 1EMVAB 2VLNAB 2VLOAB 2VLQAB
    2WPTAB 1A22AB 2B0ZAB 2B10AB 2B11AB 2B12AB 2PCBAB 2PCCAB 1KTZAB 1LFDAB
    1FCCAC 1GL0EI 1GL1AI 1HE8AB 2HLEAB 2I9BAE
    """.split(' ') if s.strip()]
