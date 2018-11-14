# Requires python 2.7
from cogent.parse.aaindex import *
from StringIO import StringIO
import pandas as pd
from itertools import combinations as comb

BLOSUM62_STR = r"""
H HENS920102
D BLOSUM62 substitution matrix (Henikoff-Henikoff, 1992)
R PMID:1438297
A Henikoff, S. and Henikoff, J.G.
T Amino acid substitution matrices from protein blocks
J Proc. Natl. Acad. Sci. USA 89, 10915-10919 (1992)
* matrix in 1/3 Bit Units
M rows = ARNDCQEGHILKMFPSTWYV, cols = ARNDCQEGHILKMFPSTWYV
      6.
     -2.      8.
     -2.     -1.      8.
     -3.     -2.      2.      9.
     -1.     -5.     -4.     -5.     13.
     -1.      1.      0.      0.     -4.      8.
     -1.      0.      0.      2.     -5.      3.      7.
      0.     -3.     -1.     -2.     -4.     -3.     -3.      8.
     -2.      0.      1.     -2.     -4.      1.      0.     -3.     11.
     -2.     -4.     -5.     -5.     -2.     -4.     -5.     -6.     -5.      6.
     -2.     -3.     -5.     -5.     -2.     -3.     -4.     -5.     -4.      2.      6.
     -1.      3.      0.     -1.     -5.      2.      1.     -2.     -1.     -4.     -4.      7.
     -1.     -2.     -3.     -5.     -2.     -1.     -3.     -4.     -2.      2.      3.     -2.      8.
     -3.     -4.     -4.     -5.     -4.     -5.     -5.     -5.     -2.      0.      1.     -5.      0.      9.
     -1.     -3.     -3.     -2.     -4.     -2.     -2.     -3.     -3.     -4.     -4.     -2.     -4.     -5.     11.
      2.     -1.      1.      0.     -1.      0.      0.      0.     -1.     -4.     -4.      0.     -2.     -4.     -1.      6.
      0.     -2.      0.     -2.     -1.     -1.     -1.     -2.     -3.     -1.     -2.     -1.     -1.     -3.     -2.      2.      7.
     -4.     -4.     -6.     -6.     -3.     -3.     -4.     -4.     -4.     -4.     -2.     -4.     -2.      1.     -5.     -4.     -4.     16.
     -3.     -3.     -3.     -5.     -4.     -2.     -3.     -5.      3.     -2.     -2.     -3.     -1.      4.     -4.     -3.     -2.      3.     10.
      0.     -4.     -4.     -5.     -1.     -3.     -4.     -5.     -5.      4.      1.     -3.      1.     -1.     -4.     -2.      0.     -4.     -2.      6.
//
"""

BASU010101_STR = r"""
H BASU010101
D Optimization-based potential derived by the modified perceptron criterion
R PMID:11391771
A Bastolla, U., Farwer, J., Knapp, E.W. and Vendruscolo, M.
T How to guarantee optimal stability for most representative 
  structures in the protein data bank
J Proteins 44, 79-96 (2001)
M rows = ARNDCQEGHILKMFPSTWYV, cols = ARNDCQEGHILKMFPSTWYV
  -0.0479
   0.1049  0.0306
   0.1049 -0.0150 -0.0917
   0.1018 -0.1859  0.0192  0.0840
  -0.1085  0.0544 -0.0844  0.1169 -1.0442
  -0.0457  0.0059 -0.0050 -0.0728  0.0715 -0.0550
   0.1346 -0.3511  0.1146  0.1581  0.1550 -0.0413  0.1259
   0.1844 -0.0251  0.1196  0.1115 -0.0982  0.1710  0.2311  0.2219
   0.0266 -0.0184  0.0386 -0.0749 -0.0701 -0.0125 -0.0827  0.0979  0.0005
  -0.0737 -0.0266  0.1485  0.1892 -0.2235 -0.0480  0.1103  0.1174 -0.0326 -0.5852
  -0.1711 -0.0651  0.0890  0.2673 -0.1305 -0.0172  0.0802  0.0782 -0.0169 -0.5112 -0.5067
   0.0691  0.0839 -0.0381 -0.1154 -0.0330 -0.0735 -0.2403  0.1963  0.0390  0.0682  0.0543  0.1216
  -0.0847 -0.0163  0.0124 -0.0197 -0.0557 -0.1038  0.0637 -0.0573 -0.0345 -0.2137 -0.1822  0.0866 -0.1059
  -0.1119 -0.0904  0.0018  0.0827 -0.3262 -0.0171  0.0885  0.0789 -0.1250 -0.3791 -0.5450 -0.0416 -0.1785 -0.3088
   0.1462 -0.0614  0.1560  0.2386  0.0545  0.1127  0.2241  0.2131  0.0295  0.0882  0.0745  0.1099 -0.0069 -0.0604  0.1077
   0.0464  0.0442  0.1452  0.0424 -0.0132  0.1169  0.0823  0.1075 -0.0005  0.0332  0.0959  0.1690  0.0185  0.0398  0.1626  0.0941
   0.0310 -0.0210  0.0155  0.1043 -0.0013 -0.0243  0.0675  0.1763  0.0681 -0.0700 -0.0316  0.0467  0.0018 -0.1120  0.1908  0.0228  0.0150
  -0.0880 -0.2070 -0.0250 -0.0124 -0.1176 -0.0540 -0.0967 -0.1567 -0.0200 -0.1961 -0.2639 -0.1152 -0.0775 -0.3405 -0.0910 -0.0802  0.0052 -0.1066
  -0.1408 -0.1369 -0.1149 -0.1165 -0.2444 -0.1431 -0.0522 -0.0176 -0.1976 -0.3164 -0.2614 -0.1120 -0.1621 -0.4212 -0.1326  0.0214 -0.1445 -0.3209 -0.2793
  -0.1431  0.0475  0.1180  0.2728 -0.2349  0.1061  0.1010  0.1859 -0.0039 -0.4223 -0.4593  0.0609 -0.2127 -0.4001  0.0868  0.1766  0.0119 -0.2898 -0.2792 -0.5193
//
"""

SKOJ970101_STR = r"""
H SKOJ970101
D Statistical potential derived by the quasichemical approximation
R PMID:9070450
A Skolnick, J., Jaroszewski, L., Kolinski, A. and Godzik, A.
T Derivation and testing of pair potentials for protein folding.
  When is the quasichemical approximation correct?
J Protein Science 6, 676-688 (1997)
M rows = ARNDCQEGHILKMFPSTWYV, cols = ARNDCQEGHILKMFPSTWYV
   0.8
   0.6 -0.1
   0.9  0.0  0.1
   1.2 -0.5  0.4  0.9
   0.6  0.7  0.6  0.8 -1.3
   0.6  0.2  0.2  0.6  0.3  0.3
   1.2 -0.4  0.5  0.9  0.9  0.8  1.2
   1.2  0.2  0.7  0.8  1.3  0.8  1.1  1.5
   0.4 -0.1  0.2 -0.1  0.1  0.1  0.1  0.7 -0.8
  -0.6 -0.2  0.5  0.5 -0.5  0.1  0.4  0.4 -0.1 -1.4
  -0.3 -0.1  0.4  0.7 -0.4  0.2  0.6  0.4 -0.1 -1.3 -1.2
   1.3  1.1  0.7  0.2  1.3  0.5  0.0  0.9  1.0  0.5  0.5  2.1
  -0.3  0.2  0.3  0.6 -0.3  0.1  0.4  0.5 -0.4 -1.0 -1.0  0.6 -1.1
  -0.2 -0.4  0.1  0.5 -0.6 -0.1  0.4  0.3 -0.4 -1.3 -1.3  0.4 -1.3 -1.5
   0.6  0.1  0.5  0.9  0.4  0.2  0.7  0.8  0.0  0.0  0.0  0.9 -0.2 -0.1  0.4
   0.9  0.2  0.7  0.7  0.6  0.6  0.6  0.8  0.0  0.3  0.4  0.9  0.4  0.1  0.6  0.6
   0.5  0.0  0.3  0.5  0.5  0.4  0.4  0.5  0.1 -0.3  0.0  0.8  0.0 -0.2  0.2  0.4  0.1
  -0.6 -0.6  0.0  0.0 -0.7 -0.4 -0.1  0.0 -0.9 -1.3 -1.4 -0.1 -1.5 -1.5 -0.8 -0.1 -0.2 -1.2
  -0.4 -0.7 -0.2 -0.2 -0.1 -0.3 -0.2  0.1 -0.8 -1.0 -0.9 -0.2 -1.1 -1.0 -0.5  0.1 -0.2 -1.2 -0.8
  -0.4  0.0  0.5  1.0 -0.6  0.2  0.5  0.5  0.0 -1.2 -1.2  0.7 -0.8 -1.1 -0.1  0.4 -0.2 -1.1 -0.8 -1.2
//
"""

ARGP820101_STR = """
H ARGP820101
D Hydrophobicity index (Argos et al., 1982)
R PMID:7151796
A Argos, P., Rao, J.K.M. and Hargrave, P.A.
T Structural prediction of membrane-bound proteins
J Eur. J. Biochem. 128, 565-575 (1982)
C JOND750101    1.000  SIMZ760101    0.967  GOLD730101    0.936
  TAKK010101    0.906  MEEJ810101    0.891  ROSM880104    0.872
  CIDH920105    0.867  LEVM760106    0.865  CIDH920102    0.862
  MEEJ800102    0.855  MEEJ810102    0.853  ZHOH040101    0.841
  CIDH920103    0.827  PLIV810101    0.820  CIDH920104    0.819
  LEVM760107    0.806  NOZY710101    0.800  GUYH850103   -0.808
  PARJ860101   -0.835  WOLS870101   -0.838  BULH740101   -0.854
I    A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
    0.61    0.60    0.06    0.46    1.07      0.    0.47    0.07    0.61    2.22
    1.53    1.15    1.18    2.02    1.95    0.05    0.05    2.65    1.88    1.32
//
"""

FASG760101_STR = """
H FASG760101
D Molecular weight (Fasman, 1976)
R 
A Fasman, G.D., ed.
T 
J "Handbook of Biochemistry and Molecular Biology", 3rd ed., Proteins - Volume 
  1, CRC Press, Cleveland (1976)
C FAUJ880103    0.979  CHOC760101    0.978  LEVM760102    0.966
  CHAM820101    0.962  CHOC750101    0.956  LEVM760105    0.951
  PONJ960101    0.945  CHAM830106    0.943  TSAJ990102    0.940
  TSAJ990101    0.935  BIGC670101    0.919  GOLD730102    0.918
  KRIW790103    0.910  HARY940101    0.910  GRAR740103    0.908
  FAUJ880106    0.899  RADA880106    0.870  WOLS870102    0.866
  MCMT640101    0.845  CHAM830105    0.839  ROSG850101    0.838
  DAWD720101    0.833  FAUJ880104    0.825  OOBM770102    0.821
  LEVM760107    0.815  RADA880103   -0.954
I    A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
   89.09  174.20  132.12  133.10  121.15  146.15  147.13   75.07  155.16  131.17
  131.17  146.19  149.21  165.19  115.13  105.09  119.12  204.24  181.19  117.15
//
"""

JURD980101_STR = """
H JURD980101
D Modified Kyte-Doolittle hydrophobicity scale (Juretic et al., 1998)
R 
A Juretic, D., Lucic, B., Zucic, D. and Trinajstic, N.
T Protein transmembrane structure: recognition and prediction by using 
  hydrophobicity scales through preference functions
J Theoretical and Computational Chemistry, 5, 405-445 (1998)
C KYTJ820101    0.996  CHOC760103    0.967  OLSK800101    0.943
  NADH010102    0.931  JANJ780102    0.928  NADH010101    0.925
  EISD860103    0.901  DESM900102    0.900  NADH010103    0.900
  EISD840101    0.895  RADA880101    0.893  MANP780101    0.887
  WOLR810101    0.881  PONP800103    0.879  JANJ790102    0.879
  NADH010104    0.873  BASU050103    0.871  CHOC760104    0.870
  PONP800102    0.869  JANJ790101    0.868  WOLR790101    0.864
  MEIH800103    0.861  PONP800101    0.858  NAKH920108    0.858
  RADA880108    0.857  PONP800108    0.856  COWR900101    0.855
  ROSG850102    0.854  CORJ870101    0.849  PONP930101    0.849
  RADA880107    0.842  BLAS910101    0.841  BIOV880101    0.840
  MIYS850101    0.837  FAUJ830101    0.833  CIDH920104    0.832
  BASU050101    0.830  DESM900101    0.829  WARP780101    0.827
  KANM800104    0.826  LIFS790102    0.824  RADA880104    0.824
  NADH010105    0.821  ROSM880105    0.818  NISK800101    0.816
  CORJ870104    0.812  NISK860101    0.808  CORJ870103    0.808
  BIOV880102    0.805  CORJ870107    0.804  ARGP820102    0.802
  ARGP820103    0.800  CORJ870108   -0.806  MIYS990104   -0.813
  VHEG790101   -0.814  KRIW790101   -0.824  MIYS990105   -0.829
  MIYS990103   -0.845  CHOC760102   -0.851  ROSM880101   -0.851
  MIYS990101   -0.852  MONM990101   -0.853  JANJ780103   -0.853
  MIYS990102   -0.853  RACS770102   -0.855  FASG890101   -0.857
  ENGD860101   -0.861  PRAM900101   -0.862  JANJ780101   -0.862
  GUYH850101   -0.864  GRAR740102   -0.864  PUNT030102   -0.869
  MEIH800102   -0.879  GUYH850104   -0.880  KUHL950101   -0.884
  PUNT030101   -0.884  ROSM880102   -0.894  GUYH850105   -0.900
  OOBM770101   -0.903
I    A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
    1.10   -5.10   -3.50   -3.60    2.50   -3.68   -3.20   -0.64   -3.20    4.50
    3.80   -4.11    1.90    2.80   -1.90   -0.50   -0.70   -0.46    -1.3     4.2
//
"""

KYTJ820101_STR = """
H KYTJ820101
D Hydropathy index (Kyte-Doolittle, 1982)
R PMID:7108955
A Kyte, J. and Doolittle, R.F.
T A simple method for displaying the hydropathic character of a protein
J J. Mol. Biol. 157, 105-132 (1982)
C JURD980101    0.996  CHOC760103    0.964  OLSK800101    0.942
  JANJ780102    0.922  NADH010102    0.920  NADH010101    0.918
  DESM900102    0.898  EISD860103    0.897  CHOC760104    0.889
  NADH010103    0.885  WOLR810101    0.885  RADA880101    0.884
  MANP780101    0.881  EISD840101    0.878  PONP800103    0.870
  WOLR790101    0.869  NAKH920108    0.868  JANJ790101    0.867
  JANJ790102    0.866  BASU050103    0.863  PONP800102    0.861
  MEIH800103    0.856  NADH010104    0.856  PONP800101    0.851
  PONP800108    0.850  CORJ870101    0.848  WARP780101    0.845
  COWR900101    0.845  PONP930101    0.844  RADA880108    0.842
  ROSG850102    0.841  DESM900101    0.837  BLAS910101    0.836
  BIOV880101    0.829  RADA880107    0.828  BASU050101    0.826
  KANM800104    0.824  LIFS790102    0.824  CIDH920104    0.824
  MIYS850101    0.821  RADA880104    0.819  NAKH900111    0.817
  CORJ870104    0.812  NISK800101    0.812  FAUJ830101    0.811
  ROSM880105    0.806  ARGP820103    0.806  CORJ870103    0.806
  NADH010105    0.804  NAKH920105    0.803  ARGP820102    0.803
  CORJ870107    0.801  MIYS990104   -0.800  CORJ870108   -0.802
  KRIW790101   -0.805  MIYS990105   -0.818  MIYS990103   -0.833
  CHOC760102   -0.838  MIYS990101   -0.840  MIYS990102   -0.840
  MONM990101   -0.842  GUYH850101   -0.843  FASG890101   -0.844
  RACS770102   -0.844  ROSM880101   -0.845  JANJ780103   -0.845
  ENGD860101   -0.850  PRAM900101   -0.850  JANJ780101   -0.852
  GRAR740102   -0.859  PUNT030102   -0.862  GUYH850104   -0.869
  MEIH800102   -0.871  PUNT030101   -0.872  ROSM880102   -0.878
  KUHL950101   -0.883  GUYH850105   -0.883  OOBM770101   -0.899
I    A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
     1.8    -4.5    -3.5    -3.5     2.5    -3.5    -3.5    -0.4    -3.2     4.5
     3.8    -3.9     1.9     2.8    -1.6    -0.8    -0.7    -0.9    -1.3     4.2
//
"""

iPTMs1_STR = """
AA A C D E F G H I K L M N P Q R S T V W Y
A 0.297 0.061 0.033 0.039 0.036 0.115 0.044 0.039 0.040 0.044 0.057 0.041 0.065 0.043 0.036 0.113 0.066 0.059 0.026 0.029
C 0.013 0.510 0.003 0.003 0.004 0.004 0.004 0.008 0.004 0.007 0.009 0.009 0.017 0.004 0.004 0.014 0.014 0.010 0.004 0.004
D 0.017 0.007 0.419 0.057 0.008 0.010 0.025 0.012 0.029 0.012 0.015 0.042 0.016 0.032 0.025 0.028 0.024 0.013 0.006 0.010
E 0.047 0.016 0.128 0.318 0.024 0.014 0.046 0.046 0.128 0.042 0.052 0.058 0.025 0.145 0.094 0.060 0.065 0.044 0.018 0.030
F 0.018 0.009 0.008 0.010 0.392 0.009 0.032 0.025 0.012 0.024 0.040 0.013 0.014 0.015 0.011 0.011 0.014 0.020 0.107 0.109
G 0.083 0.013 0.014 0.009 0.014 0.654 0.023 0.008 0.010 0.008 0.013 0.023 0.022 0.014 0.012 0.037 0.015 0.011 0.015 0.012
H 0.013 0.006 0.014 0.011 0.019 0.009 0.414 0.009 0.011 0.008 0.009 0.020 0.011 0.015 0.016 0.011 0.012 0.008 0.014 0.025
I 0.038 0.035 0.022 0.038 0.050 0.011 0.029 0.213 0.054 0.083 0.094 0.072 0.017 0.046 0.037 0.031 0.050 0.130 0.025 0.032
K 0.037 0.019 0.051 0.099 0.022 0.012 0.034 0.050 0.207 0.047 0.043 0.056 0.020 0.095 0.124 0.043 0.046 0.045 0.015 0.022
L 0.115 0.084 0.058 0.093 0.125 0.027 0.071 0.220 0.135 0.447 0.236 0.133 0.028 0.116 0.088 0.084 0.118 0.180 0.077 0.111
M 0.020 0.014 0.010 0.015 0.028 0.006 0.011 0.033 0.016 0.031 0.161 0.015 0.005 0.022 0.015 0.013 0.018 0.022 0.012 0.018
N 0.028 0.029 0.054 0.033 0.018 0.021 0.047 0.050 0.041 0.035 0.030 0.247 0.014 0.036 0.035 0.038 0.043 0.054 0.010 0.026
P 0.017 0.022 0.008 0.006 0.008 0.008 0.010 0.005 0.006 0.003 0.004 0.005 0.597 0.005 0.008 0.012 0.014 0.008 0.007 0.006
Q 0.030 0.014 0.042 0.086 0.021 0.014 0.036 0.033 0.073 0.031 0.046 0.037 0.014 0.221 0.061 0.032 0.038 0.028 0.015 0.019
R 0.026 0.012 0.035 0.058 0.017 0.012 0.041 0.028 0.100 0.025 0.033 0.038 0.023 0.064 0.318 0.031 0.037 0.025 0.016 0.022
S 0.072 0.041 0.034 0.032 0.014 0.033 0.025 0.021 0.030 0.021 0.024 0.036 0.029 0.029 0.027 0.303 0.065 0.026 0.011 0.026
T 0.046 0.047 0.032 0.039 0.019 0.014 0.029 0.036 0.035 0.032 0.037 0.045 0.037 0.037 0.035 0.071 0.274 0.044 0.015 0.021
V 0.065 0.052 0.026 0.041 0.044 0.016 0.032 0.146 0.054 0.076 0.070 0.088 0.032 0.044 0.037 0.045 0.069 0.257 0.016 0.034
W 0.005 0.003 0.002 0.003 0.040 0.004 0.009 0.005 0.003 0.005 0.007 0.003 0.005 0.004 0.004 0.003 0.004 0.003 0.535 0.023
Y 0.013 0.009 0.009 0.011 0.097 0.007 0.038 0.014 0.010 0.019 0.023 0.017 0.009 0.012 0.013 0.018 0.013 0.014 0.054 0.420
"""


class _AAIndex2(object):

    def __init__(self, stream, name):
        self._dict = list(AAIndex2Parser()(stream))[0].Data
        self._name = name

    def __getitem__(self, t):
        row, col = t
        ret = self._dict[row][col]
        if ret is None:
            ret = self._dict[col][row]
        return ret

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)


class _AAIndex1(object):

    def __init__(self, stream, name):
        self._dict = list(AAIndex1Parser()(stream))[0].Data
        self._name = name

    def __getitem__(self, aa):
        ret = self._dict[aa]
        return ret

    def __str__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)


amino_acids = "ARNDCQEGHILKMFPSTWYV"

ARGP820101 = _AAIndex1(StringIO(ARGP820101_STR), "ARGP820101")  # Hydrophobicity

FASG760101 = _AAIndex1(StringIO(FASG760101_STR), "FASG760101")  # Molecular weight

KYTJ820101 = _AAIndex1(StringIO(KYTJ820101_STR), "KYTJ820101")  # Kyte-Doolittle, 1982

JURD980101 = _AAIndex1(StringIO(JURD980101_STR), "JURD980101")  # Modified Kyte-Doolittle (Juretic et al., 1998)

BLOSUM62 = _AAIndex2(StringIO(BLOSUM62_STR), "BLOSUM62")

BASU010101 = _AAIndex2(StringIO(BASU010101_STR), "BASU010101")  # Bastolla-Vendruscolo (Bastolla et al., 2001).

SKOJ970101 = _AAIndex2(StringIO(SKOJ970101_STR), "SKOJ970101")  # Skolnick (et al 1997)


iPTMs1 = {}
iptm_df = pd.read_csv(StringIO(iPTMs1_STR), sep=" ")
for r1, r2 in comb(amino_acids, 2):
    ix1 = iptm_df.AA == r1
    ix2 = iptm_df.AA == r2
    iPTMs1[(r1, r2)] = iptm_df[ix1][r2].values[0]
    iPTMs1[(r2, r1)] = iptm_df[ix2][r1].values[0]
    iPTMs1[(r1, r1)] = iptm_df[ix1][r1].values[0]
    iPTMs1[(r2, r2)] = iptm_df[ix2][r2].values[0]

MaxASA_emp = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "E": 214.0,
    "Q": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0
}

MaxASA_theory = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "E": 223.0,
    "Q": 225.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0
}

if __name__ == "__main__":
    import numpy as np
    keys = [(a1, a2) for a1 in amino_acids for a2 in amino_acids]
    min_key = keys[np.argmin([BASU010101[k] for k in keys])]
    print(min_key, BASU010101[min_key])
    print(BASU010101[('A', 'R')])
    print(SKOJ970101[('A', 'K')])
    print(ARGP820101['A'])
    print(FASG760101['V'])
    print(iPTMs1[('A', 'R')])
