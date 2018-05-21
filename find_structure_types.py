import numpy as np
import re
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator
from pymatgen.io.cif import CifParser

f = open('results.csv','r')
g = open('structuretypes_1D.txt','w')
content = f.read()
f.close()
lines = content.split('\n')[1:-1]
pattern = re.compile('[0-9]')
dim = [line.split(';')[2] for line in lines]
icsd_codes = [line.split(';')[1] for line in lines]
number_index = [line.split(';')[11] for line in lines]
structure_types = []
sm = StructureMatcher(scale=True, comparator=FrameworkComparator())
for i in range(len(icsd_codes)):
    print (i)
    if dim[i] == "1":
        structure1 = CifParser("cif_files/data_"+icsd_codes[i]+"-ICSD.cif", occupancy_tolerance=100).get_structures(primitive=False)[0]
        for j in range(i):
            if dim[j] == "1" and number_index[i] == number_index[j]:
                """if icsd_codes[i] == icsd_codes[j]:
                    structure_types.append(structure_types[j])
                    g.write(str(structure_types[-1])+'\n')
                    break"""
                structure2 = CifParser("cif_files/data_"+icsd_codes[j]+"-ICSD.cif", occupancy_tolerance=100).get_structures(primitive=False)[0]
                if sm.fit(structure1,structure2):
                    structure_types.append(structure_types[j])
                    g.write(str(structure_types[-1])+"\n")
                    break
        if len(structure_types) == i:
            structure_types.append(max(structure_types)+1)
            g.write(str(structure_types[-1])+"\n")
    else:
        structure_types.append(-1)
        g.write(str(structure_types[-1])+"\n")

g.close()
