import itertools
import numpy as np
from multiprocessing import Pool
from pymatgen.io.cif import CifParser
from filtering import Filtering
import os, re
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import symmetry_measure

f = open("results.csv","r")
lines = f.readlines()[1:]
f.close()
dim_full = [int(line.split(";")[2]) for line in lines]
icsd_full = [line.split(";")[1] for line in lines]
low_delta_full = [float(line.split(";")[3]) for line in lines]
up_delta_full = [float(line.split(";")[4]) for line in lines]


initial_string_full = []
for line in lines:
    initial_string_full.append(";".join(line.split(";")[:12])+";"+";".join(line.split(";")[-3:]))

# comment this when killing existing job and running again
#f = open("final_results.txt","w")
#print("error_code;icsd_code;dimension;lower_index;upper_index;direction;chemical_structural_formula;\
#chemical_formula_sum;cell_volume;num_atoms;space_group_symmetry;number_index;time;structure_type_number;structure_type;delta;\
#layer_anions;valence;bonded_anions;bonded_coordination;bonded_rank;pymatgen_coordination;voronoi_anions;voronoi_coordination;\
#voronoi_rank;metal_neigh_angle;metal_neigh_coordination;metal_neigh_rank;bonded_lattice;voronoi_lattice;metal_neigh_lattice")

#f.write("error_code;icsd_code;dimension;lower_index;upper_index;direction;chemical_structural_formula;\
#chemical_formula_sum;cell_volume;num_atoms;space_group_symmetry;number_index;time;structure_type_number;structure_type;delta;\
#layer_anions;valence;bonded_anions;bonded_coordination;bonded_rank;pymatgen_coordination;voronoi_anions;voronoi_coordination;\
#voronoi_rank;metal_neigh_angle;metal_neigh_coordination;metal_neigh_rank;bonded_lattice;voronoi_lattice;metal_neigh_lattice\n")
#f.close()

f = open("final_results.txt","r")
lines_results = f.readlines()[1:]
f.close()
dim_done = [int(line.split(";")[2]) for line in lines_results]
icsd_done = [line.split(";")[1] for line in lines_results]

icsd = []
dim = []
low_delta = []
up_delta = []
initial_string = []
for i in range(len(dim_full)):
    flag = 0
    for j in range(len(dim_done)):
        if dim_full[i] == dim_done[j] and icsd_full[i] == icsd_done[j]:
            flag = 1
            break
    if flag == 0:
        icsd.append(icsd_full[i])
        dim.append(dim_full[i])
        low_delta.append(low_delta_full[i])
        up_delta.append(up_delta_full[i])
        initial_string.append(initial_string_full[i])

del dim_full
del icsd_full
del low_delta_full
del up_delta_full
del initial_string_full
del lines
del lines_results
del dim_done
del icsd_done
print ("initialization done")


linear = np.array([[1,0,0],[-1,0,0]],dtype=np.float)
triangular = np.array([[1,0,0],[-np.cos(np.pi/3),np.sin(np.pi/3),0],[-np.cos(np.pi/3),-np.sin(np.pi/3),0]],dtype=np.float)
t_shape = np.array([[1,0,0],[0,1,0],[-1,0,0]],dtype=np.float)
square_planar = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],dtype=np.float)
tetrahedral = np.array([[1,0,0],[-np.cos(np.pi/3),np.sin(np.pi/3),0],[-np.cos(np.pi/3),-np.sin(np.pi/3),0],[0,0,np.sqrt(2)]],dtype=np.float) #h=sqrt(6)/3*a a=sqrt(2)
sq_pyramidal = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]],dtype=np.float)
tri_bipyramidal = np.array([[1,0,0],[-np.cos(np.pi/3),np.sin(np.pi/3),0],[-np.cos(np.pi/3),-np.sin(np.pi/3),0],[0,0,1],[0,0,-1]],dtype=np.float)
octahedral = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=np.float)
bcc = np.array([[1/2,1/2,1/2],[1/2,1/2,-1/2],[-1/2,1/2,-1/2],[-1/2,1/2,1/2],[1/2,-1/2,1/2],[1/2,-1/2,-1/2],[-1/2,-1/2,-1/2],[-1/2,-1/2,1/2]],dtype=np.float)
hexagonal = np.array([[1,0,0],[np.cos(np.pi/3),np.sin(np.pi/3),0],[-np.cos(np.pi/3),np.sin(np.pi/3),0],[-1,0,0],[-np.cos(np.pi/3),-np.sin(np.pi/3),0],[np.cos(np.pi/3),-np.sin(np.pi/3),0]],dtype=np.float)

points_perfect = {}
keys = ["2","3","4","5","6","8"]
for key in keys:
    points_perfect[key] = []
points_perfect["2"].append(linear)
points_perfect["3"].extend([triangular, t_shape])
points_perfect["4"].extend([square_planar, tetrahedral])
points_perfect["5"].extend([sq_pyramidal,tri_bipyramidal])
points_perfect["6"].extend([octahedral])
points_perfect["8"].extend([bcc])

lattices = {}
lattices["2"] = ["linear"]
lattices["3"] = ["triangular", "t_shape"]
lattices["4"] = ["square_planar", "tetrahedral"]
lattices["5"] = ["sq_pyramidal", "tri_bipyramidal"]
lattices["6"] = ["octahedral"]
lattices["8"] = ["bcc"]

metal_points_perfect = {}
keys = ["2","3","4","6"]
for key in keys:
    metal_points_perfect[key] = []
metal_points_perfect["2"] = linear
metal_points_perfect["3"] = triangular
metal_points_perfect["4"] = square_planar
metal_points_perfect["6"] = hexagonal

metal_lattices = {}
metal_lattices["2"] = "linear"
metal_lattices["3"] = "triangular"
metal_lattices["4"] = "square_planar"
metal_lattices["6"] = "hexagonal"

def calculate_info(initial_string, icsd, dim, delta):
    print (icsd,dim)
    initial_string = initial_string.rstrip()
    if dim == 0:
        for i in range(11):
            initial_string += ";-1"
    else:
        delta = up_delta
        g = open("cif_files/data_"+icsd+"-ICSD.cif","r")  #assuming all cif files are present in the cif_files folder with that particular format
        cif_file = g.read()
        g.close()
        cif_file = re.sub(r"[^\x00-\x7F]","",cif_file) #remove non-ascii characters
        cif_file = re.sub(r"(\(\d+)(\s|\n)",r"\1\)\2",cif_file) #close any open brackets (throws an error otherwise)
        structure = CifParser.from_string(cif_file, occupancy_tolerance=100).get_structures(primitive=False)[0]
        filtering = Filtering(structure)
        filtering.form_neighbor_list()
        G = filtering.define_bonds(delta)
        delta_check = filtering.check_for_delta(G,dim)
        layers_angles_stuff = filtering.calculate_layers_angles_and_stuff(G,dim)

        while not delta_check:
            delta = delta - 0.1
            if (delta >= low_delta):
                G = filtering.define_bonds(delta)
                delta_check = filtering.check_for_delta(G,dim)
            else:
                break

        if delta < low_delta:
            layers_angles_stuff = filtering.calculate_layers_angles_and_stuff(filtering.define_bonds(up_delta),dim)
            initial_string += ";"+str(up_delta)
        else:
            layers_angles_stuff = filtering.calculate_layers_angles_and_stuff(G,dim)
            initial_string += ";"+str(delta)

        for stuff in layers_angles_stuff[:-2]:
            initial_string += ";"+str(stuff)
    
        # bonded lattice
        bonded_lattice = {}
        for key, value in layers_angles_stuff[-2].items():
            points_distorted = value
            coord_no = str(layers_angles_stuff[3][key])
            if coord_no in points_perfect:
                bonded_lattice[key] = {}
                for i in range(len(points_perfect[coord_no])):
                    bonded_lattice[key][lattices[coord_no][i]] = str(np.round(symmetry_measure(points_distorted, points_perfect[coord_no][i])["symmetry_measure"],decimals=1))
            else:
               bonded_lattice[key] = -1
    
        # magnetic metal lattice
        metal_neigh_lattice = {}
        for key, value in layers_angles_stuff[-1].items():
            points_distorted = value
            coord_no = str(layers_angles_stuff[7][key])
            if coord_no in metal_points_perfect:
                metal_neigh_lattice[key] = {}
                metal_neigh_lattice[key][metal_lattices[coord_no]] = str(np.round(symmetry_measure(points_distorted, metal_points_perfect[coord_no])["symmetry_measure"],decimals=1))
            else:
               metal_neigh_lattice[key] = -1
 
        initial_string += ";"+str(bonded_lattice)+";"+str(metal_neigh_lattice)
    f = open("results_"+str(os.getpid())+".txt","a")
    print (initial_string)
    f.write(initial_string+"\n")
    f.close()

def helper_func(args):
    return calculate_info(*args)

pool = Pool()
pool.map(helper_func, itertools.izip(initial_string,icsd,low_delta,up_delta,dim))
