#!/usr/bin/env python

import numpy as np
import networkx as nx
from pymatgen import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
import re, time, argparse
from multiprocessing import Pool
from itertools import product

error_code = 0
outfile = "output-" + time.strftime("%Y%m%d%H%M%S") + ".txt"

class Filtering:
    def __init__(self,structure):
        self.table = self.alvarez_table()
        self.neigh_list = []  #neighbor_list for the atoms
        self.neigh_dist = []  #Van der Waals distance between atoms
        self.dist_matrix = [] #distance matrix assuming periodic boundary conditions
        self.structure = structure #pymatgen structure
        self.frac_coords = structure.frac_coords #keeping a separate copy so that I dont have to modify every time

    def alvarez_table(self): #parsing Van der Waals radius information from file
        with open('alvarez_table') as f:
            lines = f.read().splitlines()
        table = dict(item.split('\t') for item in lines)
        f.close()
        return table
     
    def check_bonds(self,species): #check if species is magnetic 3D metal or a non-metal
        nonmetals = [1,5,6,14,32,7,15,33,51,83,8,16,34,52,84,9,17,35,53,85] #hydrogen, metalloids, carbon, pnictogens, chalcogens, halogens
        magnetic = [23,24,25,26,27,28,29] # V to Cu
        for el in species.elements:
            try:
                if el.element.number in magnetic or el.element.number in nonmetals:
                    return True
            except AttributeError: #contains Deuterium or Tritium
                return True
        return False
    
    def check_3Dmetal(self,species): #check if species is a magnetic 3D metal or not
        magnetic = [23,24,25,26,27,28,29]
        for el in species.elements:
            try:
                if el.element.number in magnetic:
                    return True
            except AttributeError:
                pass
        return False
    
    def form_neighbor_list(self): 
        #delete non metals from structure and assign values to neigh_list, neigh_dist, dist_matrix, frac_coords
        global error_code
        remove_species = [] 
        for i in range(len(self.structure)): 
            if not self.check_bonds(self.structure.species_and_occu[i]):
                remove_species.append(i)
        for index in sorted(remove_species, reverse=True):
            del self.structure[index]

        self.structure.make_supercell([2,2,2])
        #update fractional coordinates
        frac_coords = self.structure.frac_coords
        frac_coords[frac_coords == 1.] = 0.
        self.frac_coords = frac_coords

        #obtain distance matrix
        self.dist_matrix = self.structure.distance_matrix

        st = time.time()
        
        #get neigh_list and neigh_dist
        for i in range(len(self.structure.sites)):
            if (time.time() - st > 1000):
                error_code = 2 #time out
                return
            self.neigh_list.append([])
            self.neigh_dist.append([])
            species = self.structure.species_and_occu[i]
            r1 = []
            for el in species.elements:
                try:
                    r1.append(float(self.table[str(el.element.number)]) * \
                        species.get_el_amt_dict()[str(el.element)])
                except AttributeError:
                    r1.append(float(self.table[str(1)]) * \
                        species.get_el_amt_dict()["D"])  #Deuterium. Support for Tritium not yet added.
            r1 = np.sum(r1)
            for j in range(i):
                if self.dist_matrix[i,j] < 7.0:
                    species = self.structure.species_and_occu[j]
                    r2 = []
                    for el in species.elements:
                        try:
                            r2.append(float(self.table[str(el.element.number)]) * \
                                species.get_el_amt_dict()[str(el.element)])
                        except AttributeError:
                            r2.append(float(self.table[str(1)]) * \
                                species.get_el_amt_dict()[str("D")])
                    r2 = np.sum(r2)
                    self.neigh_list[i].append(j)
                    self.neigh_dist[i].append(r1+r2)
    
    def define_bonds(self, delta): #assign bonds based on delta value
        G = nx.Graph()
        for i in range(len(self.neigh_list)):
            for j in range(len(self.neigh_list[i])):
                if self.dist_matrix[i,self.neigh_list[i][j]] < (self.neigh_dist[i][j] - delta):
                    G.add_edge(i,self.neigh_list[i][j]) 
        return G
   
    def determine_2Dness(self, delta): #obtain dimension and direction for the delta value
        G = self.define_bonds(delta)
        ranks = []
        directions = []

        for loc in np.arange(0,len(self.structure),8): #all atoms in a unit cell
            if self.check_3Dmetal(self.structure.species_and_occu[loc]): #check if it is a magnetic metal
                rank_nodes = np.array([],dtype=np.int16) 
                accepted_nodes = np.arange(8)+loc
                try:
                    for node in nx.node_connected_component(G,loc):
                        if (node in accepted_nodes):
                            rank_nodes = np.append(rank_nodes, node)
                except KeyError:
                    rank_nodes = [0]
                rank_matrix = np.array([np.round(self.frac_coords[node],decimals=3) \
                                                               for node in rank_nodes])
                rank_matrix = rank_matrix - rank_matrix[0]
                rank = np.linalg.matrix_rank(rank_matrix)
                ranks.append(rank)

                #find direction
                direction="0"
                if rank == 1:
                    direction = str(rank_matrix[1])
                elif rank == 2:
                    direction = str(np.cross(rank_matrix[1],rank_matrix[-1])*4)
                directions.append(direction)

        return [np.amax(ranks), directions[np.argmax(ranks)]]

    def get_nearest_cartesian_coords(self, node, loc):
        min_coords = []
        #get the image of the neighbors if they are closer than the one inside the cell                          
        cells = list(product([-1,0,1],repeat=3))

        min_dist = np.linalg.norm(self.structure.lattice.get_cartesian_coords(self.frac_coords[node]) \
            - self.structure.lattice.get_cartesian_coords(self.frac_coords[loc]))
        min_coords = self.structure.lattice.get_cartesian_coords(self.frac_coords[node])
        for cell in cells:
            frac_coord = cell + self.frac_coords[node]
            distance = np.linalg.norm(self.structure.lattice.get_cartesian_coords(frac_coord) \
                - self.structure.lattice.get_cartesian_coords(self.frac_coords[loc]))
            if distance < min_dist:
                min_dist = distance
                min_coords = self.structure.lattice.get_cartesian_coords(frac_coord)
        return min_coords

    def angle_calculation(self, connected_nodes, loc):
        angle = []
        nn_coords = []

        str_formula = str(self.structure.species_and_occu[loc].formula)
        nodes = np.array([node for node in connected_nodes \
            if self.structure.species_and_occu[node] == self.structure.species_and_occu[loc]])

        #get nearest magnetic_metal angle neighbors with a tolerance of 0.1 A.
        lattice_species = [self.structure.species_and_occu[node] for node in nodes]
        s = Structure(self.structure.lattice, lattice_species, self.frac_coords[nodes])
        s.make_supercell([3,3,3])
        new_frac_coords = np.round(s.frac_coords,decimals=3)
        new_frac_coords[new_frac_coords == 1.] = 0.
        distances = []
        for i in range(len(new_frac_coords)):
            distances.append(np.linalg.norm(s.lattice.get_cartesian_coords(new_frac_coords[i])-s.lattice.get_cartesian_coords([0.5,0.5,0.5])))

        new_loc = np.argmin(distances)
        loc_cart_coords = s.lattice.get_cartesian_coords(new_frac_coords[new_loc])

        #identify the two closest same species to the specie under consideration (new_loc)
        nodes = np.arange(len(s))
        distances = [s.distance_matrix[node,new_loc] for node in nodes]
        nodes = nodes[np.argsort(distances)]
        distances = np.sort(distances)

        #identify other potential second closest atoms. Useful if there are multiple such atoms.
        min_dist_node1 = nodes[1]
        try: #for 1D compounds. Not applicable now.
            min_dist_node2 = nodes[2]
        except IndexError:
            angle = '180.0'
            nn_coords = [s.lattice.get_cartesian_coords(new_frac_coords[nodes[1]])-loc_cart_coords]
            return angle, nn_coords

        nnn_list = [nodes[i] for i in range(2,len(nodes)) if distances[i] < distances[2]+0.1]
        #get the image of the neighbors if they are closer than the one inside the cell 

        nnn_coords = []
        for node in nnn_list:
            nnn_coords.append(s.lattice.get_cartesian_coords(new_frac_coords[node]))

        #get the angles between the atoms and take the non-zero minimum value.
        angles = []
        v1 = s.lattice.get_cartesian_coords(new_frac_coords[min_dist_node1]) - loc_cart_coords
        for coords in nnn_coords:
            v2 = coords - loc_cart_coords
            cos_theta = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            if np.round(cos_theta,decimals=1) == -1.0:
                angles.append(180.0)
            else:
                angles.append(np.arccos(cos_theta)*180/np.pi)
        try:
            angle.append(min(i for i in angles if i > 0.1))
        except ValueError:
            angle.append(0.0)
        angle = str(np.round(min(angle),decimals=2))

 
        nn_list = [nodes[i] for i in range(1,len(nodes)) if distances[i] <= distances[1]+0.1]
        for node in nn_list:
            nn_coords.append(s.lattice.get_cartesian_coords(new_frac_coords[node]))
        nn_coords = [coord-loc_cart_coords for coord in nn_coords]
        return angle, nn_coords

    def layer(self, connected_nodes):
        return list(set([str(self.structure.species_and_occu[node].formula) \
            for node in connected_nodes if not \
            self.check_3Dmetal(self.structure.species_and_occu[node])]))

    def valence(self, loc):
        try:
            return BVAnalyzer().get_valences(self.structure)[loc]
        except Exception:
            return -1

    def pymatgen_coordination(self, vcf, loc):
        try:
            return str(np.round(vcf.get_coordination_number(loc),decimals=2))
        except Exception:
            return -1

    def voronoi_anions(self, vor_neigh):
        try:
            return [str(n.species_and_occu.formula) for n in vor_neigh]
        except Exception:
            return -1

    def check_for_delta(self, G, dim):
        layer_anions = {}
        str_formulas = []

        for loc in np.arange(0,len(self.structure),8):
            specie = self.structure.species_and_occu[loc]
            if self.check_3Dmetal(specie): #determine is the species is magnetic metal or not
                rank_nodes = []
                accepted_nodes = np.arange(8)+loc

                try:
                    connected_nodes = nx.node_connected_component(G,loc)
                    for node in connected_nodes:
                        if (node in accepted_nodes):
                            rank_nodes.append(node)
                except KeyError:
                    continue
    
                rank_matrix = np.array([np.round(self.frac_coords[node],decimals=3) for node in rank_nodes])
                rank_matrix = rank_matrix - rank_matrix[0]
                rank = np.linalg.matrix_rank(rank_matrix)
                str_formula = str(self.structure.species_and_occu[loc].formula) 
                if rank == dim and str_formula not in str_formulas: #only proceed if the magnetic metal is of the same dim
                    str_formulas.append(str_formula)

                    #get all the non-metals in the rod or layer and valence info
                    layer_anions[str_formula] = self.layer(connected_nodes)
        for key, anions in layer_anions.items():
            if len(anions) != 0:
                return True
        return False

    def calculate_layers_angles_and_stuff(self, G, dim):
        #obtain information based on the bonding network. Dont mess with this function unless you really have to.

        layer_anions = {} #all the anions present in the layer
        valence = {} #valency of the magnetic metal atoms in the layer
        bonded_anions = {} #anions bonded directly to the magnetic metal atom
        bonded_coordination = {} #number of anions bonded
        bonded_rank = {}
        pymatgen_coordination = {} #coordination number {fraction} obtained from pymatgen
        #voronoi_anions = {} #anions bonded to the magnetic metal atom according to voronoi polyhedra
        #voronoi_coordination = {} #number of anions from above
        #voronoi_rank = {}
        bonded_vectors = {} #vectors of the neighbors using bonds
        #voronoi_vectors = {} #vectors of the neighbors using voronoi construction
        metal_neigh_angle = {} #angle formed by magnetic metal atoms with each other in the same network
        metal_neigh_vectors = {}
        metal_neigh_coordination = {}
        metal_neigh_rank = {}
        #str_formulas = []

        for loc in np.arange(0,len(self.structure),8):
            specie = self.structure.species_and_occu[loc]
            if self.check_3Dmetal(specie): #determine is the species is magnetic metal or not
                rank_nodes = []
                accepted_nodes = np.arange(8)+loc

                try:
                    connected_nodes = nx.node_connected_component(G,loc)
                    for node in connected_nodes:
                        if (node in accepted_nodes):
                            rank_nodes.append(node)
                except KeyError:
                    continue
    
                rank_matrix = np.array([np.round(self.frac_coords[node],decimals=3) for node in rank_nodes])
                rank_matrix = rank_matrix - rank_matrix[0]
                rank = np.linalg.matrix_rank(rank_matrix)
                str_formula = str(loc/8)+str(self.structure.species_and_occu[loc].formula) 
                if rank == dim: # and str_formula not in str_formulas: #only proceed if the magnetic metal is of the same dim
                    #str_formulas.append(str_formula)

                    #get all the non-metals in the rod or layer and valence info
                    layer_anions[str_formula] = self.layer(connected_nodes)
                    valence[str_formula] = self.valence(loc)

                    # get metal angles, metal neighbor coords and rank
                    formula = str(self.structure.species_and_occu[loc].formula)
                    if formula not in metal_neigh_angle:
                        metal_neigh_angle[formula], metal_neigh_vectors[formula] = self.angle_calculation(connected_nodes, loc)
                        metal_neigh_coordination[formula] = len(metal_neigh_vectors[formula])
                        metal_neigh_rank[formula] = np.linalg.matrix_rank(metal_neigh_vectors[formula])

                    #bonded anions, coordination, coords and rank
                    neigh = np.array([n for n in G[loc] if not self.check_3Dmetal(self.structure.species_and_occu[n])])
                    coords = [self.get_nearest_cartesian_coords(n,loc) for n in neigh]

                    bonded_anions[str_formula] = [str(self.structure.species_and_occu[n].formula) for n in neigh]
                    bonded_coordination[str_formula] = len(bonded_anions[str_formula])
                    bonded_vectors[str_formula] = [coords[n] - self.structure.lattice.get_cartesian_coords(self.frac_coords[loc]) for n in range(len(coords))]
                    bonded_rank[str_formula] = np.linalg.matrix_rank(bonded_vectors[str_formula])

                    vcf = VoronoiCoordFinder(self.structure)

                    # paymetgen coordination
                    pymatgen_coordination[str_formula] = self.pymatgen_coordination(vcf, loc)

                    # voronoi anions, coordination, coords and rank
                    """try:
                        vor_neigh = [n for n in vcf.get_coordinated_sites(loc) \
                            if self.check_bonds(n.species_and_occu) and not self.check_3Dmetal(n.species_and_occu)]
                        coords = [n.coords for n in vor_neigh]
                        voronoi_anions[str_formula] = self.voronoi_anions(vor_neigh)
                        voronoi_coordination[str_formula] = len(voronoi_anions[str_formula]) 
                    except Exception:
                        coords = []
                        voronoi_anions[str_formula] = -1
                        voronoi_coordination[str_formula] = -1

                    voronoi_vectors[str_formula] = [coords[n] - self.structure.cart_coords[loc] for n in range(len(coords))]
                    voronoi_rank[str_formula] = np.linalg.matrix_rank(voronoi_vectors[str_formula])"""

        return layer_anions, valence, bonded_anions, bonded_coordination, bonded_rank,  pymatgen_coordination, \
                    metal_neigh_angle, metal_neigh_coordination, metal_neigh_rank, bonded_vectors, metal_neigh_vectors
    
def get_delta_range(cif_files): #use this function in case of parallel processing
    pool = Pool() 
    pool.map(do_stuff, cif_files)

def get_structure_details(cif_file): #details from the cif file
    lines = cif_file.split("\n")
    pattern1 = re.compile("^_database_code_ICSD")
    pattern2 = re.compile("^_chemical_formula_structural")
    pattern3 = re.compile("^_chemical_formula_sum")
    pattern4 = re.compile("^_cell_volume")
    pattern5 = re.compile("_symmetry_space_group_name_H-M")
    pattern6 = re.compile("_symmetry_Int_Tables_number")
    details = ["","","","","",""] 
    for i in range(len(lines)):
        if pattern1.match(lines[i]):
            details[0] = lines[i].split()[-1].strip()
        elif pattern2.match(lines[i]):
            details[1] = lines[i].strip("_chemical_formula_structural").strip()
            if len(details[1]) < 3:
                j=2
                while lines[i+j].strip() != ";":
                    details[1] = details[1]+lines[i+j]
                    j += 1
        elif pattern3.match(lines[i]):
            details[2] = lines[i].strip("_chemical_formula_sum").strip()
            if len(details[2]) < 3:
                j=2
                while lines[i+j].strip() != ";":
                    details[2] = details[2]+lines[i+j]
                    j += 1
        elif pattern4.match(lines[i]):
            details[3] = lines[i].split()[-1].strip()
        elif pattern5.match(lines[i]):
            details[4] = lines[i].strip("_symmetry_space_group_name_H-M").strip()
        elif pattern6.match(lines[i]):
            details[5] = lines[i].strip("_symmetry_Int_Tables_number").strip()
    return details
   
def parse_file(input_file, start_cif, end_cif): # return strings of split cif files
    if end_cif == "None":
        end_cif = None
    else:
        end_cif = int(end_cif)
    f = open(input_file,"r")
    content = f.read()
    cif_files = re.compile('\ndata_|^data_').split(content)[1:]
    cif_files = ["data_"+cif_file for cif_file in cif_files]
    cif_files = cif_files[int(start_cif):end_cif]
    f.close()
    return cif_files

def do_stuff(cif_file): #this is the function that does calculations for each cif file
    global error_code

    #initialization
    error_code = 0
    dim = 0
    num_atoms = 0
    results = []
    f = open(outfile,"a")
    start = time.time()

    try:
        cif_file = re.sub(r"[^\x00-\x7F]","",cif_file) #remove non-ascii characters
        cif_file = re.sub(r"(\(\d+)(\s|\n)",r"\1\)\2",cif_file) #close any open brackets (throws an error otherwise)
        structure = CifParser.from_string(cif_file, occupancy_tolerance=100).get_structures(primitive=False)[0]
        filtering = Filtering(structure)
        num_atoms = len(structure)
        filtering.form_neighbor_list()
        if len(filtering.neigh_list) != 0 and error_code != 2:
            for i in range(30):
                delta = i*0.1
                dim_dir = filtering.determine_2Dness(delta)
                results.append(dim_dir)
                if dim_dir[0] == 0:
                    break

    except (ValueError,AssertionError,ZeroDivisionError):
        error_code = 1
    except AttributeError:
        error_code = 3 #for Deuterium (just in case)
    except MemoryError:
        error_code = 4
    except Exception:
        error_code = 5

    details = get_structure_details(cif_file)
    results = np.array(results)
    if error_code != 0 or ('1' not in results[:,0] and '2' not in results[:,0]):
        low_index = -1
        up_index = -1
        direction = "0"
        end = time.time()
        r = str(error_code)+";"+details[0].rstrip()+";0;"+str(low_index)+";" \
                   +str(up_index)+";"+direction+";"+details[1].rstrip()+";" \
                   +details[2].rstrip()+";"+details[3].rstrip()+";"+str(num_atoms)+";" \
                   +details[4].rstrip()+";"+details[5].rstrip()+layers_angles \
                   +str(np.round(end-start,decimals=2))
        print (r)
        f.write(r+"\n")
    else:
        for i in range(2,0,-1):
            if str(i) in results[:,0]:
                indices = np.where(results[:,0]==str(i))[0]
                low_index = 0.1*indices[0]
                up_index = 0.1*indices[-1]
                direction = results[indices[0]][1]
                end = time.time()
                icsd_code = details[0].rstrip()
                r = str(error_code)+";"+icsd_code+";"+str(i)+";"+str(low_index)+";" \
                       +str(up_index)+";"+direction+";"+details[1].rstrip()+";" \
                       +details[2].rstrip()+";"+details[3].rstrip()+";"+str(num_atoms)+";" \
                       +details[4].rstrip()+";"+details[5].rstrip() \
                       +str(np.round(end-start,decimals=2))
                print (r)
                f.write(r+"\n")

    f.close()

def main():
    #parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file',help='cif file containing multiple cifs downloaded from FindIt')
    parser.add_argument('cpu',help='serial or parallel')
    parser.add_argument('start_cif',help='cif file you want to start running (this is included), \
                                                                  type 0 if you want to start from first')
    parser.add_argument('end_cif',help='cif file you want to run last (this is excluded), \
                                                                  type "None" if you want to run till end')

    args = parser.parse_args()
    input_file = args.input_file
    cpu = args.cpu
    start_cif = args.start_cif
    end_cif = args.end_cif

    #open output file
    f = open(outfile,"w")
    f.write("error_code;icsd_code;dimension;lower_index;upper_index;direction;chemical_structural_formula;" \
             +"chemical_formula_sum;cell_volume;num_atoms;space_group_symmetry;number_index;layer;angle;" \
             +"valency;bonded_anions;bonded_coordination;pymat_coordination;voronoi_anions;voronoi_coordination;time\n")
    f.close()    
    """g = open("coordinate_file.txt","w")
    g.write("type;icsd_code;dim;magnetic_metal;x;y;z\n")
    g.close()"""

    #parse the cif file and run calculations
    cif_files = parse_file(input_file,start_cif,end_cif)
    if cpu == "serial":
        for cif_file in cif_files:
            do_stuff(cif_file)
    else:            
        get_delta_range(cif_files)

if __name__ == "__main__":
    main() 
