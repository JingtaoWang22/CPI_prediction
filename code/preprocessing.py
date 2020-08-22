from collections import defaultdict
import os
import pickle
import sys
import numpy as np


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
    bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol):
    # convert molecule to GNN input
    idxfunc=lambda x:x.GetIdx()

    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0

    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms,), dtype=np.int32) #atom feature ID
    fbonds = np.zeros((n_bonds,), dtype=np.int32) #bond feature ID
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    num_nbs_mat = np.zeros((n_atoms,max_nb), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())] 

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())] 
        try:
            atom_nb[a1,num_nbs[a1]] = a2
            atom_nb[a2,num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        
    for i in range(len(num_nbs)):
        num_nbs_mat[i,:num_nbs[i]] = 1

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


def Batch_Mol2Graph(mol_list):
    res = list(map(lambda x:Mol2Graph(x), mol_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return fatom_list, fbond_list, gatom_list, gbond_list, nb_list


def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    sequence = sequence.upper()
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])
    if ngram == 3:
        output = [-1]+output+[-1] # pad
    return np.array(output, np.int32)


def Batch_Protein2Sequence(sequence_list, ngram=3):
    res = list(map(lambda x:Protein2Sequence(x,ngram), sequence_list))
    return res


def get_mol_dict():
    if os.path.exists('../data/mol_dict'):
        with open('../data/mol_dict','rb') as f:
            u = pickle._Unpickler(f)
            u.encoding ='bytes'
            p = u.load()
            mol_dict = p#pickle.load(f, encoding='latin1')
    else:
        mol_dict = {}
        mols = Chem.SDMolSupplier('../data/Components-pub.sdf')
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open('../data/mol_dict', 'wb') as f:
            pickle.dump(mol_dict, f)
    #print('mol_dict',len(mol_dict))
    return mol_dict


def get_pairwise_label(pdbid, interaction_dict):
    if pdbid in interaction_dict:
        sdf_element = np.array([atom.GetSymbol().upper() for atom in mol.GetAtoms()])
        atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)
        atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)
        atom_interact = np.array(interaction_dict[pdbid]['atom_interact'], dtype=int)
        nonH_position = np.where(atom_element != ('H'))[0]
        assert sum(atom_element[nonH_position] != sdf_element) == 0
        
        atom_name_list = atom_name_list[nonH_position].tolist()
        pairwise_mat = np.zeros((len(nonH_position), len(interaction_dict[pdbid]['uniprot_seq'])), dtype=np.int32)
        for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
            atom_idx = atom_name_list.index(str(atom_name))
            assert atom_idx < len(nonH_position)
            
            seq_idx_list = []
            for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
                if bond_type == bond_type_seq:
                    seq_idx_list.append(seq_idx)
                    pairwise_mat[atom_idx, seq_idx] = 1
        if len(np.where(pairwise_mat != 0)[0]) != 0:
            pairwise_mask = True
            return True, pairwise_mat
    return False, np.zeros((1,1))


def get_fps(mol_list):
    fps = []
    for mol in mol_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)
        fps.append(fp)
    #print('fingerprint list',len(fps))
    return fps


def calculate_sims(fps1,fps2,simtype='tanimoto'):
    sim_mat = np.zeros((len(fps1),len(fps2))) #,dtype=np.float32)
    for i in range(len(fps1)):
        fp_i = fps1[i]
        if simtype == 'tanimoto':
            sims = DataStructs.BulkTanimotoSimilarity(fp_i,fps2)
        elif simtype == 'dice':
            sims = DataStructs.BulkDiceSimilarity(fp_i,fps2)
        sim_mat[i,:] = sims
    return sim_mat


def compound_clustering(ligand_list, mol_list):
    print('start compound clustering...')
    fps = get_fps(mol_list)
    sim_mat = calculate_sims(fps, fps)
    #np.save('../preprocessing/'+MEASURE+'_compound_sim_mat.npy', sim_mat)
    print('compound sim mat'+str(sim_mat.shape))
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        C_clusters = fcluster(C_link, thre, 'distance')
        len_list = []
        for i in range(1,max(C_clusters)+1):
            len_list.append(C_clusters.tolist().count(i))
        print('thre,'+str(thre)+ ',total num of compounds,'+str( len(ligand_list))+ ',num of clusters,'+str( max(C_clusters))+ ',max length,'+str( max(len_list)))
        C_cluster_dict = {ligand_list[i]:C_clusters[i] for i in range(len(ligand_list))}
        with open('../preprocessing/'+MEASURE+'_compound_cluster_dict_'+str(thre),'wb') as f:
            pickle.dump(C_cluster_dict, f, protocol=0)


def protein_clustering(protein_list, idx_list):
    print('start protein clustering...')
    protein_sim_mat = np.load('../data/pdbbind_protein_sim_mat.npy').astype(np.float32)
    sim_mat = protein_sim_mat[idx_list, :]
    sim_mat = sim_mat[:, idx_list]
    print('original protein sim_mat'+str(protein_sim_mat.shape)+ ',subset sim_mat,'+str( sim_mat.shape))
    #np.save('../preprocessing/'+MEASURE+'_protein_sim_mat.npy', sim_mat)
    P_dist = []
    for i in range(sim_mat.shape[0]):
        P_dist += (1-sim_mat[i,(i+1):]).tolist()
    P_dist = np.array(P_dist)
    P_link = single(P_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        P_clusters = fcluster(P_link, thre, 'distance')
        len_list = []
        for i in range(1,max(P_clusters)+1):
            len_list.append(P_clusters.tolist().count(i))
        print('thre,'+str( thre)+ ',total num of proteins,'+str(len(protein_list))+ ',num of clusters,'+str(max(P_clusters))+ ',max length,'+str( max(len_list)))
        P_cluster_dict = {protein_list[i]:P_clusters[i] for i in range(len(protein_list))}
        with open('../preprocessing/'+MEASURE+'_protein_cluster_dict_'+str(thre),'wb') as f:
            pickle.dump(P_cluster_dict, f, protocol=0)

def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)

if __name__ == "__main__":
    
    #MEASURE = 'KIKD' # 'IC50' or 'KIKD'
    MEASURE = 'KIKD'
    #print('Create dataset for measurement:'+str(MEASURE))
    print('Step 1/5, loading dict...')
    # load label dicts
    mol_dict = get_mol_dict()
    with open('../data/out7_final_pairwise_interaction_dict','rb') as f:
        interaction_dict = pickle.load(f,encoding='bytes')
    
    # initialize feature dicts
    wlnn_train_list = []
    #atom_dict = defaultdict(lambda: len(atom_dict))
    #bond_dict = defaultdict(lambda: len(bond_dict))
    #word_dict = defaultdict(lambda: len(word_dict))
    
    
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    
    
    for aa in aa_list:
        word_dict[aa]
    word_dict['X']
    
    # get labels
    i = 0
    pair_info_dict = {}
    f = open('../data/pdbbind_all_datafile.tsv')
    print('Step 2/5, generating labels...')
    for line in f.readlines():
        i += 1
        if i % 1000 == 0:
            print('processed sample num '+str(i))
        pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
        ###python3 convert to bytes
        cid=cid.encode()
        
        # filter interaction type and invalid molecules
        if MEASURE == 'All':
            pass
        elif MEASURE == 'KIKD':
            if measure not in ['Ki', 'Kd']:
                continue
        elif measure != MEASURE:
            continue
        if cid not in mol_dict:
            print('ligand not in mol_dict')
            continue
        mol = mol_dict[cid]
        
        # get labels
        value = float(label)
        pairwise_mask, pairwise_mat = get_pairwise_label(pdbid, interaction_dict)
        
        # handle the condition when multiple PDB entries have the same Uniprot ID and Inchi
        if inchi+' '+pid not in pair_info_dict:
            pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
        else:
            if pair_info_dict[inchi+' '+pid][6]:
                if pairwise_mask and pair_info_dict[inchi+' '+pid][3] < value:
                    pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
            else:
                if pair_info_dict[inchi+' '+pid][3] < value:
                    pair_info_dict[inchi+' '+pid] = [pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat]
    f.close()
    
    print('Step 3/5, generating inputs...')
    valid_value_list = []
    valid_cid_list = []
    valid_pid_list = []
    valid_pairwise_mask_list = []
    valid_pairwise_mat_list = []
    mol_inputs, seq_inputs = [], []
    
    compounds=[]
    interactions=[]
    adjacencies=[]
    proteins=[]
    radius=2
    ngram=3
    # get inputs
    for item in pair_info_dict:
        pdbid, cid, pid, value, mol, seq, pairwise_mask, pairwise_mat = pair_info_dict[item]
        fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
        if fa==[]:
            print('num of neighbor > 6, '+str( cid))
            continue
        
        #tsubaki
        mol = Chem.AddHs(mol)  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        words = split_sequence(seq, ngram)
        proteins.append(words)

        #interactions.append(np.array([float(interaction)]))
        # end of tsubaki
        
        
        mol_inputs.append([fingerprints,adjacency,fa, fb, anb, bnb, nbs_mat])
        seq_inputs.append(Protein2Sequence(seq,ngram=1))
        valid_value_list.append(value)
        valid_cid_list.append(cid)
        valid_pid_list.append(pid)
        valid_pairwise_mask_list.append(pairwise_mask)
        valid_pairwise_mat_list.append(pairwise_mat)
        wlnn_train_list.append(pdbid)
    
    print( 'Step 4/5, saving data...')
    
    
    dir_input = ('../cpi/dataset/monnKIKD/input/'
                 'radius3_ngram2——monn/')
    os.makedirs(dir_input, exist_ok=True)

    np.save(dir_input + 'compounds', compounds)
    np.save(dir_input + 'adjacencies', adjacencies)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'interactions', interactions)
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
    
    
    # get data pack
    fingerprint,adj,fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [np.array(fingerprint),np.array(adj),np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list), np.array(nbs_mat_list), np.array(seq_inputs), \
    np.array(valid_value_list), np.array(valid_cid_list), np.array(valid_pid_list), np.array(valid_pairwise_mask_list), np.array(valid_pairwise_mat_list)]
    
    # save data
    with open('../preprocessing/pdbbind_all_combined_input_'+MEASURE, 'wb') as f:
        pickle.dump(data_pack, f, protocol=0)
    
    np.save('../preprocessing/wlnn_train_list_'+MEASURE, wlnn_train_list)
    
    pickle_dump(atom_dict, '../preprocessing/pdbbind_all_atom_dict_'+MEASURE)
    pickle_dump(bond_dict, '../preprocessing/pdbbind_all_bond_dict_'+MEASURE)
    pickle_dump(word_dict, '../preprocessing/pdbbind_all_word_dict_'+MEASURE)
    
    print( 'Step 5/5, clustering...')
    compound_list = list(set(valid_cid_list))
    protein_list = list(set(valid_pid_list))
    # compound clustering
    mol_list = [mol_dict[ligand] for ligand in compound_list]
    compound_clustering(compound_list, mol_list)
    # protein clustering
    ori_protein_list = np.load('../data/pdbbind_protein_list.npy').tolist()
    idx_list = [ori_protein_list.index(pid.encode()) for pid in protein_list]
    protein_clustering(protein_list, idx_list)
    
    print ('='*50)
    print ('Finish generating dataset for measurement'+ str(MEASURE))
    print ('Number of valid samples'+ str(len(valid_value_list)))
    print ('Number of unique compounds'+ str(len(compound_list)))
    print ('Number of unique proteins'+ str(len(protein_list)))
