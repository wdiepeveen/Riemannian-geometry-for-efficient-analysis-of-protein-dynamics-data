import numpy as np

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import DisorderedAtom



def load_full_backbone(stu_fn, all_structs=True, quiet=False):
   
    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=True)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])
    
    structure = parser.get_structure("structure", stu_fn)

    if not quiet and len(structure) > 1:
        print(f"WARNING: {len(structure)} structures found in model file: {stu_fn}")

    if not all_structs:
        structure = [structure[0]]

    coords = []
        
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for chain in model:
            chain_len = 0
            for r in chain.get_residues():
                        if "CA" in r:
                            a = r["CA"]
                            chain_len += 1
                            if isinstance(a, DisorderedAtom):
                                coords.append(
                                    a.disordered_get_list()[0].get_vector().get_array()
                                    )
                            else:
                                coords.append(a.get_vector().get_array())
            print("Chain contains", chain_len, "CAs")

    return np.array(coords)



