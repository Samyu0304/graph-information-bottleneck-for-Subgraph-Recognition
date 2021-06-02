import rdkit
import rdkit.Chem as Chem
import json

path = './input/qed/train.txt'
Target_file = './input/qed/'

def preprocess(pairs):
    smiles = pairs[0]
    props = float(pairs[1])
    processed = {}
    processed['label'] = props
    processed['smiles'] = smiles
    count = 0

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
        count += 1

    edge = []

    for bond in mol.GetBonds():
        edge.append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])

    processed['edges'] = edge
    node = {}

    for atom in mol.GetAtoms():
        node[str(atom.GetIdx())] = atom.GetSymbol()

    processed['features'] = node

    return processed


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return mol

def save_json(data,path):
    b = json.dumps(data)
    f2 = open(path, 'w')
    f2.write(b)
    f2.close()


with open(path,'r') as f:
    data_1 = f.readlines()
data = [i.split(' ')[0:2] for i in data_1]
count = 0

for pairs in data:
    mol = get_mol(pairs[0])

    if mol is not None:
        processed = preprocess(pairs)
        save_path = Target_file + str(count) + '.json'
        save_json(processed,save_path)
        print(count)
        count += 1




