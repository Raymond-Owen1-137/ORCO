import json
import pandas as pd
import random
import sys

residue_map = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

def load_and_process_bmrb(bmrb_id):
    with open('bmrb_data_bank.json') as f:
        data = json.load(f)
    key = f'bmrb_id_{bmrb_id}'
    if key not in data:
        raise ValueError(f"No such BMRB ID: {bmrb_id}")
    spin_systems = data[key]['spin_systems']
    residues, ca_cb_data = [], []
    for s in spin_systems:
        res_type = s.get('residue_type')
        if not res_type or not res_type.isalpha():
            continue
        residues.append(res_type)
        if res_type == 'G':
            continue
        if res_type not in residue_map:
            continue
        ca = s['shifts'].get('CA')
        cb = s['shifts'].get('CB')
        if ca == 'NA' or cb == 'NA':
            continue
        try:
            ca = float(ca)
            cb = float(cb)
        except:
            continue
        ca_cb_data.append((ca, cb, residue_map[res_type]))
    return residues, ca_cb_data

def write_fasta(residues, output_file):
    sequence = ''.join(residues)
    with open(output_file, 'w') as f:
        f.write(f">BMRB_sequence\n{sequence}\n")

def write_dataset(ca_cb_data, output_file):
    random.shuffle(ca_cb_data)
    df = pd.DataFrame(ca_cb_data, columns=['CA', 'CB', 'label'])
    df.to_csv(output_file, index=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python load_json.py <bmrb_id>")
        return
    bmrb_id = sys.argv[1]
    residues, ca_cb_data = load_and_process_bmrb(bmrb_id)
    write_fasta(residues, f'bmrb_{bmrb_id}.fasta')
    write_dataset(ca_cb_data, f'bmrb_{bmrb_id}_labeled.csv')
    print(f"✅ FASTA and dataset generated for BMRB {bmrb_id}")

if __name__ == '__main__':
    main()