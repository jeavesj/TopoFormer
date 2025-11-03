import pandas as pd
import numpy as np
import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index_csv', required=True, help='Path to indexing csv with columns name, ligand, protein')
    parser.add_argument('-o', '--outdir', required=True, help='Path to desired output directory')
    parser.add_argument('--log_path', default='timing_out', default='timing_out.csv', help='Path to output timing log file')
    parser.add_argument('--script_path', default='code_pkg/main_potein_ligand_topo_embedding.py', help='Path to embedding script')
    args = parser.parse_args()
    
    timing_log = []
    
    df = pd.read_csv(args.index_csv)
    for idx, row in df.iterrows():
        t0 = time.time()
        subprocess.run(['python3', args.script_path, '--protein_file', row['protein'], '--ligand_file', row['ligand'], '--output_feature_folder', args.outdir])
        t1 = time.time() - t0
        
        timing_log.append({'name': row['name'], 'step': 'featurize', 'time_s': t1})
    
    pd.DataFrame(timing_log).to_csv(args.log_path, index=False)
    
if __name__ == '__main__':
    main()