#muscle -align TrpB_selected.fasta -output TrpB_aligned_selected.fasta

#conda install bioconda::mafft
import os
from Bio import SeqIO
import pandas as pd
import hydra

def extract_combos(sequences, data_config):
    residues = [r-1 for r in data_config.residues]
    protein = data_config.name
    full_seq = data_config.full_seq

    #save to fasta
    #if sequences is a string of the fasta file name
    if type(sequences) == str:
        os.system(f"mafft --add {sequences} --keeplength ../{protein}.fasta > temp_aligned.fasta")
    else:
        with open(f"temp.fasta", "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">{i}\n")
                f.write(f"{seq}\n")

        os.system(f"mafft --add temp.fasta --keeplength ../{protein}.fasta > temp_aligned.fasta")
    aligned = list(SeqIO.parse("temp_aligned.fasta", "fasta"))

    #extract combos
    combos = []
    for i, seq in enumerate(aligned):
        combo = "".join([seq.seq[r] for r in residues])
        combos.append(combo)
    #print number of combos containing "-"
    print("Number of generations with gaps: " + str(sum(["-" in combo for combo in combos])))
    
    #fill in unaligned positions "-" with the residue in full_seq
    for i, combo in enumerate(combos):
        for j, r in enumerate(combo):
            if r == "-":
                pos = residues[j]
                combos[i] = combos[i][:j] + full_seq[pos] + combos[i][j+1:]
    return combos

@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="TrpB")
def main(data_config):

    df = pd.read_csv("../TrpB_MSA_full.csv")
    sequences = df["sequence"].values[:100]
    combos = extract_combos(sequences, data_config)

if __name__ == "__main__":
    main()


