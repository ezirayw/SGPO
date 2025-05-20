### Installation ###
#    % wget http://eddylab.org/software/hmmer/hmmer.tar.gz
#    % tar zxf hmmer.tar.gz
#    % cd hmmer-3.4
#    % ./configure --prefix /disk1/jyang4   # replace /your/install/path with what you want, obv 
#    % make
#    % make check                                # optional: run automated tests
#    % make install                              # optional: install HMMER programs, man pages
#    % (cd easel; make install)                  # optional: install Easel tools

import subprocess
import os
from Bio import SeqIO

methods = ["jackhmmer"]
proteins = ["TrpB", "CreiLOV"] #"GB1", "TrpB", CreiLOV

for method in methods:
    for protein in proteins:

        # Define paths and filenames
        query_sequence_file = f"../{protein}/parent.fasta"  # Your query sequence in FASTA format
        uniprot_db = "/home/shared_data/uniref90.fasta"         # UniProt database in FASTA format, replace this with your own
        output_file = "jackhmmer_output" # Output file to store JACKHMMER results

        # Check if the database file exists
        if not os.path.exists(uniprot_db):
            raise FileNotFoundError(f"UniProt database file '{uniprot_db}' not found. Please download it from UniProt.")

        # Check if the query sequence file exists
        if not os.path.exists(query_sequence_file):
            raise FileNotFoundError(f"Query sequence file '{query_sequence_file}' not found. Please provide a valid query file.")

        # Command to run JACKHMMER
        jackhmmer_cmd = [
            f"/disk1/jyang4/bin/{method}", #jackhmmer
            #"--tblout", f"{method}/{protein}.tbl",  # Tabular output file
            "--domtblout", f"{method}/{protein}.domtbl",  # Domain table output
            #"-E", "1e-2",                      # E-value threshold
            "-N", "2",                  # Number of iterations, following Blalock et al. 2024
            "-A", f"{method}/{protein}_alignment.sto",      # Alignment output file
            "--cpu", "12",                      # Number of CPU cores to use
            query_sequence_file,               # Query sequence file
            uniprot_db                         # Target database
        ]

        subprocess.run(jackhmmer_cmd, check=True)

        #reformat alignment
        os.system(f"/disk1/jyang4/bin/esl-reformat fasta {method}/{protein}_alignment.sto > {method}/{protein}_alignment.fasta")
        os.system(f"/disk1/jyang4/bin/esl-reformat a2m {method}/{protein}_alignment.sto > {method}/{protein}_alignment.a2m")

        # Extract the full length fasta sequences
        # Input and output files
        input_database = "/home/shared_data/uniref90.fasta"   # UniProt FASTA database
        domtbl_file = f"{method}/{protein}.domtbl"       # JACKHMMER tabular output
        output_file = f"{method}/{protein}_seqs.fasta"

        # Parse domain table for sequence IDs
        seq_ids = set()
        with open(domtbl_file, "r") as domtbl:
            for line in domtbl:
                if not line.startswith("#"):  # Skip comments
                    seq_id = line.split()[0]  # Assuming the first column is sequence ID
                    seq_ids.add(seq_id)

        # Extract full-length sequences
        with open(output_file, "w") as out_fasta:
            for record in SeqIO.parse(input_database, "fasta"):
                if record.id in seq_ids:
                    SeqIO.write(record, out_fasta, "fasta")

        print(f"Full-length sequences saved to {output_file}")

        # subprocess.run(["/disk1/jyang4/bin/esl-reformat", "fasta", f"{protein}_alignment.sto", ">", f"{protein}_alignment.fasta"], check=True)
        # subprocess.run(["/disk1/jyang4/bin/esl-reformat", "a2m", f"{protein}_alignment.sto", ">", f"{protein}_alignment.a2m"], check=True)

        #/disk1/jyang4/bin/esl-reformat fasta jackhmmer/TrpB_alignment.sto > jackhmmer/TrpB_alignment.fasta