############################ Libraries ############################
# Basics
import copy, csv, glob, gzip, math, os, pickle, re, requests, signal, statistics, sys, warnings
from datetime import datetime
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pandas as pd
from scipy import sparse
from time import sleep
from time import time
from tqdm import tqdm

# BioPython
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction
import RNA

# Primer annealing
import primer3 as pr

# Multiple jobs
from multiprocessing import Pool, cpu_count
import psutil

############################
# General
############################

flatten = lambda *args: (
    result for mid in args
    for result in (
        flatten(*mid)
        if isinstance(mid, (tuple, list))
        else (mid,)))

def f_print_10(i, message1, message2):
    '''
    Message for the iteration
    1,2,3,4,5,,,9, 10,20,30,,,,90, 100,200,300,,,900,...
    '''
    if i < 10 or str(i)[1:].count('0') == len(str(i))-1:
        now = datetime.now()
        print(datetime.now(), message1, i, message2)

def f_estimate_time(func, i_total):
    '''
    Estimate the time to complete calculating `func`
    '''
    i = 1
    while True:
        time_start = datetime.now()
        temp = func(i)
        time_elapsed = datetime.now() - time_start
        print("Estimating", i, time_elapsed)
        if time_elapsed.seconds > 1:
            break
        i += 1
    estimated_time = datetime.now() + time_elapsed * i_total / (10**i)
    print("Estimated completion time:", estimated_time.strftime("%Y-%m-%d %H:%M:%S"))

def f_download_file(dir_database, url, filename):
    '''
    Download a file from the web, showing progress
    '''
    file = os.path.join(dir_database, "source", filename)
    print("Download from:", url)
    # Get file size
    response = requests.head(url, allow_redirects=True)
    size = response.headers.get('content-length', -1)
    print('Size: '+str(size)+' bytes')
    print("Download to:", file)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024*16):
                if chunk:
                    f.write(chunk)
                    print('.', end='', flush=True)

def f_save_csv(l, filename):
    '''
    Save list as csv
    '''
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(l)
    print(datetime.now(), "Saved", filename)

def f_show_stat(pd_NGG):
    '''
    Show basic statistics
    '''
    print(pd_NGG.shape[0], 'entries are found')
    print('Result contained targets for', 
        pd_NGG['gene'].nunique(dropna=False), 'genes')
    pd_NGG_count = pd_NGG['gene'].value_counts()
    x_NGG_count_1 = sum(pd_NGG_count == 1)
    x_NGG_count_2 = sum(pd_NGG_count == 2)
    x_NGG_count_3 = sum(pd_NGG_count >= 3)
    x_NGG_count_6 = sum(pd_NGG_count >= 6)
    x_NGG_count_3 -= x_NGG_count_6
    l_temp = [x_NGG_count_1 + x_NGG_count_2, x_NGG_count_3, x_NGG_count_6]
    print('Number of genes with [1-2, 3-5, 6-] targets:', l_temp)

def f_custom_sort(filename):
    '''
    Custom sorting function to sort based on chromosome number
    '''
    # Extract the number after "chr_"
    num = int(filename.split("chr_")[1].split(".")[0])
    return num

def f_concatenate_csv_in_dir(dir_name):
    # Get all files with ".csv" extension
    list_csv_file = [file for file in os.listdir(dir_name) if file.endswith(".csv")]
    # Sort files using custom_sort, in chromosome order
    list_csv_file = sorted(list_csv_file, key=f_custom_sort)    
    dfs = []  # To store dataframes
    # Load each .csv file into a dataframe and store in the list dfs
    for csv_file in list_csv_file:
        print("Load", csv_file)
        file_path = os.path.join(dir_name, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df

def f_cut_export(data, mode, dir_save, prefix, n_cut_size=1000000):
    '''
    Cut the list of [20nt]NGG sequences into chunks and export
    `data` list of sequences or the target candidate data
    `mode`: 
        "qPCR": save Pandas.DataFrame containing `start`, `chr`, `strand`
        "off_target": save list of pseudo-binary string (0,1) of [20nt] part only
    `dir_save`, `prefix`: Output file specification
    `n_cut_size`: default=1M, a practical size for typical nodes as of 2023
    '''
    if mode == "qPCR":
        # Save Pandas.DataFrame containing `start`, `chr`, `strand`
        print(datetime.now(), "Cut into chunks & export")
        print("Output to:", dir_save)
        pd_export = data[["start", "chr", "strand"]]
        for chunk_start in range(0, len(pd_export), n_cut_size):
            chunk_end = min((chunk_start + n_cut_size), len(pd_export))
            chunk = pd_export[chunk_start: chunk_end]
            filename = prefix + "_" + "0" * (15-len(str(chunk_start))) + str(chunk_start) + ".pickle"
            filename = os.path.join(dir_save, filename)
            with open(filename, mode='wb') as f:
                pickle.dump(chunk, f)
    
    if mode == "off_target":
        # Save list of pseudo-binary string (0,1) of [20nt] part only
        # The sequence should already have been made to uppercase & ATGC-only
        print(datetime.now(), "Cut into chunks & export as pseudo-binary string")
        print("Output to:", dir_save)
        dict_translate = {'A': '1000', 'T': '0100', 'G': '0010', 'C': '0001'}
        list_result = [l[2][:20].translate(str.maketrans(dict_translate)) for l in data]
        for chunk_start in range(0, len(data), n_cut_size):
            chunk_end = min((chunk_start + n_cut_size), len(data))
            chunk = list_result[chunk_start: chunk_end]
            filename = prefix + "_" + "0" * (15-len(str(chunk_start))) + str(chunk_start) + ".pickle"
            filename = os.path.join(dir_save, filename)
            with open(filename, mode='wb') as f:
                pickle.dump(chunk, f)
    print("Done.")

def get_n_core():
    '''
    Get the number of available CPU cores
    '''
    n_core_all = os.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    # `50 - cpu_percent` instead of `100 - cpu_percent`:
    # this is for sparing some CPU power for other users
    n_core_available = int(n_core_all * (50 - cpu_percent) / 100)
    
    print("All cores:", n_core_all)
    print("Idle cores:", n_core_available)

    return n_core_available

############################
# Initial
############################

def f_make_dir(directory):
    '''
    Make a directory only if not already existing
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def f_setup_directory(dir_database):
    '''
    Initial setup - make folders
    '''
    list_dir = [
    dir_database,
    os.path.join(dir_database, "result"),
    os.path.join(dir_database, "result", "chromosome"),
    os.path.join(dir_database, "result", "NGG_target_candidate"),
    os.path.join(dir_database, "result", "NGG_target_candidate", "all"),
    os.path.join(dir_database, "result", "NGG_target_candidate", "all", "common"),
    os.path.join(dir_database, "result", "NGG_target_candidate", "all", "union"),
    os.path.join(dir_database, "result", "NGG_target_candidate", "for_offtarget_search"),
    os.path.join(dir_database, "result", "NGG_target_candidate", "for_qPCR_search"),
    os.path.join(dir_database, "result", "NGG_genomic"),
    os.path.join(dir_database, "result", "NGG_genomic", "csv"),
    os.path.join(dir_database, "result", "NGG_genomic", "pickle"),
    os.path.join(dir_database, "result", "off_target"),
    os.path.join(dir_database, "result", "qPCR"),
    os.path.join(dir_database, "result", "final"),
    os.path.join(dir_database, "result", "final", "final_sort_name"),
    os.path.join(dir_database, "result", "final", "final_sort_pos"),
    os.path.join(dir_database, "source"),
    ]
    [f_make_dir(directory) for directory in list_dir]    
    os.chdir(dir_database)

def f_initial_directory_setup(dir_database, download_list):
    '''
    Initial setup
    '''
    print('##############################################')
    print(datetime.now(), "Download source data")
    for url, filename in download_list.items():
        f_download_file(dir_database, url, filename)

############################
# Parse the sequence/annotation files 
############################

def f_parse_sequence(dir_database):
    '''
    Extract chromosomal genomic sequence per chromosome
    '''
    print('##############################################')
    print(datetime.now(), "Extract genome sequence")
    print("Loading genome data...")
    os.chdir(os.path.join(dir_database, "source"))
    with gzip.open("genome.gz", "rt") as handle:
        gb_record = list(SeqIO.parse(handle, "fasta"))
    print(datetime.now(), len(gb_record), 
        "records were found. Picking out full chromosomes only...")
    os.chdir(os.path.join(dir_database, "result", "chromosome"))
    i_seq = 1
    list_chr_id = []
    list_chr_desc = []
    for sequence in gb_record:
        # Remove unlocalized contigs or scaffolds
        # and save the full chromosomes only
        seq_id = sequence.id
        seq_desc = sequence.description
        if ((seq_id.find('NC_') > -1) or (seq_id.find('NZ_') > -1)) \
            and seq_desc.find('contig') == -1 \
            and seq_desc.find('mitochon') == -1 \
            and seq_desc.find('patch') == -1 \
            and seq_desc.find('scaffold') == -1 \
            and seq_desc.find('unlocalized') == -1 \
            and seq_desc.find('NW_') == -1 \
            and seq_desc.find('unplaced') == -1:
            print(seq_id, seq_desc)
            # Append to chromosome list table
            list_chr_id.append([i_seq, seq_id])
            list_chr_desc.append([i_seq, seq_desc])
            # Convert the sequences
            # All upper, ATGCN only
            str_temp = sequence.seq.upper()
            str_temp = ''.join(base if base in "ATGCN" else 'N' for base in str_temp)
            sequence.seq = Seq(str_temp)
            # Save as fasta
            filename_chr = "chr_" + str(i_seq) + ".fasta"
            SeqIO.write([sequence], filename_chr, "fasta")            
            print(datetime.now(), "Saved", filename_chr)
            i_seq += 1
    print(i_seq - 1, "Chromosome(s) found")
    print()
    # Save the chromosome list table
    f_save_csv(list_chr_id, 'list_chr_id.csv')
    f_save_csv(list_chr_desc, 'list_chr_desc.csv')
    dict_chr = {l[1]:int(l[0]) for l in list_chr_id}
    # Chromosome ID-number correspondence table
    # This is exceptionally saved in a .pickle format to conserve the variable types
    # The information inside is equivalent to `list_chr_id.csv`
    filename_list_chr = "dict_chr.pickle"
    with open(filename_list_chr, mode='wb') as f:
        pickle.dump(dict_chr, f)
    print(datetime.now(), "Saved to", filename_list_chr)

def f_extract_mRNA_annotation(gb_annotation, dict_chr):
    '''
    Extract gene-coding mRNA only
    '''
    pd_annotation = pd.Series([s.description for s in gb_annotation])
    pd_annotation = pd_annotation[pd_annotation.str.contains('mrna')]
    print('mRNA:', pd_annotation.shape[0])
    # Probable entries only (Omit XM, keep NM)
    pd_annotation = pd_annotation[pd_annotation.str.contains('_mrna_NM_')]
    print('Omit XM:', pd_annotation.shape[0])
    pd_annotation = pd_annotation[pd_annotation.str.contains('lcl\|NC_')]
    print('On a full chromosome:', pd_annotation.shape[0])
    pd_annotation = pd_annotation[pd_annotation.str.contains('gene=')]
    print('gene-coding: ', pd_annotation.shape[0])
    pd_annotation = pd_annotation[~pd_annotation.str.contains('\[partial=5\'\]')]
    print('Not 5prime-partial: ', pd_annotation.shape[0])
    pd_annotation = pd_annotation[~pd_annotation.str.contains('\[partial=3\'\]')]
    print('Not 3prime-partial: ', pd_annotation.shape[0])
    pd_annotation = pd_annotation[~pd_annotation.str.contains('exception=')]
    print('No special annotation: ', pd_annotation.shape[0])
    
    # Omit unnecessary tags
    # Ref: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/file-formats/annotation-files/about-ncbi-gff3/
    # Some fields contain [ and ], so the find-replace function must catch the outer []
    pd_annotation = (pd_annotation.str.replace(' \[db_xref.+?\] \[', ' [', regex=True)
                     .str.replace(' \[product.+?\] \[', ' [', regex=True)
                     .str.replace(' \[transcript_id.+?\] \[', ' [', regex=True)
                     .str.replace(' \[gbkey.+?\]', '', regex=True)
                     .str.replace(' \[locus_tag.+?\] \[', ' [', regex=True)
                     .str.replace(' \[partial.+?\] \[', ' [', regex=True)
                    )
    
    # Check that the remaining part contains only the necessary information and not anything else
    pd_check = (pd_annotation.str.replace(' \[gene=.+?\] \[', ' [', regex=True)
                             .str.replace(' \[location=.+?\]', '', regex=True)
                             # Should remain lastly: something like "lcl|NC_000087.7_mrna_NM_001160136.1_121899"
                             .str.replace('lcl\|NC_[0-9]+\.[0-9]*_mrna_NM_[0-9]+\.[0-9]*_[0-9]*', '', regex=True)
                            )
    list_check = pd_check.values.tolist()
    list_check = [x for x in list_check if len(x)!=0]
    print()
    if len(list_check) == 0:
        print("All entries in the data are formatted as expected")
    else:
        print("The following part of the data is formatted differently. Please check and fix the code.")
        print(list_check)
        exit
    print()
    
    # Extract to table
    pd_annotation = pd_annotation.str.extract('lcl\|(?P<chr>.+)'\
        '_mrna_(?P<ID>.+) '\
        '\[gene=(?P<gene>.+)'\
        '\] \[location=(?P<mRNA>.+)\]', expand=False)
    
    print("Make sure again that the source is full chromosome in `dict_chr`")
    l_extract = [
        item in list(dict_chr.keys())
        for item in pd_annotation['chr'].values]
    pd_annotation = pd_annotation[l_extract]
    print('Passed all filters:', pd_annotation.shape[0])
    pd_annotation = pd_annotation.reset_index(drop=True)
    pd_annotation["chr_file"] = pd_annotation["chr"].apply(lambda x: dict_chr[x])
    return pd_annotation

def f_parse_mrna_pos(pd_annotation):
    '''
    Convert position string into np.array
    '''
    np_annotation_location = np.array(pd_annotation['mRNA'])
    np_annotation_location = [x[x.find("join") + 5:] if x.find("join") > -1 else x for x in np_annotation_location]
    np_annotation_location = [x[x.find("complement") + 11:] if x.find("complement") > -1 else x for x in np_annotation_location]
    np_annotation_location = [x[:x.find(")")] if x.find(")") > -1 else x for x in np_annotation_location]
    np_annotation_location = [x.replace('>','') for x in np_annotation_location]
    np_annotation_location = [x.replace('<','') for x in np_annotation_location]
    np_annotation_location = [x.split(',') for x in np_annotation_location]
    np_annotation_location = [[x.split('..') for x in x2] for x2 in np_annotation_location]
    np_annotation_location = [[[int(x) for x in x2] for x2 in x3] for x3 in np_annotation_location]
    pd_annotation["mRNA_pos"] = np_annotation_location
    pd_annotation["is_on_revcom"] = (pd_annotation["mRNA"].str.contains('complement'))
    pd_annotation = pd_annotation.drop("mRNA", axis=1)
    return pd_annotation

def f_common_intervals(list1, list2):
    '''
    Calculate exonal common parts
    (which exist in all isoforms)
    '''
    overlaps = []
    for a in list1:
        for b in list2:
            # Check if intervals overlap
            if a[1] >= b[0] and a[0] <= b[1]:
                overlap_start = max(a[0], b[0])
                overlap_end = min(a[1], b[1])
                overlaps.append([overlap_start, overlap_end])
    return overlaps

def f_mrna_common(pd_annotation, pd_gene, dict_chr):
    '''
    Calculate exonal common parts
    (which exist in all isoforms)
    '''
    print('##############################################')
    print(datetime.now(), "Calculate exonal common parts")
    i_end = len(pd_gene)
    list_pos_common = []
    list_entry = []
    
    for i_gene in range(i_end):
        gene_name = pd_gene.iloc[i_gene]
        f_print_10(i_gene, "Process gene No.", gene_name)
        # Extract the data of the gene of interest
        pd_process = pd_annotation[pd_annotation["gene"] == gene_name].reset_index()
        chr = pd_process['chr'].iloc[0]
        is_on_revcom = pd_process['is_on_revcom'].iloc[0]
        for i_row, row in pd_process.iterrows():
            list_temp = row["mRNA_pos"]
            if i_row == 0: # Make the common part list for the first time
                list_pos_common_single = copy.deepcopy(list_temp)
            if i_row > 0: # Get the common part
                list_pos_common_single = f_common_intervals(list_pos_common_single, list_temp)
        ids = pd_process["ID"].tolist()
        list_entry_single = [chr, is_on_revcom, ids, gene_name]
        list_pos_common.append(list_pos_common_single)
        list_entry.append(list_entry_single)
    
    pd_annotation_common = pd.concat([
        pd.DataFrame(list_entry, columns=["chr", "is_on_revcom", "mRNA", "gene"]),
        pd.Series(list_pos_common).rename('mRNA_pos')
    ], axis=1)

    pd_annotation_common["chr_file"] = pd_annotation_common["chr"].apply(lambda x: dict_chr[x])
    
    return pd_annotation_common

def f_union_intervals(list1, list2):
    '''
    Calculate exonal union parts
    (which exist in at least one isoform)
    '''
    # Merge both lists
    intervals = list1 + list2
    # Sort intervals based on start
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals:
        last_merged = merged[-1]
        # Check if the current interval overlaps or is adjacent to the last merged interval
        if current[0] <= last_merged[1]:
            # Merge overlapping intervals
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # If no overlap, just append the interval to the merged list
            merged.append(current)
    return merged

def f_mrna_union(pd_annotation, pd_gene, dict_chr):
    '''
    Calculate exonal union parts
    (which exist in at least one isoform)
    '''
    print('##############################################')
    print(datetime.now(), "Calculate exonal union parts")
    i_end = len(pd_gene)
    list_pos_union = []
    list_entry = []    
    for i_gene in range(i_end):
        gene_name = pd_gene.iloc[i_gene]
        f_print_10(i_gene, "Process gene No.", gene_name)
        # Extract the data of the gene of interest
        pd_process = pd_annotation[pd_annotation["gene"] == gene_name].reset_index()
        chr = pd_process['chr'].iloc[0]
        is_on_revcom = pd_process['is_on_revcom'].iloc[0]
        for i_row, row in pd_process.iterrows():
            list_temp = row["mRNA_pos"]
            if i_row == 0: # Make the union part list for the first time
                list_pos_union_single = copy.deepcopy(list_temp)
            if i_row > 0: # Get the union part
                list_pos_union_single = f_union_intervals(list_pos_union_single, list_temp)
        ids = pd_process["ID"].tolist()
        list_entry_single = [chr, is_on_revcom, ids, gene_name]
        list_pos_union.append(list_pos_union_single)
        list_entry.append(list_entry_single)
    
    pd_annotation_union = pd.concat([
        pd.DataFrame(list_entry, columns=["chr", "is_on_revcom", "mRNA", "gene"]),
        pd.Series(list_pos_union).rename('mRNA_pos')
    ], axis=1)

    pd_annotation_union["chr_file"] = pd_annotation_union["chr"].apply(lambda x: dict_chr[x])
    
    return pd_annotation_union

def f_parse_annotation(dir_database):
    '''
    Extract annotation
    '''    
    print('##############################################')
    print(datetime.now(), "Extract mRNA annotation")
    
    # Load data
    os.chdir(os.path.join(dir_database, "result", "chromosome"))
    with open('dict_chr.pickle', 'rb') as f:
        dict_chr = pickle.load(f)
    os.chdir(os.path.join(dir_database, "source"))
    with gzip.open("rna.gz", "rt") as handle:
        gb_annotation = list(SeqIO.parse(handle, "fasta"))
    print(datetime.now(), len(gb_annotation), "records were found")

    # Extract gene-coding mRNA only
    pd_annotation = f_extract_mRNA_annotation(gb_annotation, dict_chr)
    
    # Convert position string into np.array
    pd_annotation = f_parse_mrna_pos(pd_annotation)
    print()
    print("Example data:")
    print(pd_annotation.head())
    print()

    # Export
    filename = os.path.join(dir_database, "result", "mrna_pos.csv")
    pd_annotation.to_csv(filename, index=False)
    print("Saved to", filename)
    filename = os.path.join(dir_database, "result", "mrna_pos.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_annotation, f)
    print("Saved to", filename)

    # Gene list
    pd_gene = pd_annotation["gene"].drop_duplicates()
    n_gene = len(pd_gene)
    print(n_gene, "genes are found")    
    filename = os.path.join(dir_database, "result", "mrna_listed_gene.csv")
    pd_gene.to_csv(filename, index=False)
    print("Saved to", filename)

    # Calculate exonal common parts
    # (which exist in all isoforms)
    pd_annotation_common = f_mrna_common(pd_annotation, pd_gene, dict_chr)

    # Export
    filename = os.path.join(dir_database, "result", "mrna_common.csv")
    pd_annotation_common.to_csv(filename, index=False)
    print("Saved to", filename)
    filename = os.path.join(dir_database, "result", "mrna_common.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_annotation_common, f)
    print("Saved to", filename)

    # Calculate exonal union parts
    # (which exist in at least one isoform)
    pd_annotation_union = f_mrna_union(pd_annotation, pd_gene, dict_chr)

    # Export
    filename = os.path.join(dir_database, "result", "mrna_union.csv")
    pd_annotation_union.to_csv(filename, index=False)
    print("Saved to", filename)
    filename = os.path.join(dir_database, "result", "mrna_union.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_annotation_union, f)
    print("Saved to", filename)

############################
# Extract NGG from exonal parts 
############################

def f_extract_NGG_single_chr(chr, seq_f, seq_r, pd_process):
    '''
    Extract NGG on exonal regions
    '''
    # Process
    result_f = []
    result_r = []
    len_total = len(seq_f)
    
    for i_gene, row in pd_process.iterrows():
        gene = row['gene']
        for pos in row['mRNA_pos']:
            # Extract the substring corresponding to mRNA_pos
            # Most cases have [start, end]
            if len(pos) == 2:
                start_f, end_f = pos
            # Some rare cases have only [start] (1-nt fragment)
            else:
                start_f = pos[0]
                end_f = pos[0]
            # Python notation starts from 0, not 1
            # Pad 20 to cover 5'/3' ends
            start_f = start_f - 1 - 20
            end_f = end_f - 1 + 20

            start_r = len_total - end_f - 1
            end_r = len_total - start_f - 1
            
            substring_f = seq_f[start_f:end_f + 1]
            substring_r = seq_r[start_r:end_r + 1]
            
            # Find all occurrences of
            # 23-letter-long substrings ending in "GG"
            for match in re.finditer(r'(?=([ACG].{20}GG))', substring_f):
                result_f.append([gene, match.group(1), 
                                start_f + match.start() + 1, # Python notation starts from 0, not 1
                                start_f + match.start() + 23]
                              )
    
            for match in re.finditer(r'(?=([ACG].{20}GG))', substring_r):
                result_r.append([gene, match.group(1), 
                                start_r + match.start() + 1, # Python notation starts from 0, not 1
                                start_r + match.start() + 23]
                              )
    
    result_r = [[l[0], l[1], len_total - l[2] + 1, len_total - l[3] + 1] for l in result_r]
    
    # Convert results to a DataFrame
    result_f = pd.DataFrame(result_f, columns=["gene", "target", "start", "end"])
    result_r = pd.DataFrame(result_r, columns=["gene", "target", "start", "end"])
    result_f["chr"] = chr
    result_r["chr"] = chr
    result_f["strand"] = "+"
    result_r["strand"] = "-"
    result = pd.concat([result_f, result_r], axis=0)
    result = result.sort_values("start").reset_index(drop=True)
    return result

def f_extract_NGG_exonal(dir_database):
    '''
    Extract NGG on exonal regions
    '''
    print('##############################################')
    print(datetime.now(), "Extract NGG on exonal regions")
    print('Loading data...')
    
    # Exon position info
    filename = os.path.join(dir_database, "result", "mrna_common.pickle")
    with open(filename, mode='rb') as f:
        pd_annotation_common = pickle.load(f)
    
    filename = os.path.join(dir_database, "result", "mrna_union.pickle")
    with open(filename, mode='rb') as f:
        pd_annotation_union = pickle.load(f)
    
    # Chromosome list
    filename = os.path.join(dir_database, "result", "chromosome", "dict_chr.pickle")
    with open(filename, mode='rb') as f:
        dict_chr = pickle.load(f)
    list_chr = [value for key, value in dict_chr.items()]
    
    # Process per chromosome
    for chr in list_chr:
        print(datetime.now(), "Process chr No.", chr)
        filename = os.path.join(dir_database, "result", "chromosome", "chr_"+str(chr)+".fasta")
        with open(filename, "rb") as f:
            seq_f = list(SeqIO.parse(filename, "fasta"))[0].seq
        seq_r = seq_f.reverse_complement()
        seq_f = str(seq_f)
        seq_r = str(seq_r)
        print('Length = ', len(seq_f))
    
        # Common parts
        pd_process = pd_annotation_common[pd_annotation_common["chr_file"] == chr]
        pd_process = pd_process.reset_index(drop=True)[["gene", "mRNA_pos", "is_on_revcom"]]
        pd_result_single_chr = f_extract_NGG_single_chr(chr, seq_f, seq_r, pd_process)
        
        filename = os.path.join(dir_database, "result", "NGG_target_candidate", "all", "common", "NGG_common_chr_" + str(chr) + ".csv")
        pd_result_single_chr.to_csv(filename, index=False)
        print("Saved to", filename)
        filename = os.path.join(dir_database, "result", "NGG_target_candidate", "all", "common", "NGG_common_chr_" + str(chr) + ".pickle")
        with open(filename, mode='wb') as f:
            pickle.dump(pd_result_single_chr, f)
        print("Saved to", filename)
    
        # Union parts
        pd_process = pd_annotation_union[pd_annotation_union["chr_file"] == chr]
        pd_process = pd_process.reset_index(drop=True)[["gene", "mRNA_pos", "is_on_revcom"]]
        pd_result_single_chr = f_extract_NGG_single_chr(chr, seq_f, seq_r, pd_process)
        
        filename = os.path.join(dir_database, "result", "NGG_target_candidate", "all", "union", "NGG_union_chr_" + str(chr) + ".csv")
        pd_result_single_chr.to_csv(filename, index=False)
        print("Saved to", filename)
        filename = os.path.join(dir_database, "result", "NGG_target_candidate", "all", "union", "NGG_union_chr_" + str(chr) + ".pickle")
        with open(filename, mode='wb') as f:
            pickle.dump(pd_result_single_chr, f)
        print("Saved to", filename)

############################
# Sieve target sequences
############################

gap_penalty = -5 # both for opening and extanding

def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval

def match_score(alpha, beta):
    if alpha == '-' or beta == '-':
        return gap_penalty
    else:
        if alpha == 'A':
            if beta == 'A':
                return 10
            elif beta == 'G':
                return -1
            elif beta == 'C':
                return -3
            elif beta == 'T':
                return -4
        if alpha == 'G':
            if beta == 'A':
                return -1
            elif beta == 'G':
                return 7
            elif beta == 'C':
                return -5
            elif beta == 'T':
                return -3
        if alpha == 'C':
            if beta == 'A':
                return -3
            elif beta == 'G':
                return -5
            elif beta == 'C':
                return 9
            elif beta == 'T':
                return 0
        if alpha == 'T':
            if beta == 'A':
                return -4
            elif beta == 'G':
                return -3
            elif beta == 'C':
                return 0
            elif beta == 'T':
                return 8

def finalize(align1, align2):
    align1 = align1[::-1]    #reverse sequence 1
    align2 = align2[::-1]    #reverse sequence 2
    
    i,j = 0,0
    
    #calcuate identity, score and aligned sequeces
    symbol = ''
    found = 0
    score = 0
    identity = 0
    for i in range(0,len(align1)):
        # if two AAs are the same, then output the letter
        if align1[i] == align2[i]:                
            symbol = symbol + align1[i]
            identity = identity + 1
            score += match_score(align1[i], align2[i])
    
        # if they are not identical and none of them is gap
        elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-': 
            score += match_score(align1[i], align2[i])
            symbol += ' '
            found = 0
    
        #if one of them is a gap, output a space
        elif align1[i] == '-' or align2[i] == '-':          
            symbol += ' '
            score += gap_penalty
    
    return score

def needle(seq1, seq2):
    '''
    Needlman-Wunsch score
    '''
    m, n = len(seq1), len(seq2)  # length of two sequences
    
    # Generate DP table and traceback path pointer matrix
    score = zeros((m+1, n+1))      # the DP table
   
    # Calculate DP table
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + match_score(seq1[i-1], seq2[j-1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)

    # Traceback and compute the alignment 
    align1, align2 = '', ''
    i,j = m,n # start from the bottom right cell
    while i > 0 and j > 0: # end toching the top or the left edge
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]

        if score_current == score_diagonal + match_score(seq1[i-1], seq2[j-1]):
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
        elif score_current == score_up + gap_penalty:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1

    # Finish tracing up to the top left cell
    while i > 0:
        align1 += seq1[i-1]
        align2 += '-'
        i -= 1
    while j > 0:
        align1 += '-'
        align2 += seq2[j-1]
        j -= 1

    return finalize(align1, align2)

def f_sieve_target_1(dir_database):
    '''
    Sieve the target candidates, part 1
    - The very minimal requirements
    '''
    print('##############################################')
    print(datetime.now(), "Sieve the target candidates, part 1")
    print(datetime.now(), "Concatenate csv files for exonal common parts")
    dir_name = filename = os.path.join(dir_database, "result", "NGG_target_candidate", "all", "common")
    pd_common = f_concatenate_csv_in_dir(dir_name)
    print()
    print(datetime.now(), "Concatenate csv files for exonal union parts")
    dir_name = filename = os.path.join(dir_database, "result", "NGG_target_candidate", "all", "union")
    pd_union = f_concatenate_csv_in_dir(dir_name)
    print()
    print(datetime.now(), "Merge")
    pd_merged = pd_union.merge(pd_common, indicator='variant_common', how='left')
    pd_merged['variant_common'] = pd_merged['variant_common'].replace({'both': 1, 'left_only': 0})
    
    print(datetime.now(), "Convert to actual gRNA sequence by changing the initial to G")
    pd_merged['target_initialG'] = 'G' + pd_merged['target'].str[1:]
    print(datetime.now(), "Omit duplicates")
    pd_merged = pd_merged.drop_duplicates(subset='target_initialG', keep=False).reset_index(drop=True)
    
    print(datetime.now(), "Drop entries containing N")
    pd_merged["invalid_base"] = pd_merged["target_initialG"].apply(lambda x: "N" in x)
    pd_merged = pd_merged[pd_merged["invalid_base"] == False].reset_index(drop=True)
    pd_merged = pd_merged.drop("invalid_base", axis=1)
    f_show_stat(pd_merged)

    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_merged_before_structure.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_merged, f)
    print(datetime.now(), "Saved to", filename)


def f_add_RNA_structure(dir_database, n_core):
    '''
    Apply RNAfold calculation in parallel, using `multiprocessing
    '''
    print('##############################################')
    print(datetime.now(), "Calculate sgRNA secondary structure")
    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_merged_before_structure.pickle")
    with open(filename, mode='rb') as f:
        pd_merged = pickle.load(f)

    print("Add stem-loop sequences")
    pd_merged["sgRNA"] = pd_merged["target_initialG"].str[:20] + \
        'GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUUU'
 
    # Estimate completion time, as this part takes much time
    def f_test(i):
        return pd_merged["sgRNA"][:10**i].apply(lambda x: RNA.fold(x))
    f_estimate_time(f_test, pd_merged.shape[0] / n_core)
    
    # Split dataframe into chunks, and throw to multiprocessing
    def split_dataframe(df, n):
        chunks = np.array_split(df, n)
        return chunks
    chunks = split_dataframe(pd_merged, n_core)
    # Apply RNAfold
    result = cmulti.main_multiprocessing(cmulti.fold_rna_apply, chunks, n_core)
    
    # Merge the result
    list_structure = pd.concat(result)
    pd_merged[["structure", "energy"]] = pd.DataFrame(list_structure.tolist())
    
    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_merged_with_structure.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_merged, f)
    print(datetime.now(), "Saved to", filename)

def f_sieve_target_2(dir_database):
    '''
    Sieve the target candidates, part 2
    - Important requirements, but all the entries will be preserved for being traced back later
    '''
    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_merged_with_structure.pickle")
    with open(filename, mode='rb') as f:
        pd_merged = pickle.load(f)

    # Flag if GC >= 60%
    print(datetime.now(), "Flag if GC > 60%")
    pd_merged["GC"] = pd_merged["target_initialG"].apply(lambda x: gc_fraction(x[:20]) * 100)
    pd_merged["omit_GC60"] = pd_merged["GC"].apply(lambda x: 1 if x >= 60 else 0)
    
    # Flag if containing TTTT
    print(datetime.now(), "Flag if containing TTTT")
    pd_merged["omit_TTTT"] = pd_merged["target_initialG"].apply(lambda x: 1 if "TTTT" in x else 0)
    
    # Dimer calculation
    print(datetime.now(), "Flag if dimerizing with vector sequence")
    # Estimate completion time, as this part takes much time
    def f_test(i):
        return pd_merged["target_initialG"].iloc[0:(10**i)].apply(lambda x: needle('AAAAGCACCGACTCGGTGCC', x[:20]))
    f_estimate_time(f_test, pd_merged.shape[0])
    # Proceed to calculation
    pd_merged["NW_score"] = pd_merged["target_initialG"].apply(lambda x: needle('AAAAGCACCGACTCGGTGCC', x[:20]))
    print('Flag if score > 60')
    pd_merged["omit_dimerize_with_vector"] = pd_merged["NW_score"].apply(lambda x: 1 if x > 60 else 0)

    # Evaluate by RNAfold result
    pd_merged["omit_wrong_structure"] = pd_merged["structure"].apply(lambda x: 0 if
                                                                     (x[20:50] == "(((((((.((((....))))...)))))))"
                                                                      and
                                                                      x[68:80] == "((((....))))"
                                                                      and
                                                                      x[81:96] == "((((((...))))))") else 1
                                                                    )
    pd_merged["omit_high_energy"] = pd_merged["energy"].apply(lambda x: 0 if x < -18 else 1)

    # Export
    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_merged_all_before_offtarget.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_merged, f)
    print(datetime.now(), "Saved to", filename)

def f_prepare_massive_search(dir_database):
    '''
    Prepare the list for offtarget & qPCR search
    Entries with `Omit_` flags are dropped to save calculation time
    Some genes with few entries are additionally salvaged
    '''
    print(datetime.now(), "Prepare the list of target candidates for off-target search")
    print()
    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_merged_all_before_offtarget.pickle")
    with open(filename, mode='rb') as f:
        pd_merged = pickle.load(f)    
    print("Show stats for the `whole` dataset with all candidates:")
    f_show_stat(pd_merged)
    print()
    
    print("Show stats for the `strict` dataset with candidates passing filters:")
    pd_strict = (pd_merged[
     (pd_merged["omit_GC60"] == 0) &
     (pd_merged["omit_TTTT"] == 0) &
     (pd_merged["omit_dimerize_with_vector"] == 0) &
     (pd_merged["omit_wrong_structure"] == 0) &
     (pd_merged["omit_high_energy"] == 0)
     ])
    
    f_show_stat(pd_strict)
    print()

    # Salvage
    # Threshold loosened to 6+1 instead of 6...
    # in case a target is dropped in the subsequent off-target search
    # because of perfect match
    print("For the genes with only < 6 + 1 candidates, try salvaging candidates from the `whole` list:")
    pd_count = pd_strict['gene'].value_counts()
    pd_count_enough = pd_count[pd_count >= 7]
    list_gene_enough = pd_count_enough.keys().tolist()
    
    set_gene_salvage = set(pd_merged['gene'].unique()) - set(list_gene_enough)
    
    print("Salvage applied to", len(set_gene_salvage), "genes in combination")
    print()
    
    pd_salvage = pd_merged[pd_merged["gene"].isin(set_gene_salvage)]
    pd_strict_plus_salvage = pd.concat([pd_strict, pd_salvage])
    pd_strict_plus_salvage = pd_strict_plus_salvage.sort_index().drop_duplicates()
    
    print("Show stats for the salvaged dataset:")
    f_show_stat(pd_strict_plus_salvage)

    print()
    print("Save as .pickle")

    filename = os.path.join(dir_database, "result", "NGG_target_candidate", "pd_selected_before_offtarget.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(pd_strict_plus_salvage, f)
    print(datetime.now(), "Saved to", filename)

    # Cut in chunks and save
    prefix = "NGG_target_candidate"

    # For offtarget search: cut into 1M chunk
    list_merged = pd_strict_plus_salvage[["start", "end", "target_initialG"]].values.tolist()
    dir_save = os.path.join(dir_database, "result", "NGG_target_candidate", "for_offtarget_search")
    f_cut_export(list_merged, "off_target", dir_save, prefix)

    # For qPCR search: cut into 10K chunk
    dir_save = os.path.join(dir_database, "result", "NGG_target_candidate", "for_qPCR_search")
    f_cut_export(pd_strict_plus_salvage, "qPCR", dir_save, prefix, n_cut_size=10000)


############################
# Extract NAG/NGG on whole genome
############################

def f_extract_NGG_genomic(dir_database):
    '''
    Extract all [20nt]NAG/NGG on genome
    NAG can also act as off-target
    '''
    # Chromosome list
    print('##############################################')
    print(datetime.now(), "Extract all [20nt]NAG/NGG on genome")
    filename = os.path.join(dir_database, "result", "chromosome", "dict_chr.pickle")
    with open(filename, mode='rb') as f:
        dict_chr = pickle.load(f)

    total_NGG_num = 0
    for chr_name, chr in dict_chr.items():
        print(datetime.now(), "Process chr No.", chr, "(accession:", chr_name, ")")
        filename = os.path.join(dir_database, "result", "chromosome", "chr_"+str(chr)+".fasta")
        with open(filename, "rb") as f:
            seq_f = list(SeqIO.parse(filename, "fasta"))[0].seq
        seq_r = seq_f.reverse_complement()
        seq_f = str(seq_f)
        seq_r = str(seq_r)
        len_total = len(seq_f)
        match_f = re.finditer(".....................[AG]G", seq_f, re.I)
        # Use a capturing group inside a lookahead
        # in order to capture all overlapping matches
        # (by default, `re.finditer only captures non-overlapping matches)
        match_f = re.finditer(r'(?=(.....................[AG]G))', seq_f)
        match_r = re.finditer(r'(?=(.....................[AG]G))', seq_r)
        list_match_f = [[m.start() + 1, m.start() + 23, m.group(1)] for m in match_f if not "N" in m.group(1)]
        list_match_r = [[len_total - m.start() - 22, len_total - m.start(), m.group(1)] for m in match_r if not "N" in m.group(1)]
        print(len(list_match_f), "(forward),", len(list_match_r), "(rev-com) NAG/NGG were found")
        total_NGG_num += len(list_match_f) + len(list_match_r)
        # Export as csv
        filename = os.path.join(dir_database, "result", "NGG_genomic", "csv", "NGG_genomic_chr_" + str(chr) + "_f.csv")
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list_match_f)
        filename = os.path.join(dir_database, "result", "NGG_genomic", "csv", "NGG_genomic_chr_" + str(chr) + "_r.csv")
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(list_match_r)
        # Export as pickle, binarized 20-nt sequence only
        dir_save = os.path.join(dir_database, "result", "NGG_genomic", "pickle")
        prefix = "NGG_genomic_chr_" + '0' * (3-len(str(chr))) + str(chr) + "_f"
        f_cut_export(list_match_f, "off_target", dir_save, prefix)
        prefix = "NGG_genomic_chr_" + '0' * (3-len(str(chr))) + str(chr) + "_r"
        f_cut_export(list_match_r, "off_target", dir_save, prefix)
    print(total_NGG_num, "NAG/NGG were found in the whole genome")

############################
# Off-target search
# Functions involving `multiprocessing` are 
# defined in `custom_multicore_off_target.py`
############################

def f_offtarget_prepare_args(dir_database):

    '''
    Prepare the arguments to be passed to
    `cmulti.calc_offtarget_score` (via `cmulti.main_multiprocessing`)
    '''

    # List files in directories
    dir_genome = os.path.join(dir_database, "result", "NGG_genomic", "pickle")
    dir_target = os.path.join(dir_database, "result", "NGG_target_candidate", "for_offtarget_search")    
    files_genome = [os.path.join(dir_genome, file) for file in os.listdir(dir_genome)]
    files_target = [os.path.join(dir_target, file) for file in os.listdir(dir_target)]
    
    # Combinations of files from two directories
    list_args = [(dir_database, file_genome, file_target, 200) for file_genome in files_genome for file_target in files_target]
    
    n_core_available = get_n_core()

    return list_args, n_core_available

def f_offtarget_result_sum(dir_database):
    '''
    Sum up off-target scores calculated in parallel jobs
    '''
    print(datetime.now(), "Sum up the calculated off-target scores")
    dir_score = os.path.join(dir_database, "result", "off_target")
    print("Process files in", dir_score)
    list_file = [os.path.join(dir_score, file) for file in os.listdir(dir_score)]
    list_file = sorted(list_file)
    df_result = []
    print(len(list_file), "files are found")
    for i, file_score in enumerate(list_file, 1):
        # filename_complete = "offtarget_score_" + index_target + "_x_" + index_genome + ".pickle"
        file_name_base = os.path.splitext(os.path.basename(file_score))[0]
        file_name_base_split = file_name_base.split("_")
        index_target = int(file_name_base_split[2])

        print(datetime.now(), "Processing file", i, ":", file_name_base)
        
        # Load each chunk
        with open(file_score, 'rb') as f:
            result_chunk = pickle.load(f)
        result_chunk = [[l[1] + index_target, l[2]] for l in result_chunk]
        df_chunk = pd.DataFrame(result_chunk, columns=['index_target', 'off_target_score'])
        
        # Concatenate chunks, summing up scores for the same target
        if i == 1:
            df_result = df_chunk.copy()
        else:
            df_result = pd.concat([df_result, df_chunk])
        df_result = (df_result
                     .groupby('index_target', as_index=False).sum()
                     .astype({'index_target': 'int64', 'off_target_score': 'float64'})
                    )

    filename = os.path.join(dir_database, "result", "off_target_score_sum.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(df_result, f)
    print(datetime.now(), "Saved to", filename)
    print(datetime.now(), "Done.")

############################
# qPCR primers
# Functions involving `multiprocessing` are 
# defined in `custom_multicore_qPCR.py`
############################

def f_qPCR_prepare_args(dir_database):
    '''
    Prepare the arguments to be passed to
    `cmulti.calc_qPCR` (via `cmulti.main_multiprocessing`)
    '''
    dir_target = os.path.join(dir_database, "result", "NGG_target_candidate", "for_qPCR_search")
    files_target = [os.path.join(dir_target, file) for file in os.listdir(dir_target)]
    
    # Combinations of files from two directories
    list_args = [(dir_database, file_target) for file_target in files_target]
    
    n_core_available = get_n_core()

    return list_args, n_core_available

def f_qPCR_result_aggregate(dir_database):
    '''
    Collect qPCR primers calculated in parallel jobs
    '''
    print(datetime.now(), "Sum up the calculated off-target scores")
    dir_primer = os.path.join(dir_database, "result", "qPCR")
    print("Process files in", dir_primer)
    list_file = [os.path.join(dir_primer, file) for file in os.listdir(dir_primer)]
    list_file = sorted(list_file)
    df_result = []
    print(len(list_file), "files are found")
    for i, file_primer in enumerate(list_file, 1):
        # filename example: qPCR_primer_000000000000000.pickle
        file_name_base = os.path.splitext(os.path.basename(file_primer))[0]
        file_name_base_split = file_name_base.split("_")
        index_target = int(file_name_base_split[2])
        print(datetime.now(), "Processing file", i, ":", file_name_base)
        # Load each chunk
        with open(file_primer, 'rb') as f:
            df_chunk = pickle.load(f)
        # Concatenate chunks, summing up scores for the same target
        if i == 1:
            df_result = df_chunk.copy()
        else:
            df_result = pd.concat([df_result, df_chunk])
    print(df_result.shape[0], "entries were found in total. Please confirm that this number is the same as the number of gRNA targets.")
    filename = os.path.join(dir_database, "result", "qPCR_primer_aggregated.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(df_result, f)
    print(datetime.now(), "Saved to", filename)
    print(datetime.now(), "Done.")

############################
# Final processing
############################

def check_non_overlapping(group):
    '''
    Flag non-overlapping guides
    This function is applied to a subset of Pandas DataFrame
        corresponding to each single gene
    Add a column `non_overlapping`, whose value is:
    - 1 for the first entry
    - 1 for the second or later entry, where
        its `start` value differs from the `start` value
        in all of the previous entry with
        1 in `non_overlapping` column
        by 20 or more
    - 0 otherwise
    '''
    group['non_overlapping'] = 0
    for i in group.index:
        if i == group.index[0] or all(
            abs(
                group.at[i, 'start'] - group.loc[group['non_overlapping'] == 1, 'start']
               ) >= 53
        ):
            group.at[i, 'non_overlapping'] = 1
    return group

def get_chunk_summary(chunk, chunk_summary, filename_suffix):
    '''
    Get the list of genes for each split file
    '''
    chunk_summary_single = chunk[["gene"]].drop_duplicates().reset_index(drop=True)
    chunk_summary_single["file"] = filename_suffix
    if len(chunk_summary) == 0:
        chunk_summary = chunk_summary_single.copy()
    else:
        chunk_summary = pd.concat([chunk_summary, chunk_summary_single], axis=0)
    return chunk_summary

def f_cut_save_final(dir_database, data, filename_base):
    '''
    Export the finalized version of database
    as .csv, cut into small files
    '''
    dirname = os.path.join(dir_database,
                            "result", 
                            "final",
                            filename_base)    
    print(datetime.now(), "Cut into chunks & export to", dirname)
    
    chunk_size = 10000
    chunk_start = 0
    chunk_summary = []
    n_file = 0
    while chunk_start < len(data):
        chunk_end = chunk_start + chunk_size
        
        # If the chunk_end of the DataFrame is reached, take the rest of the DataFrame
        if chunk_start + chunk_size >= len(data):
            n_file += 1
            chunk = data[chunk_start:]
            filename_suffix = "0" * (5-len(str(n_file))) + str(n_file)
            filename = os.path.join(dirname,
                                    filename_base + "_" + filename_suffix + ".csv")
            chunk.to_csv(filename, index=False)
            chunk_summary = get_chunk_summary(chunk, chunk_summary, filename_suffix)
            break

        # Expand the chunk to cover all the entries with the same gene
        while chunk_end < len(data) and data.iloc[chunk_end - 1]['gene'] == data.iloc[chunk_end]['gene']:
            chunk_end += 1
        n_file += 1
        chunk = data[chunk_start:chunk_end]
        filename_suffix = "0" * (5-len(str(n_file))) + str(n_file)
        filename = os.path.join(dirname,
                                filename_base + "_" + filename_suffix + ".csv")
        chunk.to_csv(filename, index=False)
        chunk_summary = get_chunk_summary(chunk, chunk_summary, filename_suffix)
        chunk_start = chunk_end
    print(datetime.now(), "Completed export to", n_file, "file(s)")

    # Export summary
    dirname = os.path.join(dir_database,
                            "result", 
                            "final")    
    filename = os.path.join(dirname,
                            filename_base + "_summary.csv")

    chunk_summary.to_csv(filename, index=False)
    print(datetime.now(), "Exported summary to", filename)
    print()

def f_save_final(dir_database, data, filename_base):
    '''
    Export the finalized version of database
     - As .pickle
     - As .csv
     - As .csv, cut into small files
    '''
    
    # Export as .pickle
    filename = os.path.join(dir_database,
                            "result", 
                            "final",
                            filename_base + ".pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)
    print(datetime.now(), "Saved to", filename)

    # Export as .csv
    filename = os.path.join(dir_database,
                            "result", 
                            "final",
                            filename_base + ".csv")
    # Index was omitted as it may change with future updates
    data.to_csv(filename, index=False)
    print(datetime.now(), "Saved to", filename)
    
    # Chop and save
    f_cut_save_final(dir_database, data, filename_base)

def f_final_processing(dir_database):
    '''
    Combine the results of all analyses and export
     - Transform off-target scores to the final form
     - Flag where qPCR primers are missing
    '''
    print(datetime.now(), "Start final processing")
    print("Reading data...")
    
    # Everything else than off-target search and qPCR search
    filename = os.path.join(dir_database, "result", 
                            "NGG_target_candidate", 
                            "pd_selected_before_offtarget.pickle")
    print("Everything else than off-target search and qPCR search:", filename)
    with open(filename, mode='rb') as f:
        pd_main = pickle.load(f).reset_index(drop=True)

    # Off-target score
    filename = os.path.join(dir_database, "result", "off_target_score_sum.pickle")
    print("Off-target scores:", filename)
    with open(filename, mode='rb') as f:
        pd_offtarget = pickle.load(f).set_index('index_target', drop=True)

    # qPCR primers
    filename = os.path.join(dir_database, "result", "qPCR_primer_aggregated.pickle")
    print("qPCR primers:", filename)
    with open(filename, mode='rb') as f:
        pd_qpcr = pickle.load(f).reset_index(drop=True)

    # Combine all datasets
    pd_combined = pd.concat([pd_main, pd_offtarget, pd_qpcr], axis=1)

    print(datetime.now(), "Final calculation of off-target scores...")
    # Fill in zero for the targets with no off-targets (which were calculated as NaN)
    pd_combined['off_target_score'] = pd_combined['off_target_score'].fillna(0)

    # Subtract scores corresponding to targetting itself
    pd_G = pd_combined[pd_combined['target'].str.startswith('G')]
    pd_ATC = pd_combined[~pd_combined['target'].str.startswith('G')]
    pd_G['off_target_score'] -= 100000
    pd_ATC['off_target_score'] -= 1
    pd_combined = pd.concat([pd_G, pd_ATC]).sort_index()
    pd_combined['off_target_score'] = 100 / (1 + pd_combined['off_target_score'])
    pd_combined['omit_offtarget_score_low'] = pd_combined['off_target_score'].apply(
        lambda x: 1 if x < 50 else 0)
    
    # Flag for missing qPCR primer(s)
    print(datetime.now(), "Final calculation of qPCR primers...")
    list_search_columns = ['1st_left_seq', '1st_left_Tm',
                           '1st_right_1_seq', '1st_right_1_Tm', '1st_right_1_product_size', 
                           '1st_right_2_seq', '1st_right_2_Tm', '1st_right_2_product_size',
                           '1st_right_3_seq', '1st_right_3_Tm', '1st_right_3_product_size',
                           '2nd_left_seq', '2nd_left_Tm',
                           '2nd_right_1_seq', '2nd_right_1_Tm', '2nd_right_1_product_size',
                           '2nd_right_2_seq', '2nd_right_2_Tm', '2nd_right_2_product_size',
                           '2nd_right_3_seq', '2nd_right_3_Tm', '2nd_right_3_product_size']
    
    pd_combined['omit_qPCR_not_found'] = pd_combined[list_search_columns].apply(
        lambda row: 1 if "not_found" in row.values else 0, 
        axis=1
    )

    # AAV construction primer
    print(datetime.now(), "Add primer sequences for AAV construction...")
    pd_combined["AAV_primer_f"] = pd_combined["target_initialG"].apply(
        lambda x: x[:20] + "GTTTTAGAGCTAGAAATAGCAAGTTAAA")
    pd_combined["AAV_primer_r"] = pd_combined["target_initialG"].apply(
        lambda x: str(Seq(x[:20]).reverse_complement() + "GGTGTTTCGTCCTTTCCACAAGATA"))

    # Finally, show basic statistics
    f_show_stat(pd_combined)

    # Sort by off-target score (highest to lowest) 
    # for each gene
    print(datetime.now(), "Sort...")

    # Preserving the order of genes
    pd_sorted_1 = (pd_combined
     .groupby('gene', group_keys=False, sort=False)
     .apply(lambda x: x.sort_values('off_target_score', ascending=False))
     .reset_index(drop=True)
    )

    # Annotate non-overlapping entries
    # Apply the function to each 'gene' group
    pd_sorted_1 = (pd_sorted_1
        .groupby('gene').apply(check_non_overlapping)
        .reset_index(drop=True))

    print()
    print(datetime.now(), "Export the final version")

    f_save_final(dir_database, pd_sorted_1, "final_sort_pos")
    
    # Sort genes by name
    # Prepare another DataFrame for sorting by
    # gene name, case-INsensitive
    pd_combined_with_lower = pd_combined.copy()
    pd_combined_with_lower['gene_lower'] = pd_combined_with_lower['gene'].str.lower()
    # Sort
    pd_sorted_2 = (pd_combined_with_lower
     .groupby('gene_lower', group_keys=False, sort=True)
     .apply(lambda x: x.sort_values('off_target_score', ascending=False))
     .reset_index(drop=True)
     .drop("gene_lower", axis=1)
    )

    # Annotate non-overlapping entries
    # Apply the function to each 'gene' group
    pd_sorted_2 = (pd_sorted_2
        .groupby('gene').apply(check_non_overlapping)
        .reset_index(drop=True))

    # Save
    f_save_final(dir_database, pd_sorted_2, "final_sort_name")

    print(datetime.now(), "All done, congratulations!")