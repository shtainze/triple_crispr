# Definition of functions using `multiprocessing`

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

############################
# General
############################

def main_multiprocessing(func, list_args, core_num):    
    '''
    Multiprocessing for any arbitrary function `func`
    '''
    print(datetime.now(), "Start processing", len(list_args), "jobs in total")
    print("Use", core_num, "cores in parallel")
    with Pool(min(core_num, cpu_count() - 2)) as pool:
        result = pool.map(func, list_args)
    print(datetime.now(), "Done parallel processing.")
    return result

flatten = lambda *args: (
    result for mid in args
    for result in (
        flatten(*mid)
        if isinstance(mid, (tuple, list))
        else (mid,)))

def dif_adjacent_list(l_input):
    '''
    Iterate through the list and for
        each element, compute the difference
        with the next element in the list
    Omit the empty difference and
        append the last element
    Finally, reverse the order and
        flatten it as a plain 2-d list
    '''
    l_result = []    
    for i in range(len(l_input) - 1):
        current_set = set(map(tuple, l_input[i]))
        next_set = set(map(tuple, l_input[i+1]))
        difference = current_set - next_set
        l_result.append(list(map(list, difference)))
    l_result = [l for l in l_result if len(l) > 0]
    l_result.append(l_input[-1])

    l_result_2 = []
    for l in list(reversed(l_result)):
        l_result_2.extend(l)
    return l_result_2

############################
# RNAfold for sgRNA structure prediction
############################

def fold_rna(x):
    return RNA.fold(x)


def fold_rna_apply(chunk):
    '''
    Apply RNA.fold to each chunk in parallel
    '''
    return chunk["sgRNA"].apply(fold_rna)

############################
# Off-target score
############################

# Weight of mismatches for calculating off-target score
s_weight = 1 - np.array([0, 0, 0.014, 0, 0,
                         0.395, 0.317, 0, 0.389, 0.079,
                         0.445, 0.508, 0.613, 0.851, 0.732,
                         0.828, 0.615, 0.804, 0.685, 0.583])

def offtarget_score_single(mismatch_pos_single):
    '''
    Calculate an offtarget score based on a list of
    mismatch positions
    '''
    n_mismatch = len(mismatch_pos_single)
    if n_mismatch == 0:
        n_score = 100000  # Omit if a perfect match
    else:
        if n_mismatch == 1:
            n_meanDistance = 19
        if n_mismatch > 1:
            n_meanDistance = (mismatch_pos_single[-1] - mismatch_pos_single[0])/(n_mismatch - 1)
        n_weight_prod = np.prod(s_weight[mismatch_pos_single])
        n_score = n_weight_prod * (1 / (1 + 4 * (19 - n_meanDistance) / 19))\
            * (1 / n_mismatch ** 2)
    return n_score

def calc_offtarget_score(args):
    '''
    Calculate offtarget scores for a set of files for:
    - a chunk of NAGs/NGGs on the genome
    - a chunk of gRNA target candidates
    The result output will be a list of lists,
        each of which having three members:
    [index_genome:
        which genomic NAG/NGG inside `file_genome`
     index_target:
        which exonal NGG inside `file_target`
     score:
        a single off-target score between the above two,
        later to be summed up for each exonal NGG
    ]
    '''
    dir_database, file_genome, file_target, size_chunk = args
    index_target = os.path.splitext(os.path.basename(file_target))[0]
    index_genome = os.path.splitext(os.path.basename(file_genome))[0]
    index_target = index_target.replace("NGG_target_candidate_", "")
    index_genome = index_genome.replace("NGG_genomic_chr_", "")

    # Empty temporary file to mark the job is ongoing
    filename_temp = "offtarget_score_" + index_target + "_x_" + index_genome + ".temp.pickle"
    filename_temp = os.path.join(dir_database, "result", "off_target", filename_temp)
    # Final result
    filename_complete = "offtarget_score_" + index_target + "_x_" + index_genome + ".pickle"
    filename_complete = os.path.join(dir_database, "result", "off_target", filename_complete)

    # Skip if already working
    if os.path.isfile(filename_temp):
        return 0    
    # Skip if already done
    if os.path.isfile(filename_complete):
        return 0

    # Proceed if not yet done and not being done
    print(datetime.now(), "Calculate off-target scores for",
        "targets:", index_target,
        "genome:", index_genome,
        "in process", os.getpid())

    # Dump an empty temporary file to mark the job is ongoing
    with open(filename_temp, mode='wb') as f:
        pickle.dump([], f)

    result = []

    # Load lists
    with open(file_genome, 'rb') as f_genome:
        list_genome = pickle.load(f_genome)
    with open(file_target, 'rb') as f_target:
        list_target = pickle.load(f_target)

    list_target = [list(x) for x in list_target]
    np_target = np.array(list_target, dtype=np.int32)

    # Convert lists to numpy arrays
    size_genome = len(list_genome)
    i_end = math.ceil(size_genome / size_chunk)
    for i in range(i_end):
        i_chunk_start = i * size_chunk
        i_chunk_end = min((i+1)*size_chunk, size_genome)
        list_genome_chunk = [list(x) for x in list_genome[i_chunk_start: i_chunk_end]]
        np_genome = np.array(list_genome_chunk, dtype=np.int32)        
        # Extract the 0,1,2,3,4-mismatch pairs
        product = np.dot(np_genome, np_target.T)
        match = np.where(product > 15)
        match_genome = np_genome[match[0]]
        match_target = np_target[match[1]]
        match_cross = match_genome * match_target
        # Mismatch positions
        match_pos = [np.nonzero(l)[0] for l in match_cross]
        match_pos = [np.floor(l/4) for l in match_pos]
        match_pos = [np.array(l, dtype=np.int32) for l in match_pos]
        mismatch_pos = [np.setdiff1d(range(20), l) for l in match_pos]
        # Calculate score
        list_score = [offtarget_score_single(l) for l in mismatch_pos]
        # Position in the entire dataset
        match_final = np.array(match)
        match_final[0] = match_final[0] + i_chunk_start
        # Append to result
        result_chunk = match_final.tolist()
        result_chunk.append(list_score)
        result_chunk = [list(l) for l in zip(*result_chunk)]
        result.extend(result_chunk)
    # Output
    with open(filename_complete, mode='wb') as f:
        pickle.dump(result, f)

    # Delete the temporary file
    os.remove(filename_temp)
    print(datetime.now(), "Output to", filename_complete)
    return 0

############################
# qPCR primer search
############################


# Parameters
N_PRIMER_LENGTH_MIN = 21
N_PRIMER_LENGTH_MAX = 32
N_PRODUCT_SIZE_MAX = 200
DICT_GLOBAL_PARAMETERS_RIGHT = {
    # Overall
    'PRIMER_TASK': 'generic',
    'PRIMER_PICK_LEFT_PRIMER': 0,
    'PRIMER_PICK_RIGHT_PRIMER': 1,
    'PRIMER_THERMODYNAMIC_OLIGO_ALIGNMENT': 1,
    'PRIMER_THERMODYNAMIC_TEMPLATE_ALIGNMENT': 0,
    'PRIMER_TM_FORMULA': 1,
    'PRIMER_NUM_RETURN': 10000,
    'PRIMER_EXPLAIN_FLAG': 1,
    'PRIMER_FIRST_BASE_INDEX': 1,
    # Sequence
    'PRIMER_LIBERAL_BASE': 0,  # Strict!
    'PRIMER_MAX_POLY_X': 5,
    'PRIMER_MUST_MATCH_THREE_PRIME': 'nnnnv',  # Avoid final T
    'PRIMER_MIN_SIZE': N_PRIMER_LENGTH_MIN,
    'PRIMER_OPT_SIZE': 29,
    'PRIMER_MAX_SIZE': N_PRIMER_LENGTH_MAX,
    'PRIMER_GC_CLAMP': 0,
    'PRIMER_MIN_GC': 30.0,
    'PRIMER_OPT_GC_PERCENT': 50.0,
    'PRIMER_MAX_GC': 80.0,
    'PRIMER_MAX_END_GC': 5,
    'PRIMER_MAX_TEMPLATE_MISPRIMING': 12.00,
    'PRIMER_MAX_TEMPLATE_MISPRIMING_TH': 47.00,
    'PRIMER_MAX_NS_ACCEPTED': 1,  # Omit later
    # Structure
    'PRIMER_MIN_TM': 60,
    'PRIMER_OPT_TM': 65,
    'PRIMER_MAX_TM': 70,
    'PRIMER_MAX_SELF_ANY': 100,  # Strict!
    'PRIMER_MAX_SELF_ANY_TH': 47.00,
    'PRIMER_MAX_SELF_END': 3,  # Strict!
    'PRIMER_MAX_SELF_END_TH': 47.00,
    'PRIMER_MAX_HAIRPIN_TH': 47.00,
    'PRIMER_MAX_END_STABILITY': 9.0,
    # Combination of Forward + Revcom
    'PRIMER_PRODUCT_SIZE_RANGE': [[60, 200]],
    'PRIMER_PAIR_MAX_DIFF_TM': 10.0,
    'PRIMER_PAIR_MAX_COMPL_ANY': 3,  # Strict!
    'PRIMER_PAIR_MAX_COMPL_ANY_TH': 47.00,
    'PRIMER_PAIR_MAX_COMPL_END': 3.00,
    'PRIMER_PAIR_MAX_COMPL_END_TH': 47.00,
    'PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING': 24.00,
    'PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING_TH': 47.00,
    # Solution
    'PRIMER_SALT_CORRECTIONS': 1,
    'PRIMER_SALT_DIVALENT': 1.5,
    'PRIMER_SALT_MONOVALENT': 50.0,
    'PRIMER_DNA_CONC': 50.0,
    'PRIMER_DNTP_CONC': 0.6,
    # REDUNDANT OR AS DEFAULT
    'PRIMER_MIN_3_PRIME_OVERLAP_OF_JUNCTION': 4,
    'PRIMER_MIN_5_PRIME_OVERLAP_OF_JUNCTION': 4,
    'PRIMER_PRODUCT_OPT_SIZE': 60,
    'PRIMER_INSIDE_PENALTY': -1.0,
    'PRIMER_OUTSIDE_PENALTY': 0.0,
    'PRIMER_PICK_ANYWAY': 0,  # Strict!
    'PRIMER_SEQUENCING_ACCURACY': 20,
    'PRIMER_SEQUENCING_INTERVAL': 250,
    'PRIMER_SEQUENCING_LEAD': 50,
    'PRIMER_SEQUENCING_SPACING': 500,
    'PRIMER_MIN_END_QUALITY': 0,
    'PRIMER_MIN_QUALITY': 0,
    'PRIMER_QUALITY_RANGE_MAX': 100,
    'PRIMER_QUALITY_RANGE_MIN': 0,
    'PRIMER_LOWERCASE_MASKING': 0,
    # 'PRIMER_MISPRIMING_LIBRARY': 'NONE', Need not specify in the latest version
    'PRIMER_MAX_LIBRARY_MISPRIMING': 12.00,
    'PRIMER_PAIR_MAX_LIBRARY_MISPRIMING': 24.00,
    'PRIMER_LIB_AMBIGUITY_CODES_CONSENSUS': 0,
    'PRIMER_WT_END_QUAL': 0.0,
    'PRIMER_WT_END_STABILITY': 0.0,
    'PRIMER_WT_LIBRARY_MISPRIMING': 0.0,
    'PRIMER_WT_NUM_NS': 0.0,
    'PRIMER_WT_POS_PENALTY': 0.0,
    'PRIMER_WT_SEQ_QUAL': 0.0,
    'PRIMER_WT_TEMPLATE_MISPRIMING': 0.0,
    'PRIMER_WT_TEMPLATE_MISPRIMING_TH': 0.0,
    'PRIMER_WT_HAIRPIN_TH': 0.0,
    'PRIMER_WT_SELF_ANY': 0.0,
    'PRIMER_WT_SELF_ANY_TH': 0.0,
    'PRIMER_WT_SELF_END': 0.0,
    'PRIMER_WT_SELF_END_TH': 0.0,
    'PRIMER_PAIR_WT_COMPL_ANY': 0.0,
    'PRIMER_PAIR_WT_COMPL_ANY_TH': 0.0,
    'PRIMER_PAIR_WT_COMPL_END': 0.0,
    'PRIMER_PAIR_WT_COMPL_END_TH': 0.0,
    'PRIMER_PAIR_WT_DIFF_TM': 0.0,
    'PRIMER_PAIR_WT_IO_PENALTY': 0.0,
    'PRIMER_PAIR_WT_LIBRARY_MISPRIMING': 0.0,
    'PRIMER_PAIR_WT_TEMPLATE_MISPRIMING': 0.0,
    'PRIMER_PAIR_WT_TEMPLATE_MISPRIMING_TH': 0.0,
    'PRIMER_PAIR_WT_PRODUCT_SIZE_GT': 0.0,
    'PRIMER_PAIR_WT_PRODUCT_SIZE_LT': 0.0,
    'PRIMER_PAIR_WT_PRODUCT_TM_GT': 0.0,
    'PRIMER_PAIR_WT_PRODUCT_TM_LT': 0.0,
    'PRIMER_PAIR_WT_PR_PENALTY': 1.0,
    'PRIMER_WT_GC_PERCENT_GT': 0.0,
    'PRIMER_WT_GC_PERCENT_LT': 0.0,
    'PRIMER_WT_SIZE_GT': 1.0,
    'PRIMER_WT_SIZE_LT': 1.0,
    'PRIMER_WT_TM_GT': 1.0,
    'PRIMER_WT_TM_LT': 1.0,
    'PRIMER_PICK_INTERNAL_OLIGO': 0,
    'PRIMER_INTERNAL_DNA_CONC': 50.0,
    'PRIMER_INTERNAL_DNTP_CONC': 0.0,
    'PRIMER_INTERNAL_MAX_GC': 80.0,
    'PRIMER_INTERNAL_MAX_HAIRPIN_TH': 47.00,
    'PRIMER_INTERNAL_MAX_LIBRARY_MISHYB': 12.00,
    'PRIMER_INTERNAL_MAX_NS_ACCEPTED': 0,
    'PRIMER_INTERNAL_MAX_POLY_X': 5,
    'PRIMER_INTERNAL_MAX_SELF_ANY': 12.00,
    'PRIMER_INTERNAL_MAX_SELF_ANY_TH': 47.00,
    'PRIMER_INTERNAL_MAX_SELF_END': 12.00,
    'PRIMER_INTERNAL_MAX_SELF_END_TH': 47.00,
    'PRIMER_INTERNAL_MAX_SIZE': 27,
    'PRIMER_INTERNAL_MAX_TM': 63.0,
    'PRIMER_INTERNAL_MIN_GC': 20.0,
    'PRIMER_INTERNAL_MIN_QUALITY': 0,
    'PRIMER_INTERNAL_MIN_SIZE': 18,
    'PRIMER_INTERNAL_MIN_TM': 57.0,
    # 'PRIMER_INTERNAL_MISHYB_LIBRARY': 'NONE', Need not specify in the latest version
    'PRIMER_INTERNAL_OPT_GC_PERCENT': 50.0,
    'PRIMER_INTERNAL_OPT_SIZE': 20,
    'PRIMER_INTERNAL_OPT_TM': 60.0,
    'PRIMER_INTERNAL_SALT_DIVALENT': 0.0,
    'PRIMER_INTERNAL_SALT_MONOVALENT': 50.0,
    'PRIMER_INTERNAL_WT_END_QUAL': 0.0,
    'PRIMER_INTERNAL_WT_GC_PERCENT_GT': 0.0,
    'PRIMER_INTERNAL_WT_GC_PERCENT_LT': 0.0,
    'PRIMER_INTERNAL_WT_HAIRPIN_TH': 0.0,
    'PRIMER_INTERNAL_WT_LIBRARY_MISHYB': 0.0,
    'PRIMER_INTERNAL_WT_NUM_NS': 0.0,
    'PRIMER_INTERNAL_WT_SELF_ANY': 0.0,
    'PRIMER_INTERNAL_WT_SELF_ANY_TH': 0.0,
    'PRIMER_INTERNAL_WT_SELF_END': 0.0,
    'PRIMER_INTERNAL_WT_SELF_END_TH': 0.0,
    'PRIMER_INTERNAL_WT_SEQ_QUAL': 0.0,
    'PRIMER_INTERNAL_WT_SIZE_GT': 1.0,
    'PRIMER_INTERNAL_WT_SIZE_LT': 1.0,
    'PRIMER_INTERNAL_WT_TM_GT': 1.0,
    'PRIMER_INTERNAL_WT_TM_LT': 1.0,
}

def f_calc_dimer_single(seq1, seq2):
    '''
    Calculate longest consecutive dimer length
    between `seq1` and `seq2`
    '''
    ndarray_seq1 = np.array(list(seq1))
    ndarray_seq2 = np.array(list(seq2))
    ndarray_match = ndarray_seq1 == ndarray_seq2
    ndarray_pos = np.where(ndarray_match)[0]
    if len(ndarray_pos) > 0:
        ndarray_match = ndarray_match[ndarray_pos[0]: ndarray_pos[-1] + 1]
        n_result = 2 * sum(ndarray_match) - len(ndarray_match)
    else:
        n_result = 0
    return n_result

def f_calc_dimer(seq1, seq2):
    '''
    Calculate the longest consecutive dimer length
    between ANY subparts of `seq1` and `seq2`
    '''
    n_max = 0
    # Set ranges for which to try alignments
    i_range = [
        [max(len(seq1) - i, 0),
         min(len(seq1), len(seq1) - i + len(seq2)),
         max(len(seq2) - i, 0),
         min(len(seq2), len(seq2) - i + len(seq1))
        ] for i in range(1, len(seq1) + len(seq2))]
    # Try all alignments
    l_seq = [
        [Seq(seq1[item[0]: item[1]]),
         Seq(seq2[item[2]: item[3]]).reverse_complement()]
        for item in i_range]
    l_seq = [f_calc_dimer_single(item[0], item[1]) for item in l_seq]
    return max(l_seq)

def f_qPCR_dimer_3end(seq1, seq2, n_nt):
    '''
    Calculate how many nucleotides are identical
    in `seq1` and `seq2`, for the last `n_nt` nucleotides
    '''
    # Last `n_nt` nucleotides of `seq1`, forward
    np_seq1 = np.array(list(seq1[-n_nt:]))
    # Last `n_nt` nucleotides of `seq1`, rev-com
    np_seq2 = np.array(list(
        Seq(seq2[-n_nt:]).reverse_complement()))
    return sum(np_seq1 == np_seq2)

def f_qPCR_sieve_unit(l_temp_0, n_threshold_0, n_sieve):
    '''
    Sieve `l_single_0` according to two criteria:
    Stop if only `n_sieve` candidates are left
        → flag="limited"
    Stop if all candidates have a value not more than `n_threshold_0`
        → flag="unlimited"
    '''
    flag = "limited"
    l_temp = copy.deepcopy(l_temp_0)
    if len(l_temp) != 0:  # if not empty
        n_item = len(l_temp[0]) - 1  # last column
        if len(l_temp) > n_sieve:  # Ignore if already finished sieving
            l_temp.sort(key=itemgetter(n_item))
            l_temp_2 = [item for item in l_temp if item[n_item] <= n_threshold_0]
            if len(l_temp_2) >= n_sieve:  # stop if n_threshold_0 is achieved
                l_temp = l_temp_2
                flag = "unlimited"
            else:   # sieve until n_sieve candidates are left
                for i in range(n_sieve, len(l_temp)):
                    if l_temp[i - 1][n_item] < l_temp[i][n_item]:
                        l_temp = l_temp[0: i]
                        break
        l_temp = [item[0: n_item] for item in l_temp]
    return l_temp, flag

def f_qPCR_listup_left(str_search):
    '''
    List up left primer candidates
    Just listing up everything possible
    '''
    l_primer = []
    for n_primer_length in range(N_PRIMER_LENGTH_MIN, N_PRIMER_LENGTH_MAX + 1):
        n_search_start = N_PRIMER_LENGTH_MAX - n_primer_length
        n_search_end = n_search_start + n_primer_length
        l_primer_append = [
            [i, str_search[i: i + n_primer_length]]
            for i in range(n_search_start, n_search_end + 1)]
        l_primer.extend(l_primer_append)
    return l_primer

def f_qPCR_sieve_left(l_left_0, n_overlap_0):
    '''
    Pick only one left primer
    But store the list of leftovers for
        a possible later salvage
    '''
    l_left = copy.deepcopy(l_left_0)
    l_result = []
    # Omit if containing indefinite bases
    l_left = [
        item for item in l_left
        if len(re.findall('[^ATGC]', item[1])) == 0] 
    # Obligatory: overlap on the CRISPR cut site
    l_left = [
        item for item in l_left
        if item[0] <= N_PRIMER_LENGTH_MAX - n_overlap_0 and
        item[0] + len(item[1]) >= N_PRIMER_LENGTH_MAX] 
    # Obligatory: 55 <= Tm <= 70
    l_left = [
        item + [pr.calc_tm(
            item[1], mv_conc=50, dv_conc=0, dntp_conc=0, dna_conc=50,
            tm_method='santalucia', salt_corrections_method='santalucia')]
        for item in l_left]
    l_left = [
        item for item in l_left
        if 55 <= item[2] and item[2] <= 70]
    l_result.append(l_left)
    # Optional from here on
    # Continue until onl one entry remains
    if len(l_left) > 0:       
        # Sequence: last should not be T
        l_left_temp = [
            item for item in l_left
            if item[1][-1] != 'T']
        if len(l_left_temp) != 0:
            l_left = l_left_temp
        l_result.append(l_left)
        # 3' = leftmost
        l_left = [
            item + [item[0] + len(item[1])]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 0, 1)
        l_result.append(l_left)
        # Tm: 60-70
        l_left = [
            item + [abs(item[2] - 65)]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 5, 1)
        l_result.append(l_left)
        # No self dimer
        l_left = [
            item + [f_qPCR_dimer_3end(item[1], item[1], 2)]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 1, 1)
        l_result.append(l_left)
        l_left = [
            item + [f_qPCR_dimer_3end(item[1], item[1], 6)]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 3, 1)
        l_result.append(l_left)
        l_left = [
            item + [f_calc_dimer(item[1], item[1])]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 3, 1)
        l_result.append(l_left)
        # Tm =  closest to 65
        l_left = [
            item + [abs(item[2] - 65)]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 0, 1)
        l_result.append(l_left)
        # GC % = closest to 50
        l_left = [
            item + [abs(gc_fraction(item[1]) * 100 - 50)]
            for item in l_left]
        l_left, flag = f_qPCR_sieve_unit(l_left, 5, 1)
        l_result.append(l_left)
    if len(l_left) > 0:
        l_left = l_left[0] # There's only one entry
    l_result = dif_adjacent_list(l_result)
    return l_result

def f_qPCR_parse_right(l_right):
    '''
    Extract the info of right primers:
    sequence, TM, GC%, location, length
    Convert to Pandas.DataFrame for ease of processing
    Return a list at the end
    '''
    pd_right = pd.DataFrame(l_right)
    pd_right = pd_right[["SEQUENCE", "COORDS", "TM", "GC_PERCENT"]]
    #pd_right[["pos", "len"]] = pd_right["COORDS"].apply(lambda x: pd.Series([x[0], x[1]]))
    pd_right['pos'] = pd_right['COORDS'].map(lambda x: x[0])
    pd_right['len'] = pd_right['COORDS'].map(lambda x: x[1])
    pd_right = pd_right.drop(["COORDS"], axis=1)
    return pd_right.values.tolist()

def f_qPCR_sieve_right(l_right_0, l_left):
    '''
    Pick three right primers
    Flag if any obligatory selection hits the limit
    '''
    flag_limit_hit = 100000
    l_right = copy.deepcopy(l_right_0)
    l_right = [
        item for item in l_right
        if len(re.findall('[^ATGC]', item[0])) == 0]  # Omit unrecognized bases
    # Obligatory: product size = 60-200
    l_right = [item for item in l_right
        if item[3] - l_left[0] >= 60
        and item[3] - l_left[0] <= 200]
    # Obligatory: no self dimer
    l_right = [
        item + [f_qPCR_dimer_3end(item[0], item[0], 2)]
        for item in l_right]
    l_right, flag = f_qPCR_sieve_unit(l_right, 1, 3)
    if flag == "limited" and flag_limit_hit == 100000:
        flag_limit_hit = 1
    l_right = [
        item + [f_qPCR_dimer_3end(item[0], item[0], 6)]
        for item in l_right]
    l_right, flag = f_qPCR_sieve_unit(l_right, 3, 3)
    if flag == "limited" and flag_limit_hit == 100000:
        flag_limit_hit = 2
    l_right = [
        item + [f_calc_dimer(item[0], item[0])]
        for item in l_right]
    l_right, flag = f_qPCR_sieve_unit(l_right, 3, 3)
    if flag == "limited" and flag_limit_hit == 100000:
        flag_limit_hit = 3
    # Obligatory: no left-right dimer
    l_right = [
        item + [f_qPCR_dimer_3end(item[0], l_left[1], 2)]
        for item in l_right]
    l_right, flag = f_qPCR_sieve_unit(l_right, 1, 3)
    if flag == "limited" and flag_limit_hit == 100000:
        flag_limit_hit = 4
    l_right = [
        item + [f_qPCR_dimer_3end(item[0], l_left[1], 6)]
        for item in l_right]
    l_right, flag = f_qPCR_sieve_unit(l_right, 3, 3)
    if flag == "limited" and flag_limit_hit == 100000:
        flag_limit_hit = 5
    l_right = [
        item + [f_calc_dimer(item[0], l_left[1])]
        for item in l_right]
    l_right, flag = f_qPCR_sieve_unit(l_right, 3, 3)
    if flag == "limited" and flag_limit_hit == 100000:
        flag_limit_hit = 6
    # Optional: sequence
    if len(l_right) > 3:
        l_right_temp = [
            item for item in l_right
            if item[0][-1] != 'T']
        if len(l_right_temp) >= 3:
            l_right = l_right_temp
    if len(l_right) > 3:
        l_right_temp = [
            item for item in l_right
            if item[0][-1] != 'A']
        if len(l_right_temp) >= 3:
            l_right = l_right_temp
    # Optional: Tm difference <= 3
    if len(l_right) > 3:
        l_right = [
            item + [abs(item[1] - l_left[2])]
            for item in l_right]
        l_right, flag = f_qPCR_sieve_unit(l_right, 3, 3)
    # Optional: Product size = around 80
    if len(l_right) > 3:
        l_right = [
            item + [item[3] - l_left[0]]
            for item in l_right]
        l_right, flag = f_qPCR_sieve_unit(l_right, 80, 3)
    # Finalize by Tm difference and product size (smaller is better)
    if len(l_right) > 3:
        l_right = [
            item + [abs(item[1] - l_left[2])]
            for item in l_right]
        l_right, flag = f_qPCR_sieve_unit(l_right, 0, 3)
    if len(l_right) > 3:
        l_right = [
            item + [item[3] - l_left[0]]
            for item in l_right]
        l_right, flag = f_qPCR_sieve_unit(l_right, 0, 3)
    # If dead locked, put on a strict rule
    if len(l_right) > 3:  
        l_right = [
            item + [100 * abs(item[1] - l_left[2]) + item[3] - l_left[0]]
            for item in l_right]
        l_right, flag = f_qPCR_sieve_unit(l_right, 0, 3)
    return l_right, flag_limit_hit

def f_calc_right(str_search, l_left_candidate):
    '''
    Calculate the right primers
    '''
    l_right = []
    flag_limit_hit = 0
    DICT_GLOBAL_PARAMETERS_RIGHT['SEQUENCE_PRIMER'] = l_left_candidate[1]
    dict_right = pr.bindings.design_primers(
         {
             'SEQUENCE_TEMPLATE': str_search,
             'SEQUENCE_EXCLUDED_REGION': [[1, 60 - N_PRIMER_LENGTH_MAX]]
         }, DICT_GLOBAL_PARAMETERS_RIGHT)
    l_right = dict_right["PRIMER_RIGHT"]
    if len(l_right) > 0:
        l_right = f_qPCR_parse_right(l_right)
        l_right, flag_limit_hit = f_qPCR_sieve_right(l_right, l_left_candidate)
    return l_right, flag_limit_hit

def f_qPCR_parse_result(l_left_0, l_right_0):
    '''
    Convert the lists of left/right primers
    into a 1-d list for later concatenation
    Members:
    left primer (sequence, Tm),
    right primer 1 (sequence, Tm, product length),
    right primer 2 (sequence, Tm, product length),
    right primer 3 (sequence, Tm, product length),
    '''
    l_left = copy.deepcopy(l_left_0)
    l_right = copy.deepcopy(l_right_0)
    l_result_temp = []
    if len(l_left) > 0 and len(l_right) > 0:
        l_result_temp = l_left[1:] + [
        [item[0], # sequence
         item[1], # Tm
         item[3] - l_left[0], # product length
        ]
        for item in l_right]
        l_result_temp = list(flatten(l_result_temp))
    elif len(l_left) > 0 and len(l_right) == 0:
        l_result_temp = l_left[1:]
        l_result_temp = list(flatten(l_result_temp))
    else:  # len(l_left) == 0
        l_result_temp = []
    return l_result_temp

def f_qPCR_calc_pair(str_search, n_overlap=17):
    '''
    Calculate the best qPCR primer sets
    `i`: index (later used to concatenate to the main dataset)
    `str_search`: sequence to search within
    `n_overlap`: number of nucleotides that must overlap
    with the target region
    '''
    l_left = f_qPCR_listup_left(str_search)
    l_left = f_qPCR_sieve_left(l_left, n_overlap)
    l_left_best = [] # The best candidate
    l_right_best = [] # The best candidate
    l_result = ["not_found"] * 11 # Default - applicable on error
    if len(l_left) > 0:
        i_left = 0
        i_left_end = len(l_left)
        flag_limit_hit_max = 0
        # Scan through `l_left` until we find 
            # a left primer that is
            # compatible with three right primers
        # Most of the time, the first one in `l_left` is enough 
        while i_left < i_left_end:
            l_left_candidate = l_left[i_left]
            l_right, flag_limit_hit = f_calc_right(str_search, l_left_candidate)
            # If three right primer pairs were found without limitation
            if flag_limit_hit == 100000:
                l_left_best = l_left_candidate.copy()
                l_right_best = l_right.copy()
                break
            # If still limited, but relatively unlimited than the previous trial:
            elif flag_limit_hit > flag_limit_hit_max:
                # Register this trial
                flag_limit_hit_max = flag_limit_hit
                l_left_best = l_left_candidate.copy()
                l_right_best = l_right.copy()
            i_left += 1
        if len(l_left_best) > 0: # If it has value
            l_result = f_qPCR_parse_result(l_left_best, l_right_best)
    return l_result

def f_qPCR_single(single_entry, dir_database, seq_f, seq_r, n_len):
    '''
    Calculate qPCR primer sets x plus/minus-strand
    '''
    strand = single_entry['strand'].values[0]
    n_start_forward_1 = single_entry["start"].values[0]
    
    if strand == '-':
        n_start_forward_1 = n_len - n_start_forward_1 + 1
    
    # Ranges within which to search for primers
    # Plus strand
    n_beforecut_1 = n_start_forward_1 + 16  # Cas9 digestion site: between 3/4nt upper to NGG
    n_search_start_1 = n_beforecut_1 - N_PRIMER_LENGTH_MAX + 3 + 1
    n_search_end_1 = n_beforecut_1 + N_PRODUCT_SIZE_MAX - 6
    # Minus strand
    n_beforecut_2 = n_len - n_beforecut_1 # Cas9 digestion site: between 3/4nt upper to NGG
    n_search_start_2 = n_beforecut_2 - N_PRIMER_LENGTH_MAX + 3 + 1
    n_search_end_2 = n_beforecut_2 + N_PRODUCT_SIZE_MAX - 6
    
    if strand == '+':
        str_search_1 = seq_f[n_search_start_1 - 1: n_search_end_1]
        str_search_2 = seq_r[n_search_start_2 - 1: n_search_end_2]
    if strand == '-':
        str_search_1 = seq_r[n_search_start_1 - 1: n_search_end_1]
        str_search_2 = seq_f[n_search_start_2 - 1: n_search_end_2]
    
    # Search 1st primers
    l_result_single_1 = f_qPCR_calc_pair(str_search_1)    
    # Search 2nd primers
    l_result_single_2 = f_qPCR_calc_pair(str_search_2)
    l_result_single_1.extend(l_result_single_2)
    return l_result_single_1, n_len

def calc_qPCR(args):
    '''
    Calculate qPCR primers for:
    - a chunk of gRNA target candidates
    '''
    dir_database, file_target = args
    index_target = os.path.splitext(os.path.basename(file_target))[0]
    index_target = index_target.replace("NGG_target_candidate_", "")

    # Empty file to mark the job is ongoing
    filename_temp = "qPCR_primer_" + index_target + ".temp.pickle"
    filename_temp = os.path.join(dir_database, "result", "qPCR", filename_temp)
    # Final result
    filename_complete = "qPCR_primer_" + index_target + ".pickle"
    filename_complete = os.path.join(dir_database, "result", "qPCR", filename_complete)

    # Skip if already working
    if os.path.isfile(filename_temp):
        return 0    
    # Skip if already done
    if os.path.isfile(filename_complete):
        return 0

    # Proceed if not yet done and not being done
    print(datetime.now(), "Calculate qPCR primers for",
        "targets:", index_target,
        "in process", os.getpid())

    # Dump an empty temporary file to mark the job is ongoing
    with open(filename_temp, mode='wb') as f:
        pickle.dump([], f)

    # Load list
    with open(file_target, 'rb') as f_target:
        pd_target = pickle.load(f_target)

    chr_previous = -100000 # Initialize as non-existent number
    n_len = -100000    
    l_result = []
    i_end = len(pd_target)
    
    for i_entry in range(i_end):
        single_entry = pd_target[i_entry:i_entry+1]
        chr_now = single_entry["chr"].values[0]
        # Load chromosomal sequence data if not yet loaded
        if chr_previous != chr_now:
            print("Load new")
            filename = os.path.join(dir_database, "result", "chromosome", "chr_"+str(chr_now)+".fasta")
            with open(filename, "rb") as f:
                seq_f = list(SeqIO.parse(filename, "fasta"))[0].seq
            seq_r = seq_f.reverse_complement()
            seq_f = str(seq_f)
            seq_r = str(seq_r)
            n_len = len(seq_f)
            chr_previous = chr_now
        l_result_single, n_len = f_qPCR_single(single_entry, dir_database, seq_f, seq_r, n_len)
        l_result.append(l_result_single)
    
    pd_result = pd.DataFrame(l_result, columns=[
        "1st_left_seq", "1st_left_Tm",
        "1st_right_1_seq", "1st_right_1_Tm", "1st_right_1_product_size",
        "1st_right_2_seq", "1st_right_2_Tm", "1st_right_2_product_size",
        "1st_right_3_seq", "1st_right_3_Tm", "1st_right_3_product_size",
        "2nd_left_seq", "2nd_left_Tm",
        "2nd_right_1_seq", "2nd_right_1_Tm", "2nd_right_1_product_size",
        "2nd_right_2_seq", "2nd_right_2_Tm", "2nd_right_2_product_size",
        "2nd_right_3_seq", "2nd_right_3_Tm", "2nd_right_3_product_size",    
    ])

    # Output
    with open(filename_complete, mode='wb') as f:
        pickle.dump(pd_result, f)

    # Delete the temporary file
    os.remove(filename_temp)
    print(datetime.now(), "Output to", filename_complete)
    return 0