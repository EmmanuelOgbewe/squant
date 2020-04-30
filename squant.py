# squant.py 
#@author Emmanuel Ogbewe 

import io
import sys
import gzip
import csv
import timer 
import scipy
import functools
from numba import jit
import numpy as np
import scipy.stats
from scipy.special import ndtr
from scipy.stats import spearmanr
import benchmarks as bm


#------Handle input------#
def parseCommandLineArgs(args=[]):
 
    if '--in' not in args or '--out' not in args:
        print("Arguments must contain --in and --out flags.")
        return 
    if len(args) != 6 and len(args) != 7:
        print("Incorrect number of arguments.")
        return 

    alignmentPath = args[3]
    output = args[5]
    if '--eqc' in args:
        useEquiv = True
    else: 
        useEquiv = False 
        print("Using equivalence class model.")


#------Helpers------#
def memoizer(f):
    memolist = [-1] * 1000
    def _f(l):
        if memolist[l] == -1:
            memolist[l] = f(l)
        return memolist[l]
    return _f


# @functools.lru_cache(maxsize=None)
@memoizer
def get_eff_len_short(l):
    mu = 200
    sd = 25

    d = scipy.stats.norm(mu, sd)

    # the (discrete) distribution up to 200
    p = np.array([d.pdf(i) for i in range(l+1)])
    # re-normalize so this is a proper distribution
    p /= p.sum()
    # the expected value of a distribution f is \sum_{i=0}^{max_i} f(i) * i
    cond_mean = np.sum([i * p[i] for i in range(len(p))])
    return l - cond_mean


def generate_eff_len(l):
    mu = 200
    return l - mu if l >= 1000 else get_eff_len_short(l)


#------End------#


#------Main Functions------#

fld = scipy.stats.norm(200, 25)
lookup = [fld.cdf(i) for i in range(1000)]

def cdf(x):
    return 1.0 if x >= 1000 else lookup[x]

    
def readFile(input_file):
    tran_dict = {}
    names_list = []
    count = 0 
    t_idx = 0 

    time = timer.Timer()
    input_file = "data/alignments.cs423.gz"
    # gz = gzip.open(input_file, 'rb')
    # f = io.BufferedReader(gz)

    time.start()
    transcript = ''
    elens = {}

    with gzip.open(input_file,'rt') as f:
        while True:
            # add to transcript dictionary
            line = f.readline()
            if t_idx == 0:
                nt = int(line.strip())
                print(nt)

            if t_idx != 0 and t_idx <= nt: 
                transcript = line.strip().split("\t")
                t_len = int(transcript[1])
                if not t_len in elens.keys():
                    elen = generate_eff_len(t_len)
                    elens[t_len] = elen
                else:
                    elen = elens[t_len]
                tran_dict[transcript[0]] = (t_len,elen,t_idx)
                names_list.append(transcript[0])
                # print(transcript[0],tran_dict[transcript[0]])
                print(t_idx)
            t_idx += 1
            
            if t_idx == (nt + 1): 
                break
        time.stop()
        print("finished processing transripts")

        time.start()
        print("processing alignments")
    # # process alignments 
        label_list = []
        prob_list = []
        ind_list = [0]
        count = 0
        ns = {}
        d = scipy.stats.norm(200, 25)
        D = d.cdf

        while True: 
            line = f.readline()
            if line == "":
                break  
            num_al = int(line)
            for al in range(num_al):
                toks = f.readline().strip().split("\t")
                tn = toks[0]
                ori = toks[1]
                tpos = int(toks[2])
                P_4 = float(toks[3])
                tlen,elen,tidx = tran_dict[tn]
                label_list.append(tidx)
                # calculate P_2
                if elen in ns:
                    P_2 = ns[elen]
                else:
                    P_2 = 1 / elen
                    ns[elen] = 1 / elen
                # P_3s
                if ori == "rc":
                    npos = tpos + 100
                else:
                    npos = tlen - tpos

                P_3 = cdf(npos)
                prob_list.append(P_2 * P_3 * P_4)

            ind_list.append(len(prob_list))
        
        time.stop()

    label_list = np.array(label_list)
    prob_list = np.array(prob_list)
    ind_list = np.array(ind_list)
    print("Starting em")
    time.start()
    eta, iterations = run_em_fullmodel(nt, label_list, ind_list, prob_list)
    print("Number of iterations: " + str(iterations))
    time.stop() 
    writeToOutput("test_out.tsv", names_list, tran_dict, ind_list, eta)


@jit(nopython=True)
def run_em_fullmodel(num_t,label_list, ind_list, p_list):
    eta = np.ones(num_t) / float(num_t) 
    eta_p  = np.zeros(num_t) 
    
    # enr = np.zeros(num_t) # estimated number of reads
    converged = False 

    it = 0
    ni = len(ind_list)-1
    readCount = 0 
    while True: 
        it += 1 
        for i in range(ni): #loop through the number of alignment blocks
            norm = 0.0 
            for j in range(ind_list[i],ind_list[i+1]):
                # re-normalize / denominator
                readCount += 1
                norm += (p_list[j] * eta[label_list[j]])

            for j in range(ind_list[i], ind_list[i+1]):
                if (ind_list[i+1] - ind_list[i]) == 1:
                    eta_p[label_list[j]] = eta_p[label_list[j]] + 1.0
                    break
                pr = p_list[j] * eta[label_list[j]]
                re_norm = pr / norm
                eta_p[label_list[j]] = eta_p[label_list[j]] + re_norm 

        # conduct M step 
        eta_p = eta_p / float(readCount)
        # check for convergence 
        for i in range(len(eta_p)):
            current = eta_p[i]
            previous = eta[i]
            change = (current-previous)
            if abs(change) <= 0.000001:
                converged = True 
            else:
                converged = False 
                break
        if converged:
            break
        
        eta = eta_p 
        eta_p = np.zeros(num_t)
        print(it)
    #truncacte small didgits in eta 
    for tr in range(len(eta_p)):
        if eta_p[tr] < 1e-10:
            eta_p[tr] = 0.0
    return eta_p, it

def writeToOutput(output_file, tn_names, tran_dict, ind_list, eta ):
   
    time = timer.Timer()
    time.start()
    print("Writing to output")
    with open (output_file, 'w') as f: 
        f.write("name\teffective_length\test_frag\n")
        for i in range(len(eta)):
            name = tn_names[i]
            _,elen,_ = tran_dict[name]
            elen = format(elen, '.3f')
            est_frag = eta[i]
            f.write(name + " " + "\t" + elen + " " + "\t" + str(est_frag) + "\n")
    time.stop()
    print("Finshed Writing to output")


#Assessing results
def readResults(expected_file,real_file):
    tsv_file = open(expected_file, newline="") 
    read_tsv = csv.reader(tsv_file, delimiter="\t" )
    count = 0 
    expected_count = np.zeros(203835)
    next(read_tsv)
    for line in read_tsv: 
        expected_count[count] = float(line[1])
        count += 1

    tsv_file.close()

    tsv_file = open(real_file, newline="") 
    read_tsv = csv.reader(tsv_file, delimiter="\t" )
    count = 0 
    real_count = np.zeros(203835)
    next(read_tsv)
    for line in read_tsv: 
        real_count[count] = float(line[2])
        count += 1
    tsv_file.close()
    return expected_count, real_count

def run_benchmarks(expected, actual):

    corr = spearmanr(actual,expected)
    print("Spearmann Correlation : " + str(corr))
    print("Mean Error: " + str(bm.me(actual, expected)))
    print("Mean Absoulte Error: " + str(bm.mae(actual,expected)))
    print("Mean Squared Error: " + str(bm.mse(actual,expected)))
    print("Root Mean Squared Error" + str(bm.rmse(actual,expected)))

    
if __name__ == "__main__":
    # parseCommandLineArgs(sys.argv)
    readFile("")
    exp,actual = readResults("data/true_counts.tsv","test_out.tsv")
    run_benchmarks(exp,actual)
    # print(generate_eff_len())