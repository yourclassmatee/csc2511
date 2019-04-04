#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    if use_cached == True:
        with open(fn_LM + '.pickle', 'rb') as handle:
            lm = pickle.load(handle)
            return lm
    else:
        lm = lm_train(data_dir, language, fn_LM)
        return lm


def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached == True:
        with open(fn_AM + '.pickle', 'rb') as handle:
            #print(fn_AM)
            am = pickle.load(handle)
            return am
    else:
        am = align_ibm1(data_dir, num_sent, max_iter, fn_AM)
        return am

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    bleu_arr = []
    for i, decoded in enumerate(eng_decoded):
        if n == 1:
            bleu_score = BLEU_score(decoded, [eng[i], google_refs[i]], 1, True)
            bleu_arr.append(bleu_score)
        elif n == 2:
            #recover bpc
            bpc = BLEU_score(decoded, [eng[i], google_refs[i]], 1, True) / \
                    BLEU_score(decoded, [eng[i], google_refs[i]], 1, False)
            bleu_score = (BLEU_score(decoded, [eng[i], google_refs[i]], 1, False) * \
                         BLEU_score(decoded, [eng[i], google_refs[i]], 2, False)) ** (1/n)
            bleu_score *= bpc
            bleu_arr.append(bleu_score)
        elif n == 3:
            # recover bpc
            bpc = BLEU_score(decoded, [eng[i], google_refs[i]], 1, True) / \
                  BLEU_score(decoded, [eng[i], google_refs[i]], 1, False)
            bleu_score = (BLEU_score(decoded, [eng[i], google_refs[i]], 1, False) * \
                          BLEU_score(decoded, [eng[i], google_refs[i]], 2, False) * \
                          BLEU_score(decoded, [eng[i], google_refs[i]], 3, False)) ** (1 / n)
            bleu_score *= bpc
            bleu_arr.append(bleu_score)

    return bleu_arr

   

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    

    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    data_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    testing_dir = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f"
    testing_file = open(testing_dir, 'r')
    testing_file = testing_file.read().split("\n")
    ref_dir = ["/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e", "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e"]

    # read test sentences
    test_sents = []
    for line in testing_file:
        if line.strip() != "":
            test_sents.append(preprocess(line,'f'))

    # read reference sentences
    ref_sents = dict()
    ref_sents["hansard"] = []
    ref_sents["google"] = []
    for i,each_ref in enumerate(ref_dir):
        ref_file = open(each_ref, 'r')
        ref_file = ref_file.read().split("\n")
        for line in ref_file:
            if line.strip() != "":
                if i == 0:
                    ref_sents["hansard"].append(preprocess(line,'e'))
                elif i == 1:
                    ref_sents["google"].append(preprocess(line, 'e'))


    f = open("Task5.txt", 'w+')
    #f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    am_training_num = [1000, 10000, 15000, 30000]
    #am_training_num = [1000]

    AMs = []
    AM_names = ["am_1k", "am_10k", "am_15k", "am_30k"]
    #AM_names = ["am_1k"]
    for i,training_num in enumerate(am_training_num):
        #print("Training AM %s"%AM_names[i])

        #set last param to true to use cached model
        AMs.append(_getAM(data_dir, training_num, 1, AM_names[i], False))
        #AMs.append(_getAM(data_dir, training_num, 1, AM_names[i], False))

    #set last param to true to use cached model
    LM = _getLM(data_dir, 'f', "lm_save", False)
    #LM = _getLM(data_dir, 'f', "lm_save", False)


    for i, AM in enumerate(AMs):
        
        f.write("### Evaluating AM model: %s ### \n"%AM_names[i])

        #print(len(test_sents), len(ref_sents["hansard"]), len(ref_sents["google"]))

        # Decode using AM #
        decoded_sents = []
        for i, fre_sent in enumerate(test_sents):
            dec_sent = decode(fre_sent, LM, AM)
            decoded_sents.append(dec_sent)
            # print(ref_sents["hansard"][i])
            # print(ref_sents["google"][i])
            # print(dec_sent)
            # print("\n")

        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write("BLEU scores with N-gram (n) = %d: "%n)
            evals = _get_BLEU_scores(decoded_sents,ref_sents["hansard"], ref_sents["google"], n)
            #print(evals)
            for v in evals:
                f.write("\t%1.4f"%v)
            all_evals.append(evals)
            f.write("\n")
        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)