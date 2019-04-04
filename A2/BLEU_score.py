import math
from lm_train import *
from align_ibm1 import *
from decode import *

def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
    bpc = 1

    candi_arr = candidate.split()
    refe_arr = []
    for refe in references:
        refe_arr.append(refe.split())

    if brevity == True:
        # find nearest len
        len_diff = math.inf
        near_refe_ind = 0

        for i,refe in enumerate(refe_arr):
            if math.fabs(len(candi_arr) - len(refe)) < len_diff:
                len_diff = math.fabs(len(candi_arr) - len(refe))
                near_refe_ind = i

        len_divide = len(refe_arr[near_refe_ind])/len(candi_arr)
        #print(len_divide)
        if len_divide >= 1:
            bpc = math.exp(1-len_divide)

        #print(bpc)

    #calculate pn
    pn = 0
    if n == 1:
        unigram_count = len(candi_arr)
        numerator = 0
        for cand_word in candi_arr:
            for refe in refe_arr:
                if cand_word in refe:
                    numerator += 1
                    break
        pn = numerator / unigram_count
    elif n == 2:
        bigram_count = len(candi_arr) - 1
        numerator = 0
        for i,cand_word in enumerate(candi_arr):
            if i < len(candi_arr) - 1:
                bigram = cand_word + " " + candi_arr[i+1]
                #print(bigram)
                for refe in references:
                    if bigram in refe:
                        numerator += 1
                        break
        pn = numerator / bigram_count

    elif n == 3:
        trigram_count = len(candi_arr) - 2
        numerator = 0
        for i, cand_word in enumerate(candi_arr):
            if i < len(candi_arr) - 2:
                trigram = cand_word + " " + candi_arr[i + 1] + " " + candi_arr[i+2]
                #print(trigram)
                for refe in references:
                    if trigram in refe:
                        numerator += 1
                        break
        pn = numerator / trigram_count

    # bpc = 1 if brevity == false
    bleu_score = bpc * pn
    return bleu_score

# if __name__ == "__main__":
#     candi = "I am fear David"
#     refe = ["I am afraid Dave", "I am scared Dave" ,"I have fear David"]
#     #print(BLEU_score(candi, refe, 1, True) * BLEU_score(candi, refe, 2, False) ** (1/2))
#     print(BLEU_score(candi, refe, 1, True))
#     print(BLEU_score(candi, refe, 2, False))
#     print(BLEU_score(candi, refe, 3, False))

