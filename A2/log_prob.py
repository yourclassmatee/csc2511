from preprocess import *
from lm_train import *
from math import log2

def log_prob(sentence, LM, smoothing=False, delta=0.0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
    log_prob_result = 0
    words = sentence.split(" ")
    words_no_space = []
    for word in words:
        if word.strip() != "":
            words_no_space.append(word)

    for i, word in enumerate(words_no_space):
        if i< len(words_no_space)-1:
            unigram_count = LM['uni'].get(word,0)
            bigram_count = 0
            if LM['bi'].get(word) is not None:
                bigram_count = LM['bi'][word].get(words_no_space[i+1],0)
            #print("bigram %d unigram %d"%(bigram_count, unigram_count))
            condi_prob = float('-inf')
            if smoothing == True and delta != 0 and vocabSize != 0:
                condi_prob = log2((bigram_count + delta)/ (unigram_count + delta*vocabSize))
            else:
                if unigram_count != 0 and bigram_count != 0:
                    condi_prob = log2(bigram_count/unigram_count)
            #print("prob %f"%condi_prob)
            log_prob_result += condi_prob

    return log_prob_result




# if __name__ == "__main__":
#     dir = "/u/cs401/A2_SMT/data/Toy"
#     lan = 'e'
#     fn = "output_e"
#     lm = lm_train(dir, lan, fn)
#     toy_file = open("/u/cs401/A2_SMT/data/Toy/toy.e", 'r')
#     toy_file = toy_file.read().split("\n")
#
#     print(len(lm["uni"]))
#     print(lm)
#
#     for line in toy_file:
#         print(preprocess(line, 'e'))
#         print(log_prob(preprocess(line, 'e'),lm,smoothing=True, delta=0.0001, vocabSize=len(lm["uni"])))
#         break