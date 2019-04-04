from log_prob import *
from preprocess import *
from bonus import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    print(len(LM["uni"]))

    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue

        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)

            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp


def preplexity_turing(LM, test_dir, language, smoothing=False):
    """
	Computes the preplexity of language model given a test corpus

	INPUT:

	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    print(len(LM["uni"]))

    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue

        opened_file = open(test_dir + ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob_good_turing(processed_line, LM, smoothing)

            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2 ** (-pp / N)
    return pp

#test
if __name__ == "__main__":

    test_LM = lm_train("/u/cs401/A2_SMT/data/Hansard/Training/", "e", "e_temp")
    #test_LM = lm_train("/h/u10/c6/00/shengmin/Desktop/A2/train/", "e", "e_temp")
    #test_dir = "/h/u10/c6/00/shengmin/Desktop/A2/test/"
    test_dir = "/u/cs401/A2_SMT/data/Hansard/Testing/"
    # print(preplexity(test_LM, test_dir, "f"))
    # print(preplexity(test_LM, test_dir, "f", smoothing=True, delta=0.01))
    # print(preplexity(test_LM, test_dir, "f", smoothing=True, delta=0.1))
    # print(preplexity(test_LM, test_dir, "f", smoothing=True, delta=0.5))
    # print(preplexity(test_LM, test_dir, "f", smoothing=True, delta=0.8))

    print(preplexity_turing(test_LM, test_dir, "e", True))

    #print(preplexity(test_LM, test_dir, "e"))