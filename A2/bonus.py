from preprocess import *
from lm_train import *
from math import log2


def get_total_uni(uni_dict):
    values = 0
    for key, value in uni_dict.items():
        values += value
    return values


def get_total_bi(bi_dict):
    values = 0
    for key, value_dict in bi_dict.items():
        for word, value in value_dict.items():
            values += value
    return values


def count_occur_uni(uni_dict, count):
    occur = 0
    for key, value in uni_dict.items():
        if value == count:
            occur += 1
    return occur


def count_occur_bi(bi_dict, count):
    occur = 0
    for word, value_dict in bi_dict.items():
        for next_word, value in value_dict.items():
            if value == count:
                occur += 1
    return occur


def log_prob_good_turing(sentence, LM, smoothing=False):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
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
        if i < len(words_no_space) - 1:
            unigram_count = LM['uni'].get(word, 0)
            bigram_count = 0
            if LM['bi'].get(word) is not None:
                bigram_count = LM['bi'][word].get(words_no_space[i + 1], 0)
            # print("bigram %d unigram %d"%(bigram_count, unigram_count))
            condi_prob = float('-inf')
            if smoothing == True:
                unigram_prob = 0
                bigram_prob = 0
                total_uni = get_total_uni(LM['uni'])
                total_bi = get_total_bi(LM['bi'])
                # unseen uni => N1/N
                if unigram_count == 0:
                    N1 = count_occur_uni(LM['uni'], 1)
                    unigram_prob = N1/total_uni
                else:
                    N1 = count_occur_uni(LM['uni'], unigram_count)
                    N2 = count_occur_uni(LM['uni'], unigram_count+1)
                    unigram_prob = (unigram_count+1)*N2/(N1*total_uni)

                if bigram_count == 0:
                    N1 = count_occur_bi(LM['bi'], 1)
                    bigram_prob = N1/total_bi
                else:
                    N1 = count_occur_bi(LM['bi'], bigram_count)
                    N2 = count_occur_bi(LM['bi'], bigram_count+1)
                    bigram_prob = (bigram_count + 1)*N2/(N1*total_bi)
                if bigram_prob != 0 and unigram_prob != 0:
                    condi_prob = log2(bigram_prob / unigram_prob)
                elif unigram_count != 0 and bigram_count != 0:
                    condi_prob = log2(bigram_count/unigram_count)

            else:
                if unigram_count != 0 and bigram_count != 0:
                    condi_prob = log2(bigram_count / unigram_count)
            # print("prob %f"%condi_prob)
            log_prob_result += condi_prob

    return log_prob_result

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
    test_dir = "/h/u10/c6/00/shengmin/Desktop/A2/test/"
    #test_dir = "/u/cs401/A2_SMT/data/Hansard/Testing/"

    print(preplexity_turing(test_LM, test_dir, "e", True))

    #print(preplexity(test_LM, test_dir, "e"))