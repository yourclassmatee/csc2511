from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    eng_sents, fre_sents = read_hansard(train_dir, num_sentences)
    # print(eng_sents)
    # print(fre_sents)
    # Initialize AM uniformly
    AM = initialize(eng_sents, fre_sents)
    #print("===========init==========")
    #print(AM)
    # Iterate between E and M steps
    for i in range(max_iter):
        AM = em_step(AM, eng_sents, fre_sents)
        # print("=======================iteration %d"%i)
        # print(AM)

    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    eng_files = []
    fre_files = []
    eng = []
    fre = []
    for subdir, dirs, files in os.walk(train_dir):
        #match e with f
        for file in files:
            file_name_split = file.split('.')
            file_name_ef = file_name_split[-1]
            #skip answerkey from toy
            if file_name_ef != 'e' and file_name_ef != 'f':
                continue
            #print(file_name_split[0:len(file_name_split)-1])
            file_name_pre =  ".".join(str(x) for x in file_name_split[0:len(file_name_split)-1])
            if file_name_ef == 'e':
                #eng_file_dict[file] = file_name_pre + ".f"
                eng_files.append(file)
                fre_files.append(file_name_pre + ".f")
    #print(eng_file_dict)
    sent_count = 0

    for i, eng_file in enumerate(eng_files):

        fre_file = fre_files[i]
        if sent_count >= num_sentences:
            break

        #print(eng_file)
        #print(fre_file)
        full_eng = os.path.join(train_dir, eng_file)
        lines_eng = open(full_eng, 'r').read().split('\n')
        full_fre = os.path.join(train_dir, fre_file)
        lines_fre = open(full_fre, 'r').read().split('\n')
        for i, line in enumerate(lines_eng):
            if sent_count >= num_sentences:
                break
            if line.strip() == "":
                continue
            # turn preproc sent to list of words
            eng_sent = preprocess(line,'e')
            eng_arr = []
            for word in eng_sent.split():
                if word.strip() != "":
                    eng_arr.append(word)
            eng.append(eng_arr)
            # turn preproc sent to list of words
            fre_sent = preprocess(lines_fre[i],'f')
            fre_arr = []
            for word in fre_sent.split():
                if word.strip() != "":
                    fre_arr.append(word)
            fre.append(fre_arr)

            sent_count += 1

    #print(len(eng), len(fre))
    return eng,fre



def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    AM = dict()
    AM['SENTSTART']=dict()
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND'] = dict()
    AM['SENTEND']['SENTEND'] = 1

    used_eng = ["SENTSTART", "SENTEND"]
    for sent in eng:
        for eng_word in sent:
            if eng_word not in used_eng:
                #print(eng_word)
                potential_fre = []
                for i,other_sent in enumerate(eng):
                    if eng_word in other_sent:
                        french_words = fre[i]
                        for fre_word in french_words:
                            if fre_word != "SENTSTART" and fre_word !="SENTEND" and fre_word.strip()!=""\
                                    and fre_word not in potential_fre:
                                potential_fre.append(fre_word)
                AM[eng_word] = dict()
                for potential_fre_word in potential_fre:
                    #print(eng_word, potential_fre_word)
                    AM[eng_word][potential_fre_word] = 1/len(potential_fre)
                used_eng.append(eng_word)

    return AM


def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	# set tcount(f,e) = 0
    tcount = dict()
    for eng_word, value in t.items():
        for fre_word, prob in value.items():
            if tcount.get(eng_word) is None:
                tcount[eng_word] = dict()
            tcount[eng_word][fre_word] = 0
    #print(tcount)

    #set total(e) = 0
    total = dict()
    for eng_word in t.keys():
        total[eng_word] = 0
    #print(total)
    #print("===================now loop=======================")

    for i,eng_sent in enumerate(eng):
        fre_sent = fre[i]
        unique_fre_words = get_unique_words(fre_sent)
        #print(unique_fre_words)

        for unique_fre in unique_fre_words:
            fre_words_occurance = count_word_instances(fre_sent, unique_fre)
            # print(fre_words_occurance)
            denom_c = 0
            unique_eng_words = get_unique_words(eng_sent)
            #print(unique_eng_words)

            for unique_eng in unique_eng_words:
                #print(unique_eng, unique_fre)
                eng_words_occurance = count_word_instances(eng_sent, unique_eng)
                # print(eng_words_occurance)
                if t[unique_eng].get(unique_fre) is not None:
                    #print(unique_eng, unique_fre, t[unique_eng][unique_fre])
                    denom_c += t[unique_eng][unique_fre] * fre_words_occurance

            for unique_eng in unique_eng_words:
                if t[unique_eng].get(unique_fre) is not None and denom_c != 0:
                    tcount[unique_eng][unique_fre] += t[unique_eng][unique_fre] * fre_words_occurance * eng_words_occurance / denom_c
                    total[unique_eng] += t[unique_eng][unique_fre] * fre_words_occurance * eng_words_occurance / denom_c

    for eng_word, value in t.items():
        for fre_word, t_count in value.items():
            #print(eng_word, fre_word)
            if total[eng_word] != 0:
                #print("tcount %d total %d"%(tcount[eng_word][fre_word], total[eng_word]))
                t[eng_word][fre_word] = tcount[eng_word][fre_word]/total[eng_word]
    return t


def get_unique_words(sent):
    # don't include sentstart sentend in training
    unique_words = ["SENTSTART", "SENTEND"]
    for word in sent:
        if word not in unique_words:
            unique_words.append(word)
    return unique_words


def count_word_instances(sent, unique_word):
    count = 0
    for word in sent:
        if word == unique_word:
            count += 1
    return count


#
# if __name__ == "__main__":
#     train_dir = "/u/cs401/A2_SMT/data/Toy"
#
#     AM = align_ibm1(train_dir, 4, 3, "am_save")
#
