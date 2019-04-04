from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
	
	# TODO: Implement Function
    language_model = dict()
    language_model['uni'] = dict()
    language_model['bi'] = dict()
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            #skip files from the other language
            if file.split('.')[-1] != language:
                continue
            #print(file)
            fullFile = os.path.join(subdir, file)
            actual_file = open(fullFile, 'r')
            for line in actual_file.read().split('\n'):
                if line.strip() == "":
                    continue

                proc_line = preprocess(line, language)
                # get rid of " "
                words = proc_line.split()
                words_no_space = []
                for word in words:
                    if word.strip() != "":
                        words_no_space.append(word)
                # do uni
                for word in words_no_space:
                    if language_model['uni'].get(word) == None:
                        language_model['uni'][word] = 1
                    else:
                        language_model['uni'][word] += 1
                #do bi
                for i,word in enumerate(words_no_space):
                    # first key not exist
                    #print(word)
                    if i < len(words_no_space)-1:
                        if language_model['bi'].get(word) == None:
                            language_model['bi'][word] = dict()
                            language_model['bi'][word][words_no_space[i + 1]] = 1
                        elif language_model['bi'][word].get(words_no_space[i+1]) == None:
                            language_model['bi'][word][words_no_space[i+1]] = 1
                        else:
                            language_model['bi'][word][words_no_space[i+1]] += 1



    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return language_model

# if __name__ == "__main__":
#     #dir = "/u/cs401/A2_SMT/data/Toy"
#     dir = "/h/u10/c6/00/shengmin/Desktop/A2/train"
#     lan = 'f'
#     fn = "output_e"
#     lm = lm_train(dir, lan, fn)
#     print(lm)