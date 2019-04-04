import re


def split_mult_punc(mat):
    #print(mat.group(1), mat.group(2))
    mult_punc = ""
    mat_len = len(mat.group(2))
    mult_punc = mat.group(1) + " "
    for i in range(mat_len):
        mult_punc += mat.group(2)[i] + " "
    return mult_punc

def split_mult_punc_1(mat):
    # 1 punc 2 word
    #print(mat.group(1), mat.group(2))
    mult_punc = ""
    mat_len = len(mat.group(1))
    for i in range(mat_len):
        mult_punc += mat.group(1)[i] + " "
    mult_punc += mat.group(2)
    return mult_punc

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function

    #handle the newline at the eof
    if in_sentence == "":
        return ""

    out_sentence = ""

    #sperate punctuations sentence-final , : ; () - +-<>= ""
    s_before = in_sentence.split()
    s_after = []
    s_final = []
    for w_before in s_before:
        w_before = w_before.lower()
        #w_after = re.sub(r"([\w|%|$]+)([\"\.\?\!\,\:\;\)\(\+\-\<\>\=\"]+)", split_mult_punc, w_before)
        w_after = re.sub(r"([^\.\?\!\,\:\;\(\)\+\-\<\>\=\"\s]+)([\.\?\!\,\:\;\(\)\+\-\<\>\=\"]+)", split_mult_punc, w_before)
        s_after.append(w_after)

    for w_after in s_after:
        w_after = re.sub(r"([\.\?\!\,\:\;\(\)\+\-\<\>\=\"]+)([^\.\?\!\,\:\;\(\)\+\-\<\>\=\"\s]+)", split_mult_punc_1, w_after)
        s_final.append(w_after)

    s_final.insert(0, "SENTSTART")
    s_final.append("SENTEND")

    for w in s_final:
        out_sentence += (w.strip() + " ")


    if language == 'e':
        return out_sentence.strip()

    #french specific
    out_sentence = re.sub(r"(l'|c'|j'|m'|n'|t'|qu'|s'|lorsqu'|puisqu')(\w+)", lambda mat: mat.group(1) + " " + mat.group(2), out_sentence)
    #out_fix_space = re.sub(r"(L'|C'|J'|M'|N'|T'|Qu'|s'|Lorsqu'|Puisqu')(\w+)", lambda mat: mat.group(1) + " " + mat.group(2), out_fix_space)

    #d' except d'abord, d'accord, d'ailleurs, d'habitude
    #out_sentence = re.sub(r"(d')(?!(accord|ailleurs|habitude|abord))(\w+)", lambda mat: mat.group(1) + " " + mat.group(3), out_sentence)
    #out_sentence = re.sub(r"(D')(?!(accord|ailleurs|habitude|abord))(\w+)", lambda mat: mat.group(1) + " " + mat.group(3), out_sentence)

    return out_sentence.strip()

# if __name__ == "__main__":
#     english_path = "/u/cs401/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.192.e"
#     french_path = "/u/cs401/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.192.f"
#     french_file = open(french_path, 'r')
#     french_file = french_file.read().split('\n')
#     english_file = open(english_path, 'r')
#     english_file = english_file.read().split('\n')
#
#     # for line in english_file:
#     #     print(line)
#     #     print(preprocess(line, 'e'))
#     # for line in french_file:
#     #     print(line)
#     #     print(preprocess(line, 'f'))
#
#     print(preprocess("('La... seance est levee a 19 h aaa ).", "f"))