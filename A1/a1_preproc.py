import sys
import argparse
import os
import json
import html
import re
import string
import spacy

indir_1000532916 = '/u/cs401/A1/data/'

abbrev_path_1000532916 = "/u/cs401/Wordlists/abbrev.english"
abbrev_1000532916 = open(abbrev_path_1000532916, 'r')
abbrev_1000532916 = abbrev_1000532916.read().split('\n')

stopword_path_1000532916 = "/u/cs401/Wordlists/StopWords"
stopwords_1000532916 = open(stopword_path_1000532916, 'r')
stopwords_1000532916 = stopwords_1000532916.read().split("\n")

nlp = spacy.load("en", disable=["parser", "ner"])


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    #comment = "and/CC course/NN mussolini/NNP literally/RB socialist/NN fail/VBD rejigg/VBD favor/NN church/NN ./. ./. call/VBD fascism/NNP ./."
    modComm = ""
    if 1 in steps:
        #remove new line
        #print(modComm)
        modComm = comment.replace("\n", " ")
    if 2 in steps:
        #replace html chars
        modComm = html.unescape(modComm)
        modComm_b = modComm.encode(encoding="ascii", errors="ignore")
        modComm = modComm_b.decode(encoding="ascii")
    if 3 in steps:
        #remove url
        modComm = re.sub(r'(www\.\S+|https?:\/\/\S+)', "", modComm)
    if 4 in steps:
        c_before = modComm.split(" ")
        c_after = []
        for word in c_before:
            # handle e.g.[some consecutive puncuations]
            if re.search(r"\be.g.", word) or word in abbrev_1000532916:
                w_after = re.sub(r"\b(e.g.)(.*)", lambda mat: mat.group(1) + " " + mat.group(2), word)
                c_after.append(w_after)
                continue
            if word in abbrev_path_1000532916:
                c_after.append(word)
            # handle $100,00 case (not necessary
            # if re.search(r"\$\d+[.|,]\d+", word):
            #     w_after = re.sub(r"(\$)(\d+[.|,]\d+)",lambda mat : mat.group(1) + " " + mat.group(2), word)
            #     c_after.append(w_after)
            #     continue
            #match after
            w_after = re.sub(r"\w+(?=[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]+)", lambda mat : mat.group(0)+" ", word)
            # match before
            w_after = re.sub(r"(?<=[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~])\w+", lambda mat : " "+mat.group(0), w_after)
            c_after.append(w_after)
        modComm = ""
        for w in c_after:
            modComm += (w + " ")
        #print(modComm)

    if 5 in steps:
        # 's 're 'm 've 'd 'll 'all n't 'n
        modComm = re.sub(r"(\w+)('s|n't|'re|'m|'ve|'d|'ll|'all|'n)", lambda mat: mat.group(1) + " " + mat.group(2), modComm)
        modComm = re.sub(r"(\w+)('S|N'T|'RE|'M|'VE|'D|'LL|'ALL)", lambda mat: mat.group(1) + " " + mat.group(2), modComm)
        #s'
        modComm = re.sub(r"(\w+s)('\s)", lambda mat: mat.group(1) + " " + mat.group(2), modComm)
        modComm = re.sub(r"(\w+S)('\s)", lambda mat: mat.group(1) + " " + mat.group(2), modComm)
        #t'
        modComm = re.sub(r"(t)('\w+)", lambda mat: mat.group(1) + " " + mat.group(2), modComm)
        #print(modComm)

    if 6 in steps:
        #use entire string instead of tokens
        utt = nlp(modComm)
        modComm = ""
        for token in utt:
            modComm += token.text + "/" + token.tag_ + " "

    if 7 in steps:
        #remove stop words
        c_before = modComm.split(" ")
        modComm = ""
        for word in c_before:
            if word.split("/")[0] in stopwords_1000532916:
                continue
            modComm += word + " "

    if 8 in steps:
        #print('TODO')
        word_tag_tuples = re.findall(r"(\S+)/(\S+)", modComm)
        words = []
        tags = []
        for word_tag in word_tag_tuples:
            words.append(word_tag[0])
            tags.append(word_tag[1])
        #concat tokens back into sentence
        words_s = ' '.join(str(e) for e in words)
        utt = nlp(words_s)

        modComm = ""
        for i,token in enumerate(utt):
            if token.lemma_[0] == '-' and tags[i][0] != '-' or token.lemma_ == words[i].lower():
                modComm += words[i] + "/" + tags[i] + " "
                # if word and lemma are equal, keep lemma
            else:
                modComm += token.lemma_ + "/" + tags[i] + " "

    if 9 in steps:
        # add newline after ! or ?
        modComm = re.sub(r"([\?|\!]\/\.\s)(?=[A-Za-z])", lambda mat: mat.group(1) + "\n", modComm)
        # add newline after . if previous word is not in abbrev
        pat = re.compile(r"\s\.\/\.")
        to_insert = []
        for mat in pat.finditer(modComm):
            cur_index = mat.start()-1
            #find prev word
            while modComm[cur_index] != ' ' and cur_index >= 0:
                cur_index -= 1

            if cur_index < 0:
                prev_word = modComm[0:mat.start()]
            else:
                prev_word = modComm[cur_index:mat.start()]
            #print(prev_word)
            if prev_word.strip().split('/')[0]+"." not in abbrev_1000532916:
                to_insert.append(mat.span()[1])
        #print(to_insert)
        for i,insert_idx in enumerate(to_insert):
            if insert_idx+i <= len(modComm)-1:
                modComm = modComm[:insert_idx+i] + "\n" + modComm[insert_idx+i:]
        #print(modComm)

    if 10 in steps:
        modComm = re.sub(r"\S+(?=/)", lambda mat: mat.group(0).lower(), modComm)
        #print(modComm)
        
    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir_1000532916):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            student_id = args.ID[0]
            out_file = args.output
            st = student_id % len(data)
            end = st + args.max

            # start looping
            i = st
            count = 0
            while count < args.max:
                line = json.loads(data[i])
                out_line = {}
                out_line["id"] = line["id"]
                out_line["body"] = line["body"]
                out_line["cat"] = file
                out_line["body"] = preproc1(out_line["body"])
                allOutput.append(out_line)
                count += 1
                i += 1
                if i >= end:
                    i = 0
            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
