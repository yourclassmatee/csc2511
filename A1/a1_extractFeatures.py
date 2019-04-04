import numpy as np
import sys
import argparse
import os
import json
import re
import csv
import math
import statistics



def load_bgl_1000532916 (bgl_path):
    bgl_dict = dict()
    with open(bgl_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            bgl_dict[row[1]] = (row[3], row[4], row[5])
    return bgl_dict

def load_war_100532916 (war_path):
    war_dict = dict()
    with open(war_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            war_dict[row[1]] = (row[2], row[5], row[8])
    return war_dict

bgl_path_1000532916 = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
war_path_1000532916 = "/u/cs401/Wordlists/Ratings_Warriner_et_al.csv"
feats_path_1000532916 = "/u/cs401/A1/feats/"
bgl_dict_1000532916 = load_bgl_1000532916(bgl_path_1000532916)
war_dict_1000532916 = load_war_100532916(war_path_1000532916)






slangs_1000532916 = ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb',
          'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr',
          'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn', 'bbs', 'cya', 'ez', 'f2f','gtr', 'ic', 'jk',
          'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml']

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    #print('TODO')

    #comment = "smh/VBD to/VBZ 'll/NN fwb/WP$ ,!./, (/-LRB- 's/VBZ fb/VBD )/-RRB- ./."
    feats = np.zeros(173)

    #print(comment)
    # TODO: your code here

    #Number of first - person pronouns
    first_person = re.findall(r"\b(i|me|my|mine|we|us|our|ours)\b", comment)
    feats[0] = len(first_person)
    #print(feats[0])

    #Number of second-person pronouns
    second_person = re.findall(r"\b(you|your|yours|u|ur|urs)\b", comment)
    feats[1] = len(second_person)

    #Number of third - person pronouns
    third_person = re.findall(r"\b(he|him|his|she|her|hers|it|its|they|them|their|theirs)\b", comment)
    feats[2] = len(third_person)

    #Number of coordinating conjunctions
    cc = re.findall(r"/CC\b", comment)
    feats[3] = len(cc)
    #print(feats[3])

    #Number of past - tense verbs
    vbd = re.findall(r"/VBD\b", comment)
    feats[4] = len(vbd)
    #print(feats[4])

    #Number of future-tense verbs
    future_tense = re.findall(r"('ll|will|gonna)/", comment)
    feats[5] = len(future_tense)
    fu2 = re.findall(r"go/\w+\sto/", comment)
    feats[5] += len(fu2)
    #print(feats[5])

    #Number of commas
    comma = re.findall(r",/,", comment)
    feats[6] = len(comma)
    #print(feats[6])

    #Number of multi-character punctuation tokens
    multi = re.findall(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]{2}/", comment)
    feats[7] = len(multi)
    #print(feats[7])

    #Number of common nouns
    nn = re.findall(r"/(NN|NNS)\b", comment)
    feats[8] = len(nn)
    #print(feats[8])

    #Number of proper nouns
    nnp = re.findall(r"/(NNP|NNPS)\b", comment)
    feats[9] = len(nnp)
    #print(feats[9])

    #Number of adverbs
    rb = re.findall(r"/RB\b", comment)
    feats[10] = len(rb)
    #print(feats[10])

    #Number of wh- words
    wh = re.findall(r"/(WDT|WP|WP$|WRB)\b", comment)
    feats[11] = len(wh)
    #print(feats[11])

    #Number of slang acronyms
    sl_count = 0
    words = re.findall(r"(?<=\b)\w+(?=/)", comment)
    for word in words:
        if word in slangs_1000532916:
            sl_count+=1
    feats[12] = sl_count
    #print(feats[12])

    #Number of words in uppercase (≥ 3 letters long)
    #impossible to have uppercase since step 10 in preprocessing change all text to lower
    upper_case_words = re.findall(r"(?<=\b)[A-Z][A-Z][A-Z]+(?=/)", comment)
    if upper_case_words is not None:
        feats[13] = len(upper_case_words)
    #print(feats[13])

    #Average length of sentences, in tokens
    sentences = comment.split("\n")
    words = re.findall(r"[\S]+",comment)
    feats[14] = len(words)/len(sentences)

    #Average length of tokens, excluding punctuation-only tokens, in characters
    token_len = 0
    words = re.findall(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~']*\w+[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~']*(?=/)", comment)
    if len(words) != 0:
        for word in words:
            token_len += len(word)
        feats[15] = token_len/len(words)

    #Number of sentences.
    sentences = comment.split("\n")
    feats[16] = len(sentences)
    #print(feats[16])

    #Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    #Average of IMG from Bristol, Gilhooly, and Logie norms
    #Average of FAM from Bristol, Gilhooly, and Logie norms
    #Standard deviation of AoA (100-700) from Bristol, Gilhooly
    #Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    #Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    words = re.findall(r"(?<=\b)\w+(?=/)", comment)
    aoa = []
    img = []
    fam = []
    for word in words:
        if word in bgl_dict_1000532916.keys():
            aoa.append(int(bgl_dict_1000532916[word][0]))
            img.append(int(bgl_dict_1000532916[word][1]))
            fam.append(int(bgl_dict_1000532916[word][2]))
    if len(aoa) != 0:
        feats[17] = statistics.mean(aoa)
    if len(img) != 0:
        feats[18] = statistics.mean(img)
    if len(fam) != 0:
        feats[19] = statistics.mean(fam)
    if len(aoa) >= 2:
        feats[20] = statistics.pstdev(aoa)
    if len(img) >= 2:
        feats[21] = statistics.pstdev(img)
    if len(fam) >= 2:
        feats[22] = statistics.pstdev(fam)
    #Average of V.Mean.Sum from Warringer norms
    #Average of A.Mean.Sum from Warringer norms
    #Average of D.Mean.Sum from Warringer norms
    #Standard deviation of V.Mean.Sum from Warringer norms
    #Standard deviation of A.Mean.Sum from Warringer norms
    #Standard deviation of D.Mean.Sum from Warringer norms
    words = re.findall(r"(?<=\b)\w+(?=/)", comment)
    vmean = []
    amean = []
    dmean = []
    for word in words:
        if word in war_dict_1000532916.keys():
            vmean.append(float(war_dict_1000532916[word][0]))
            amean.append(float(war_dict_1000532916[word][1]))
            dmean.append(float(war_dict_1000532916[word][2]))

    if len(vmean) != 0:
        feats[23] = statistics.mean(vmean)
    if len(amean) != 0:
        feats[24] = statistics.mean(amean)
    if len(dmean) != 0:
        feats[25] = statistics.mean(dmean)
    if len(vmean) >= 2:
        feats[26] = statistics.pstdev(vmean)
    if len(amean) >= 2:
        feats[27] = statistics.pstdev(amean)
    if len(dmean) >= 2:
        feats[28] = statistics.pstdev(dmean)

    #print(feats)
    return feats

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    #load 144 features

    cats = ["Left", "Center", "Right", "Alt"]

    feat_files = []
    feat_files.append(feats_path_1000532916 + "Left_feats.dat.npy")
    feat_files.append(feats_path_1000532916 + "Center_feats.dat.npy")
    feat_files.append(feats_path_1000532916 + "Right_feats.dat.npy")
    feat_files.append(feats_path_1000532916 + "Alt_feats.dat.npy")
    feats_array = []
    feats_array.append(np.load(feat_files[0]))
    feats_array.append(np.load(feat_files[1]))
    feats_array.append(np.load(feat_files[2]))
    feats_array.append(np.load(feat_files[3]))

    id_files = []
    id_files.append(feats_path_1000532916 + "Left_IDs.txt")
    id_files.append(feats_path_1000532916 + "Center_IDs.txt")
    id_files.append(feats_path_1000532916 + "Right_IDs.txt")
    id_files.append(feats_path_1000532916 + "Alt_IDs.txt")
    id_arrays = []
    id_arrays.append(open(id_files[0], 'r').read().split('\n'))
    id_arrays.append(open(id_files[1], 'r').read().split('\n'))
    id_arrays.append(open(id_files[2], 'r').read().split('\n'))
    id_arrays.append(open(id_files[3], 'r').read().split('\n'))


    # TODO: your code here
    for index, line in enumerate(data):
        id = line["id"]
        cat = line["cat"]
        feats[index][0:173] = extract1(line["body"])
        #find row number
        for cat_ind, c in enumerate(cats):
            if cat == c:
                row_num = id_arrays[cat_ind].index(id)
                feats[index][29:173] = feats_array[cat_ind][row_num]
                feats[index][173] = cat_ind
                #print(feats[index])
                break

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

