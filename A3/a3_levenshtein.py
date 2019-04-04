import os
import numpy as np
import math
import re

dataDir = '/u/cs401/A3/data/'


def preproc(sent):
    # remove first 2
    sent_arr = sent.split()
    sent_arr = sent_arr[2:]
    sent_no_first = (' ').join(sent_arr)

    # remove <>
    no_punc = re.sub(r'\<\S*?\>', '', sent_no_first)

    # remove punc
    no_punc =re.sub(r'[!\"#$%&()*+,-./:;<=>?@^_`{|}~\\]+', '', no_punc)

    return no_punc.lower()


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    # >>> wer("who is there".split(), "is there".split())
    # 0.333 0 0 1
    # >>> wer("who is there".split(), "".split())
    # 1.0 0 0 3
    # >>> wer("".split(), "who is there".split())
    # Inf 0 3 0
    # """

    #up=0 left=1 up-left=2
    #init
    n = len(r)
    m = len(h)
    R = np.zeros((n+1, m+1))
    B = np.zeros((n+1, m+1))

    for i in range(0, n+1):
        R[i,0] = i
        B[i,0] = 0
    for i in range(0, m+1):
        R[0,i] = i
        B[0,i] = 1

    B[0,0] = -1

    for i in range(1, n+1):
        for j in range(1, m+1):
            dele = R[i-1, j] + 1
            sub = R[i-1, j-1]
            if (r[i-1] != h[j-1]):
                sub += 1
            ins = R[i,j-1] +1
            three = [dele, ins, sub]
            R[i,j] = min(three)
            index_min = three.index(min(three))
            if index_min == 0: #up
                B[i,j] = 0
            elif index_min == 1: #left
                B[i,j] = 1
            else:               #up-left
                B[i,j] = 2

    #counting
    i=n
    j=m
    num_sub = 0
    num_ins = 0
    num_dele = 0
    while B[i,j] != -1 and i>=0 and j>=0:
        if B[i,j] == 0:
            num_dele += 1
            i = i-1 #up
        elif B[i,j] == 1:
            num_ins += 1
            j= j-1 #left
        elif B[i,j] == 2:
            if  R[i,j] == R[i-1,j-1] + 1:
                num_sub += 1
            j = j - 1
            i = i - 1  #up-left

    wer = float("inf")
    if n != 0:
        wer = (float(num_sub) + float(num_ins) + float(num_dele))/float(n)
    return (wer, num_sub, num_ins, num_dele)



if __name__ == "__main__":
    google_result = []
    kaldi_result = []
    out = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:

            print("processing speaker %s"%speaker)

            transcript = os.path.join(dataDir, speaker, 'transcripts.txt')
            transcript_google = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
            transcript_kaldi = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')

            transcript = open(transcript).readlines()
            transcript_google = open(transcript_google).readlines()
            transcript_kaldi = open(transcript_kaldi).readlines()

            for i,trans_line in enumerate(transcript):
                trans_line_preproc = preproc(trans_line)
                google_line_preproc = preproc(transcript_google[i])
                kaldi_line_preproc = preproc(transcript_kaldi[i])

                google = Levenshtein(trans_line_preproc.split(), google_line_preproc.split())
                google_result.append(google[0])
                kaldi = Levenshtein(trans_line_preproc.split(), kaldi_line_preproc.split())
                kaldi_result.append(kaldi[0])

                out.append("[%s] [%s] [%d] [%f] S:[%d] I:[%d] D:[%d]\n" % \
                           (speaker, "Kaldi", i, kaldi[0], kaldi[1], kaldi[2], kaldi[3]))
                out.append("[%s] [%s] [%d] [%f] S:[%d] I:[%d] D:[%d]\n"% \
                           (speaker, "Google", i, google[0], google[1], google[2], google[3]))


    google_mean = np.mean(google_result)
    google_std = np.std(google_result)
    kaldi_mean = np.mean(kaldi_result)
    kaldi_std = np.std(kaldi_result)

    fout = open("asrDiscussion.txt", 'w')
    for line in out:
        fout.write(line)

    fout.write("Google mean %f std %f, Kaldi mean %f std %f"%(google_mean, google_std, kaldi_mean, kaldi_std))
    fout.close()







