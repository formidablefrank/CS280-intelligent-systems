from decimal import Decimal
from math import log
import glob, re, time, os, sys
import cPickle as pickle


# Global vars
# initialize frequency dictionary for ham and spam {word: count}
frHam, frSpam, fileLabels = {}, {}, {}
nHam, nSpam, start_time = 0, 0, 0
pattern1 = r'([a-zA-Z]+)([.,]?)'
pattern2 = r'([a-zA-Z]+)'

class SpamFilter:
    def __init__(self, cHam, cSpam, frHam, frSpam, fileLabels, isLS):
        self.cHam = Decimal(cHam)
        self.cSpam = Decimal(cSpam)
        self.total = Decimal(self.cHam + self.cSpam)
        self.frHam = frHam
        self.frSpam = frSpam
        self.alphaDic = list(set(frHam.keys() + frSpam.keys()))
        self.infoDic = []
        self.iHam = {}
        self.iSpam = {}
        self.iOver = {}
        self.fileLabels = fileLabels
        self.count = [0, 0, 0, 0] #HtoH, HtoS, StoH, StoS
        self.isLS = 0
        print 'P(ham) =', self.cHam / self.total, '\nP(spam) =', self.cSpam / self.total
        self.extract_informative()
        #print 'SPAM FILTER BEGIN:\ncHam -> %s, cSpam -> %s\nfrHam -> %s\nfrSpam -> %s\nDic -> %s\n' % (self.cHam, self.cSpam, self.frHam, self.frSpam, self.alphaDic)

    def extract_informative(self):
        print 'Extracting informative words'
        pH = Decimal(self.cHam) / self.total
        pS = Decimal(self.cSpam) / self.total
        ctr = 0
        dot = 0
        for word in self.alphaDic:
            pWpH = self.frHam.get(word, 0) / Decimal(self.cHam)
            pWaH = 1 - pWpH
            pWpS = self.frSpam.get(word, 0) / Decimal(self.cSpam)
            pWaS = 1 - pWpS
            pWp = (self.frHam.get(word, 0) + self.frSpam.get(word, 0)) / Decimal(self.total)
            pWa = 1 - pWp
            self.iHam[word] = 0
            self.iSpam[word] = 0
            self.iOver[word] = 0
            # print word, pWpH, pWaH, pWpS, pWaS, pWp, pWa
            try:
                self.iHam[word] += pWpH * (pWpH / pWp / pH).ln()
            except Exception as e:
                pass
            try:
                self.iHam[word] += pWaH * (pWaH / pWa / pH).ln()
            except Exception as e:
                pass
            try:
                self.iSpam[word] += pWpS * (pWpS / pWp / pS).ln()
            except Exception as e:
                pass
            try:
                self.iSpam[word] += pWaS * (pWaS / pWa / pS).ln()
            except Exception as e:
                pass
            self.iOver[word] = self.iHam[word] + self.iSpam[word]
            if Decimal(ctr) / Decimal(len(self.alphaDic)) * Decimal(20.) > dot:
                sys.stdout.write('.')
                sys.stdout.flush()
                dot += 1
            ctr += 1
        print ''
        # get top 200 words for each class
        iHamDict = [i[0] for i in sorted(self.iHam.items(), key=lambda x: x[1], reverse=True)[0:200]]
        print 'Top 200 ham words:', iHamDict, '\n'
        iSpamDict = [i[0] for i in sorted(self.iSpam.items(), key=lambda x: x[1], reverse=True)[0:200]]
        print 'Top 200 spam words:', iSpamDict, '\n'
        # get top 200 words overall
        self.infoDic = [i[0] for i in sorted(self.iOver.items(), key=lambda x: x[1], reverse=True)[0:200]]
        print 'Top 200 overall words', self.infoDic, '\n'

    def classify(self, fname, wlist, mode):
        #print 'ANALYZING %s...' % filename
        probH, probS = 0.0, 0.0
        if mode == 'LS':
            vector = [word in wlist for word in self.alphaDic]
            probH = self.prob('ham', vector, self.alphaDic)
            probS = self.prob('spam', vector, self.alphaDic)
        elif mode == 'MI':
            self.isLS = 0
            vector = [word in wlist for word in self.infoDic]
            probH = self.prob('ham', vector, self.infoDic)
            probS = self.prob('spam', vector, self.infoDic)
        #print vector
        newCount = [int(probH > probS and self.fileLabels[fname] == 0), int(probH > probS and self.fileLabels[fname] != 0), int(probH <= probS and self.fileLabels[fname] == 0), int(probH <= probS and self.fileLabels[fname] != 0)]
        self.count = map(lambda x, y: x + y, self.count, newCount)

    def prob(self, typed, vector, src):
        pWiHam, pWiSpam, P = Decimal(1.), Decimal(1.), Decimal(1.)
        for ctr in range(len(vector)):
            if vector[ctr]:
                # print '%s - %f - %f' % (self.alphaDic[ctr], self.frHam.get(self.alphaDic[ctr], 0)/self.cHam, self.frSpam.get(self.alphaDic[ctr], 0)/self.cSpam)
                pWiHam *= Decimal(self.frHam.get(src[ctr], 0))/Decimal(self.cHam)
                pWiSpam *= Decimal(self.frSpam.get(src[ctr], 0))/Decimal(self.cSpam)
                if pWiHam == 0 or pWiSpam == 0:
                    break
        if typed == 'spam' and pWiSpam == 0 or typed == 'ham' and pWiHam == 0:
            #print ' - USED LAMBDA SMOOTHING'
            if self.isLS > 0:
                pWiHam, pWiSpam = Decimal(1.), Decimal(1.)
                for ctr in range(len(vector)):
                    if vector[ctr]:
                        #print '%s - %f - %f' % (self.alphaDic[ctr], self.frHam.get(self.alphaDic[ctr], 1.0) / (self.cHam + 2.0), self.frSpam.get(self.alphaDic[ctr], 1.0) / (self.cSpam + 2.0))
                        pWiHam *= Decimal(self.frHam.get(src[ctr], self.isLS)) / (self.cHam + Decimal(self.isLS) * len(src))
                        pWiSpam *= Decimal(self.frSpam.get(src[ctr], self.isLS)) / (self.cSpam + Decimal(self.isLS) * len(src))
            else:
                return 0
        # print 'pWiHam = %s\npWiSpam = %s' % (pWiHam, pWiSpam)
        evidence = (self.cHam / self.total * pWiHam + self.cSpam / self.total * pWiSpam)
        if typed == 'ham':
            P = (self.cHam / self.total * pWiHam) / evidence
            return P
        elif typed == 'spam':
            P = (self.cSpam / self.total * pWiSpam) / evidence
            return P

    def stats(self):
        tn = Decimal(self.count[0])
        fn = Decimal(self.count[1])
        fp = Decimal(self.count[2])
        tp = Decimal(self.count[3])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return [precision, recall, accuracy]

def train():
    global nHam, nSpam, frHam, frSpam, fileLabels, pattern1, pattern2, testFilter
    # labelFile = open('labels', 'r')
    # for labels in labelFile.readlines():
    #     fileName = labels.split()[1][3:]
    #     label = labels.split()[0]
    #     fileLabels[fileName] = 0 if label == 'ham' else 1
    # print fileLabels

    # get valid words from training files and add to frequency dictionary
    # folders 000 - 070
    # for folderNo in range(0, 71):
    #     print str(folderNo).zfill(3),
    #     trainFiles = glob.glob('data/' + str(folderNo).zfill(3) + '/*')
    #     ctr = 0
    #     dot = 0
    #     for trainFile in trainFiles:
    #         #print >> outputFile, 'reading %-7s' % trainFile
    #         #print 'file', trainFile
    #         label = fileLabels[trainFile]
    #         if label == 0:
    #             nHam += 1
    #         else:
    #             nSpam += 1
    #         f = open(trainFile, 'rb')
    #         fd = f.fileno()
    #         for word in list(set(os.read(fd, 1048576).split())):
    #             #print word
    #             try:
    #                 #satisfy assumption of a word given by pattern1
    #                 if (word == re.search(pattern1, word).group(0)):
    #                     #add trimmed word to respective freq dict
    #                     validWord = re.search(pattern2, word).group(0).lower()
    #                     if label == 0:
    #                         frHam[validWord] = frHam.get(validWord, 0) + 1
    #                     else:
    #                         frSpam[validWord] = frSpam.get(validWord, 0) + 1
    #                     #print validWord
    #             except Exception as e:
    #                 pass
    #         f.close()
    #         if Decimal(ctr) / Decimal(len(trainFiles)) * Decimal(20.) > dot:
    #             sys.stdout.write('.')
    #             sys.stdout.flush()
    #             dot += 1
    #         ctr += 1
    #     print ''
    #
    # inputs = [nHam, nSpam, frHam, frSpam, fileLabels]
    # pickle.dump(inputs, open('training.o', 'wb'))
    # initialize spam filter
    inputs = pickle.load(open('training.o', 'rb'))
    start_time = time.time()
    testFilter = SpamFilter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 0)
    print("--- training finished after %s seconds ---" % (time.time() - start_time))

    # serialize the classifier to save time
    pickle.dump(testFilter, open('filter.o', 'wb'))

if __name__ == '__main__':
    # try to make an output file
    try:
        outputFile = open('outputs.txt', 'w')
    except IOError:
        print 'Error creating output file'

    train()
    testing = {}
    # # get valid words from test data
    # for folderNo in range(71, 127):
    #     print str(folderNo).zfill(3),
    #     testFiles = glob.glob('data/' + str(folderNo).zfill(3) + '/*')
    #     ctr = 0
    #     dot = 0
    #     for testFile in testFiles:
    #         validWords = []
    #         f = open(testFile, 'rb')
    #         fd = f.fileno()
    #         for word in list(set(os.read(fd, 1048576).split())):
    #             # print word
    #             try:
    #                 if (word == re.search(pattern1, word).group(0)):
    #                     validWords.append(re.search(pattern2, word).group(0).lower())
    #             except Exception as e:
    #                 pass
    #         f.close()
    #         if Decimal(ctr) / Decimal(len(testFiles)) * Decimal(20.) > dot:
    #             sys.stdout.write('.')
    #             sys.stdout.flush()
    #             dot += 1
    #         ctr += 1
    #         testing[testFile] = validWords
    #     print ''
    #
    # pickle.dump(testing, open('testing.o', 'wb'))

    testFilter = pickle.load(open('model.o', 'rb'))
    testing = pickle.load(open('testing.o', 'rb'))
    print 'Classifying test data'
    start_time = time.time()
    # for LS in [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    #     print LS
    #     testFilter.isLS = LS
    #     for testFile in testing:
    #         testFilter.classify(testFile, testing[testFile], 'LS')
    #     print >> outputFile, LS, testFilter.stats()
    # reset stats before using MI
    testFilter.count = [0, 0, 0, 0]
    testFilter.isLS = 2.0
    for testFile in testing:
        testFilter.classify(testFile, testing[testFile], 'MI')
    print >> outputFile, 'MI', testFilter.stats()

    print("--- testing finished after %s seconds ---" % (time.time() - start_time))
