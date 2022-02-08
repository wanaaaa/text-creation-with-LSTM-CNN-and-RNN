import json
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim import corpora
from gensim.models import Word2Vec
import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)


w2vMyModel = Word2Vec.load('./w2vTrainedModel/w2v.10th.10vec.model')

def splitToTrainTestFun(wordSentenceDbLi):
    splitPoint = int(len(wordSentenceDbLi) * 0.95)
    trainXYLi = wordSentenceDbLi[:splitPoint]
    testXYLi = wordSentenceDbLi[splitPoint:]
    return trainXYLi, testXYLi

def dataToXYListRead(fileName):
    with open(fileName) as file:
        porter_stemmer = PorterStemmer()
        lineCount = 0
        wordSentenceDbLi = []
        while True:
            line = file.readlines(1)
            if not line:
                break
            # if lineCount == 20:
            #     break
            jsonLine = json.loads(line[0])

            # noStopWords = remove_stopwords(jsonLine['text'])
            # stemWords = porter_stemmer.stem(noStopWords)
            stemWords = porter_stemmer.stem(jsonLine['text'])
            tokenWords = simple_preprocess(stemWords, deacc=True)

            # print(tokenWords)
            wordSentenceDbLi.append(tokenWords)
            lineCount += 1
        # yelpDic = corpora.Dictionary(wordSentenceDbLi)
        # yelpDic.save('yelpDictionary.dict')
        # print(yelpDic.token2id)
        # print(yelpDic[8])

        return wordSentenceDbLi

def createWordVecModelFun(fileName):
    print("reading data====>>>")
    wordSentenceDbLi = dataToXYListRead(fileName)

    print("training w2v.....")
    w2v_model = Word2Vec(wordSentenceDbLi, size=100, workers=10, window=3, sg=1)
    w2v_model.save('./w2vTrainedModel/w2v.10th.100vec.model')
    print("end of training word2Vec---->>>")

def wordToXYvecFun(xList, yWord):
    QinWVmodel = 1
    xVec = []
    yVec = []
    for word in xList:
        try:
            x10Vec = w2vMyModel[word]
            xVec.append(x10Vec)
            # print("x converting--->", word, x10Vec)
        except KeyError:
            # print("The word X___", word,  "does not appear in this model")
            QinWVmodel = 0
    #
    try:
        yVec = w2vMyModel[yWord]
        # print("y converting-->", yWord, yVec)
    except KeyError:
        # print("The word Y--->", word, "does not appear in this model")
        QinWVmodel = 0


    return xVec, yVec, QinWVmodel

def processData(xyList):
    xVecDbList = []
    yVecList = []
    xLenth = 4
    for sentence in xyList:
        if len(sentence) < xLenth:
            break
        xSentence = sentence[0:4]
        yWord = sentence[4]
        # print(xSentence, "->", yWord)
        xVecSeqList, yWordVec, QinWVmodel = wordToXYvecFun(xSentence, yWord)
        if QinWVmodel == 1:
            xVecDbList.append(xVecSeqList)
            yVecList.append(yWordVec)

        newSentence = sentence[4:]

        while(len(newSentence) > 5):
            xSentence = newSentence[0:4]
            yWord = newSentence[4]
            xVecSeqList, yWordVec, QinWVmodel = wordToXYvecFun(xSentence, yWord)
            if QinWVmodel == 1:
                xVecDbList.append(xVecSeqList)
                yVecList.append(yWordVec)

            newSentence = newSentence[4:]

    # print("xxxx->", xVecDbList)
    # print('yyyy->', yVecList)
    return xVecDbList, yVecList

def myDataLoader(xVecDbList, yVecList, batchSize):
    xVecDbList = torch.FloatTensor(xVecDbList).to('cuda')
    yVecList = torch.FloatTensor(yVecList).to('cuda')
    xyTensor = TensorDataset(xVecDbList, yVecList)
    trainSampler = RandomSampler(xyTensor)
    xyDataLoader = DataLoader(xyTensor, sampler=trainSampler, batch_size=batchSize)

    return xyDataLoader

class NextWordRNN(torch.nn.Module):
    def __init__(self, batchSize, numSequence, numFeature):
        super(NextWordRNN, self).__init__()

        self.hiddenDim = 3
        self.numLayers = 1
        # self.inputSize = 5 # number of input in a x
        self.inputSize = numFeature # number of input in a x
        self.NumSequence = numSequence
        self.NumFeature = numFeature

        self.rnn = torch.nn.RNN(self.inputSize, self.hiddenDim, batch_first=True)
        # self.fc = torch.nn.Linear(self.hiddenDim, 5)
        self.fcHidden = torch.nn.Linear(self.hiddenDim, 1)
        self.fcFinal = torch.nn.Linear(self.NumSequence, self.NumFeature)

    def forward(self, x):
        batchSize = len(x)
        hidden = self.init_hidden(batchSize)

        out, hidden = self.rnn(x, hidden)
        out = self.fcHidden(out)
        out = out.squeeze()
        out = self.fcFinal(out)

        # print("before view==>", out.shape )
        out = out.view(-1, self.NumFeature)
        # print("after view==>", out.shape )

        return out, hidden

    def init_hidden(self, batchSize):
        hidden = torch.zeros(self.numLayers, batchSize, self.hiddenDim, device='cuda')
        return hidden

def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()