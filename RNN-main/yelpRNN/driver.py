from funClass import *
import torch
import torch.optim as optim
import  numpy as np

# createWordVecModelFun('./wordDataJson/review10th.json')

# ================================
wordSentenceDbLi = dataToXYListRead('./wordDataJson/review1000th.json')
trainXYLi, testXYLi = splitToTrainTestFun(wordSentenceDbLi)

# xVec, yVec = processData(trainXYLi)
#
# batchSize = 3000
# xyTainSet = myDataLoader(xVec, yVec, batchSize=batchSize)
# rnnModel = NextWordRNN(batchSize=batchSize, numSequence=4, numFeature=10)
# rnnModel.to(device='cuda')
# criterion = torch.nn.MSELoss()
# optimizer = optim.Adam(rnnModel.parameters(), lr=0.001)
#
# numEpochs = 30
# for epoch in range(numEpochs):
#     rnnModel.train()
#     train_running_loss = 0.0
#     train_acc = 0.0
#     for i, data in enumerate(xyTainSet):
#         optimizer.zero_grad()
#
#         rnnModel.init_hidden(batchSize=batchSize)
#
#         xTrain, yTrain = data
#         yPredict, hidden = rnnModel(xTrain)
#
#         loss = criterion(yPredict, yTrain)
#         loss.backward()
#         optimizer.step()
#     #
#         train_running_loss += loss.detach().item()
#         # train_acc += get_accuracy(yPredict, yTrain, batchSize)
#     print("epoch-->", epoch)
#     torch.save(rnnModel.state_dict(), './rnnModel/rnnModel.model')
#
#     torch.cuda.empty_cache()

# ===========================================================
# ***************************************************************
xVec, yVec = processData(testXYLi)

batchSize = 1
xyTestSet = myDataLoader(xVec, yVec, batchSize=batchSize)
rnnModel = NextWordRNN(batchSize=batchSize, numSequence=4, numFeature=10)
rnnModel.load_state_dict(torch.load('./rnnModel/rnnModel.model'))
rnnModel.to('cuda')

w2vMyModel = Word2Vec.load('./w2vTrainedModel/w2v.10th.10vec.model')
count = 0
for data in xyTestSet:
    xTest, yTest = data
    yPredict, hidden = rnnModel(xTest)
    # yPredict = yPredict.squeeze()
    yTest = yTest.cpu().detach().numpy()
    yOri = w2vMyModel.most_similar(yTest, topn=1)

    yPredict = yPredict.cpu().detach().numpy()
    # similarWord = w2vMyModel.most_similar(positive=yPredict, topn=1)
    similarWord = w2vMyModel.most_similar(yPredict, topn=1)
    print(yOri, "-->", similarWord)
    if count == 10:
        break
    count += 1

# yPredict = [ 0.6160, -0.1247, -0.7061,  0.2317, -0.0927, -0.5006,  0.6845, -0.2665,
#           0.0829,  0.0766]
# yPredict = np.array(yPredict)
# similarWord = w2vMyModel.most_similar([yPredict],[], topn=1)
# similarWord = [yPredict]. w2v
# print(similarWord)
#


