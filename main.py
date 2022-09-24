# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:# sentences = ["i like dog", "i love coffee", "i hate milk"]
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

class textRNN(nn.Module):#两层的hidden layer
    def __init__(self):
        super(textRNN, self).__init__()
        self.rnn=nn.RNN(input_size=n_class,hidden_size=n_hidden,batch_first=True)
        self.W=nn.Linear(in_features=n_hidden,out_features=n_class)
        self.b=nn.Parameter(torch.ones([n_class]))

    def forward(self,input_data):
        # print('之前input_data.shape==',input_data.shape)
        # input_data = input_data.transpose(0, 1)#??????
        # print('之后input_data.shape==',input_data.shape)
        outputs,_=self.rnn(input_data)
        #print('outputs.shape==',outputs.shape)
        #outputs.shape= [3,2,100]
        # outputs=outputs[-1]
        temp=torch.Tensor(np.zeros((3,100)))
        for i in range(outputs.shape[0]):
            temp[i]=outputs[i,1]






        #截取之后的outputs.shape==[2,100]

        model=self.W(temp)+self.b

        return model




if __name__ == '__main__':
    n_step = 2 # number of cells(= number of Step)
    n_hidden = 100 # number of hidden units in one cell

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)

    batch_size = len(sentences)#batch_size==3

    model = textRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()#input_batch保存了训练序列

    input_batch = torch.FloatTensor(input_batch)
    #print('input_batch.shape==',input_batch.shape)#[3,2,7]

    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(100000):
        optimizer.zero_grad()


        # input_batch : [batch_size, n_step, n_class]
        output = model(input_data=input_batch)
        #print('output==',output)
        #print('target_batch==',target_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    #print('input==',input)

    # Predict


    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

