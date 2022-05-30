import numpy as np
import torch
from torch import tensor
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import logging

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = False
if USE_CUDA:
    torch.cuda.manual_seed(0)

class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384
    n_layers = 2
    lr = 2e-5
    bidirectional = True
    drop_prob = 0.55
    # training params
    epochs = 10
    print_every = 10
    clip = 5  # gradient clipping
    use_cuda = USE_CUDA
    bert_path = 'bert-base-uncased'
    save_path = './bert_bilstm.pth'
    sampleRate = 2
    labelSelected = 2  # 2/3, 2:Why label; 3:What label


class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.bert = BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = self.bert(x)[0]
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)

        if self.bidirectional:
            hidden_last_L = hidden_last[-2]
            hidden_last_R = hidden_last[-1]
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )
        return hidden



def test_model(input,net,h):
    output = net(input, h)
    logging.info("Model predicted: %s", output)
    logging.info("Model predicted: %s", output)
    #output = torch.nn.Softmax(dim=1)(output)
    output = torch.sigmoid(output)
    logging.info("Response Score: %s",output.data[0][1].data.item())
    return output.data[0][1].data.item()



def setupModel():
    model_config = ModelConfig()
    if model_config.labelSelected == 2:
        print("Why Message")
    else:
        print("What Message")

    net = bert_lstm(model_config.bert_path,
                    model_config.hidden_dim,
                    model_config.output_size,
                    model_config.n_layers,
                    model_config.bidirectional,
                    model_config.drop_prob
                    )
    net.load_state_dict(torch.load(model_config.save_path,map_location=torch.device('cpu')))
    if (model_config.use_cuda):
        net.cuda()
    net.eval()
    

    # init hidden state
    h = net.init_hidden(1)

    net.eval()

    h = tuple([each.data for each in h])

    tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)
    return net, h, tokenizer