import random
import numpy as np
import argparse
import torch 
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import torch.nn as nn
import os 
# import csv
from transformers import BertTokenizer
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

class Engagement_cls():
    '''This class classifies each query and response pairs as 0(not engaging) or 1 (engaging)
    '''
    def __init__(self, train_dir, batch_size, mlp_hidden_dim, num_epochs,\
                regularizer = 0.01, lr=1e-4, dropout = 0.1, optimizer="Adam",\
                ftrain_queries_embed=None, ftrain_replies_embed=None, fvalid_queries_embed=None, fvalid_replies_embed=None, ftest_queries_embed=None ,ftest_replies_embed=None):
        print('***************model parameters********************')
        print('mlp  layers {}'.format(mlp_hidden_dim))
        print('learning rate {}'.format(lr))
        print('drop out rate {}'.format(dropout))
        print('batch size {}'.format(batch_size))
        print('optimizer {}'.format(optimizer))
        print('regularizer {}'.format(regularizer))
        print('***************************************************')
       

        self.train_dir = train_dir
        self.batch_size = batch_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.optim = optimizer
        self.reg= regularizer
        self.query = None
        self.reply = None
        self.query_emb = {}
        self.reply_emb = {}
    
    
    def preprocess_input(self, query, reply):
        # convert query, reply into bert embedding  
        # Need to confirm whether need + eos
        MAX_LENGTH = 60
        tokenizer = BertTokenizer.from_pretrained("/work/b07u1234/tien/Chatbot-Project/uncased_L-24_H-1024_A-16/")
        self.query = query
        self.reply = reply
        
        for q in self.query:
            temp = torch.zeros(1, MAX_LENGTH)
            query = tokenizer.encode(q + "<|endoftext|>", return_tensors='pt').to('cuda')
            # print(query)
            # print(query.size())
            # print(temp)
            # print(temp.size())
            temp[ : , : len(query[0])] = query[:]
            self.query_emb[q] = temp
        for r in self.reply:
            temp = torch.zeros(1, MAX_LENGTH)
            reply = tokenizer.encode(r + "<|endoftext|>", return_tensors='pt').to('cuda')
            temp[ : , : len(reply[0])] = reply
            self.reply_emb[r] = temp
        
        
    def generate_eng_score(self):
        '''for all pairs of queries and replies predicts engagement scores
        Params:
            fname_ground_truth: file includes the queries and their ground-truth replies
            foname: file includes the queries, ground truth replies, generated replies (from self.test_replies) and engagement_score of queries and generated replies with following format:
                query===groundtruth_reply===generated_reply===engagement_score of query and generated_reply

        '''

        # if not os.path.isfile(self.train_dir+'best_model_finetuned.pt'):
        #     print('There is not any finetuned model on DD dataset to be used!\nPlease first try to finetune trained model.')
        #     return
        model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(self.train_dir +  'best_model_finetuned.pt'))
        info = torch.load(self.train_dir + 'best_model_finetuned.info')
        model.eval()

        # fw_pred_labels = open(self.data_dir + ofile, 'w')
        # fr_groundtruth_replies = open(self.data_dir + fname_ground_truth, 'r')
        # groundtruth_replies =fr_groundtruth_replies.readlines() 

        print('begining of prediction')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data, param.shape)
        # for stidx in range(0, self.test_size, self.batch_size):
            # x_q = self.test_queries[stidx:stidx + self.batch_size]
            # x_r = self.test_replies[stidx:stidx + self.batch_size]
            # x_groundtruth_r = groundtruth_replies[stidx:stidx + self.batch_size]
        model_output = model(self.query, self.reply, self.query_emb, self.reply_emb)
        pred_eng = torch.nn.functional.softmax(model_output, dim=1)
            # for ind in range(len(self.query_emb)):
                # fw_pred_labels.write(x_q[ind]+'==='+x_groundtruth_r[ind].split('\n')[0]+'==='+x_r[ind]+'==='+str(pred_eng[ind][1].item())+'\n')
         
        print('The engagingness score for specified replies has been predicted!')
        return pred_eng[:][1].item()


    def get_eng_score(self, query, q_embed, reply, r_embed, model):
        '''for a pair of query and reply predicts engagement scores
        Params:
            query: input query
            q_embed: embeddings of query
            reply: input reply
            r_embed: embeddings of reply
           
        '''
        if not os.path.isfile(self.train_dir+'best_model_finetuned.pt'):
            print('There is not any finetuned model on DD dataset to be used!\nPlease first try to finetune trained model.')
            return
            
        model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(self.train_dir +  'best_model_finetuned.pt'))
        info = torch.load(self.train_dir + 'best_model_finetuned.info')
        model.eval()

        model_output = model(query, reply, q_embed, r_embed)
        pred_eng = torch.nn.functional.softmax(model_output, dim=1)
        return pred_eng

 
 
class  BiLSTM(nn.Module):
    '''The engagement classification model is a three layer mlp classifier with having tanh as activation functions which takes the embeddings of query and reply as input and pass their average into the mlp classifier
    '''
    def __init__(self, mlp_hidden_dim=[128], dropout=0.2):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        num_classes=2
        self.mlp_hidden_0 = nn.Linear(768, mlp_hidden_dim[0], bias=True)
        self.mlp_hidden_1 = nn.Linear(mlp_hidden_dim[0], mlp_hidden_dim[1], bias=True)
        self.mlp_hidden_2 = nn.Linear(mlp_hidden_dim[1], mlp_hidden_dim[2], bias=True)
        self.mlp_out = nn.Linear(mlp_hidden_dim[2], num_classes, bias=True)


    def forward(self, queries_input, replies_input,  queries_embeds, replies_embeds):

        for q in queries_input:
            # print("Query is ",q)
            # print("Query Embedding is ",queries_embeds[q])
            if q not in queries_embeds.keys():
                print('the query {} embedding has not been found in the embedding file'.format(q))
        # X_q = torch.zeros(1,1,60)
        # X_r = torch.zeros(1,1,60)
        X_q = torch.tensor([queries_embeds[q].cpu().detach().numpy() for q in queries_input]).to("cuda")
        # print("Q is ",X_q)
        for r in replies_input:
            # print("Query is ",r)
            # print("Reply Embedding is ",replies_embeds[r])
            if r not in replies_embeds.keys():
                print('the reply {} embedding has not been found in the embedding file'.format(r))
        X_r = torch.tensor([replies_embeds[r].cpu().detach().numpy() for r in replies_input]).to("cuda")
        
        if torch.cuda.is_available():
            X_q, X_r = X_q.cuda(), X_r.cuda()
        mlp_input=X_q.add(X_r)
        mlp_input = torch.div(mlp_input,2)
        print(mlp_input.size())
        mlp_h_0 = torch.tanh(self.mlp_hidden_0(mlp_input))
        mlp_h_0= self.dropout(mlp_h_0)
    
        mlp_h_1 = torch.tanh(self.mlp_hidden_1(mlp_h_0))
        mlp_h_1= self.dropout(mlp_h_1)

        mlp_h_2 = torch.tanh(self.mlp_hidden_2(mlp_h_1))
        mlp_h_2= self.dropout(mlp_h_2)

        mlp_out= self.mlp_out(mlp_h_2)
        return mlp_out

def main():
    parser = argparse.ArgumentParser(description='Parameters for engagement classification')
    parser.add_argument('--mlp_hidden_dim', type=int, default=[64, 32, 8],
                    help='number of hidden units in mlp')
    parser.add_argument('--epochs', type=int, default=400,
                    help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
    parser.add_argument('--dropout', type=float, default=0.8,
                    help='dropout rate')
    parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling type for getting sentence embeddings from words embeddings')
    parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer for training model')
    parser.add_argument('--reg', type=float, default=0.001,
                    help='l2 regularizer for training model')
    parser.add_argument('--mode', type=str,
                    help="""train: to train the model 
                          test: to test the model on ConvAI test set
                          testAMT: to test the model on 297 utterances (50 randomly selected dialogs from ConvAI) annotated by Amazon turkers
                          finetune: to finetune the trained model on 300 pairs selected from Daily Dialogue dataset annotated by Amazon turkers 
                          predict: to predict engagement scores for query and generated replies (based on attention-based seq-to-seq model) of Daily Dilaogue dataset""")
    args = parser.parse_args()

    # data_dir = './../data/'
    #train_dir = './../model/'
    train_dir = "/work/b07u1234/tien/Chatbot-Project/PredictiveEngagement/model/"
    queries =  ["Hi, how are you ?", "Are you ok right now ?", "Can I friend you on facebook, lady ?" ]
    replies =  ["It's really good, and you ?", "No, stay away from me !", "Yes, definitely, what's your facebook ID ?"]
    # queries = [queries[0]]
    # replies = [replies[0]]
    eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs, \
                                args.reg, args.lr, args.dropout, args.optimizer)
    eng_cls.preprocess_input(queries, replies )
    score = eng_cls.generate_eng_score()
    for q, r, s in zip(queries, replies, score):
        print("-------------------------------------------------------------------------")
        print(f"Query: {q}")
        print(f"Replies: {r}")
        print(f"Score: {s}")
        print("-------------------------------------------------------------------------")


main()