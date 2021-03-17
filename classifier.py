from transformers import BertTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
class post_set(Dataset):
    def __init__(self, post, tokenizer):
        #eos = [tokenizer.encoder["<|endoftext|>"]]
        with open(post) as f:
            table = f.readlines()
            
        temp = []
        m = []
        self.ll = []
        self.labels = []
        for l in table:
            # srcs be like   I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.	
            # tgt be like    Was this a friend you were in love with, or just a best friend?
            try:
                srcs, tgt = l.split('\t')
                temp_token = tokenizer.encode(srcs)
                temp_mask = [1 for i in range(len(temp_token))]
                if len(temp_token) >= 100: continue
                temp.append(temp_token[:])
                m.append(temp_mask)
                self.ll.append(len(temp_token))
                self.labels.append(int(tgt))
            except:
                pass
           # print(srcs)
        print(len(temp))
        self.post = pad_sequence([torch.LongTensor(x) for x in temp], batch_first=True, padding_value=0)
        self.mask = pad_sequence([torch.LongTensor(x) for x in m], batch_first=True, padding_value=0)
        
    def __getitem__(self, index):

        return self.post[index], self.labels[index], self.mask[index], self.ll[index]

    def __len__(self):
        return len(self.post)

batch_size = 8
input_file = sys.argv[1]
output_path = sys.argv[2]
training_curve_path = sys.argv[3]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

train_input = post_set(input_file + "train.txt", tokenizer)
valid_input = post_set(input_file + "valid.txt", tokenizer)
train_dataloader = DataLoader(train_input, batch_size=batch_size, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(valid_input, batch_size=batch_size, shuffle=True, num_workers=2)
EPOCH = 5
loss = 0
batch = 0
valid_batch = 0
train_loss = []
valid_loss = []
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# plt.plot(valid_loss)
# plt.ylabel("loss")
# plt.xlabel("500 batch")
# plt.title(f"valid_curve")
# plt.savefig(training_curve_path + "Valid_curve.jpg")
for epoch in range(EPOCH):
    loss = 0
    model.train()
    model.to(device_0)
    for inputs_id, label, mask, ll in tqdm(train_dataloader):
        
        # inputs = tokenizer(inputs_id, return_tensors="pt")
        
        labels = torch.tensor(label).unsqueeze(0).to(device_0)  # Batch size 1
        l, logits = model(inputs_id.to(device_0), labels=labels)
        
        loss += l
        
        if batch % 4 == 0:
            print("Training Loss : ", end = "")
            print(loss)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  
            loss = 0
        # print(outputs)
        # loss = 0
        # break
                
        if valid_batch % 500 == 0: 
            model.eval()
            valid = []
            for inputs_id, label, mask, ll in tqdm(valid_dataloader):
                
                # inputs = tokenizer(inputs_id, return_tensors="pt")
                labels = torch.tensor(label).unsqueeze(0).to(device_0)  # Batch size 1
                loss, logits = model(inputs_id.to(device_0), labels=labels)
                
                valid.append(loss.item())
            print("Valid Loss : ", end = "")
            print(sum(valid)/len(valid))
            valid_loss.append(sum(valid)/len(valid))
            model.train()
            
        batch += 1
        valid_batch += 1
        
            
    
    # valid set
    
    
    torch.save(model, output_path + f"classifier_epoch_{epoch}.pkl")
        
    plt.plot(train_loss)
    plt.ylabel("loss")
    plt.xlabel("batch")
    plt.title(f"Epoch {epoch+1}")
    plt.savefig(training_curve_path + f"Training_curve_{epoch}.jpg")
    plt.clf()
    batch = 0
    train_loss = []
plt.plot(valid_loss)
plt.ylabel("loss")
plt.xlabel("500 batch")
plt.title(f"valid_curve")
plt.savefig(training_curve_path + "Valid_curve.jpg")
plt.clf()

        # logits = outputs.logits