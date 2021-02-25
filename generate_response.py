from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
#from gpt2_training.train_utils import load_model
import torch
import sys

def load_model(model, checkpoint):
    
    model_state_dict = torch.load(checkpoint)

    model_state_dict = fix_state_dict_namespace(model_state_dict)

    start_model = model
    if (hasattr(model, "transformer")
        and all(not s.startswith('transformer.')
                for s in model_state_dict.keys())):
       
        start_model = model.transformer
    start_model.load_state_dict(model_state_dict)

    
    model.cuda()
 
    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict
model_name = sys.argv[1]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#config = GPT2Config.from_json_file("/work/b07u1234/tien/Chatbot-Project/configs/117M/config.json")
from transformers import GPT2LMHeadModel    
model = GPT2LMHeadModel.from_pretrained('temp_model')
#model = load_model(GPT2LMHeadModel(config), model_name)
# model = 
# model.load_state_dict(torch.load(model_name))
model.to('cuda').eval()
# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt').to('cuda')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        do_sample=True, 
        max_length=1000,
        top_k=50, 
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
