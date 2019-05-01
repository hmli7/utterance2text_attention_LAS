#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt



# In[2]:


import paths
import config
import util
import importlib
import data
import loss
import decay
import sequence2sequence_Atten_modules_withgumbel
import trainer, trainer_with_tensorboar, trainer_with_tensorboar_inference
import char_language_model


# In[3]:


reload_packages = [paths, util, config, data, loss, decay, sequence2sequence_Atten_modules_withgumbel, trainer, trainer_with_tensorboar, trainer_with_tensorboar_inference, char_language_model]
for package in reload_packages:
    importlib.reload(package)
# importlib.reload(data)


# In[4]:


from tensorboardX import SummaryWriter


# In[5]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# # Load Data

# In[6]:


data_helper = data.Data()


# In[7]:


train_loader = data_helper.get_loader(mode='train')
val_loader = data_helper.get_loader(mode='val')
test_loader = data_helper.get_loader(mode='test')


# - [ ] debug loss while validation
#     - why loss is so large for almost the same prediction
# - [x] add teacher force decay
# - [x] add grumbel lr decay
# - [ ] try to predict using early models
# - [ ] weight initialization according to the paper
# - [ ] init bias with tf
# - [ ] pretrain word embedding

# # Define model

# In[8]:


data_helper.LANG.n_chars


# In[9]:


parameters = {
    'embed_size' : 40, 
    'hidden_size' : 256, 
    'n_layers' : 1, # number of hidden layers in the first lstm layer
    'n_plstm'  : 3, # one normal lstm, three plstm
    'mlp_hidden_size' : 128,
    'mlp_output_size' : 128,
    # ---- decoder ----
    'decoder_vocab_size' : data_helper.LANG.n_chars, 
    'decoder_embed_size' : 256, 
    'decoder_hidden_size' : 512, 
    'decoder_n_layers' : 2, 
    'decoder_mlp_hidden_size' : 256,
    'decoder_padding_value' : char_language_model.EOS_token,
    'GUMBEL_SOFTMAX' :False
}
# use eos token to pad
# use the vocab_size as the padding value for label embedding and pad_sequence; 
# cannot use -1 for label, which will cause error in criterion


# In[10]:


model = sequence2sequence_Atten_modules_withgumbel.Sequence2Sequence(**parameters) 


# In[11]:


model


# In[12]:


optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3*0.5, momentum=0.9, nesterov=True)
best_epoch, best_vali_loss, starting_epoch = 0, 400, 0


# In[13]:


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=1, verbose=True)


# In[14]:


criterion = nn.CrossEntropyLoss(reduction='none')
# criterion = loss.SoftCrossEntropy(reduction='none') # training uses gumbel softmax smoothed ground truth to calculate loss
validation_criterion = nn.CrossEntropyLoss(reduction='none') # validation uses hard label ground truth to calculate loss


# In[17]:


output_path = os.path.join(paths.output_path, 'experiment_baseline_outputs')


# In[15]:


# proceeding from old models
model_path = os.path.join(output_path, 'baseline_s2s_5.pth.tar')
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
starting_epoch = checkpoint['epoch']+1
# best_vali_acc = checkpoint['best_vali_acc']
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)
optimizer.load_state_dict(checkpoint['optimizer_label_state_dict'])
best_vali_loss = checkpoint['best_vali_loss']
best_epoch = checkpoint['best_epoch']
print("=> loaded checkpoint '{}' (epoch {})"
      .format(model_path, checkpoint['epoch']))
# del checkpoint, model_state_dict


# In[ ]:


for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()


# In[16]:


init_TEACHER_FORCING_Ratio=0.9
decay_scheduler = decay.BasicDecay(initial_rate=init_TEACHER_FORCING_Ratio, anneal_rate=0.8, min_rate=0, every_step=5, mode='step')

for i in range(20):
    print(decay_scheduler.get_rate(i))


# In[ ]:


with SummaryWriter(os.path.join(output_path, "tensorboard_logs/train_pytorch"), comment='training') as tLog, SummaryWriter(os.path.join(output_path, "tensorboard_logs/val_pytorch"), comment='testing') as vLog: 
#     tLog.add_graph(model, test_input, True)
    trainer_with_tensorboar_inference.run(model = model,
                                optimizer = optimizer,
                                criterion = criterion,
                                validation_criterion = validation_criterion,
                                train_dataloader = train_loader,
                                valid_dataloader = val_loader,
                                language_model = data_helper.LANG,
                                best_epoch = best_epoch,
                                best_vali_loss = best_vali_loss, 
                                DEVICE = DEVICE, 
                                tLog = tLog,
                                vLog = vLog,
                                teacher_forcing_scheduler = decay_scheduler,
                                scheduler=scheduler, 
                                start_epoch=starting_epoch, 
                                model_prefix=config.model_prefix,
                                output_path=output_path)


# In[17]:


# %pdb
# trainer.run(model = model,
#             optimizer = optimizer, 
#             criterion = criterion, 
#             train_dataloader = train_loader, 
#             valid_dataloader = val_loader, 
#             language_model = data_helper.LANG,
#             best_epoch = best_epoch, 
#             best_vali_loss = best_vali_loss, 
#             DEVICE = DEVICE, 
#             TEACHER_FORCING_Ratio = 1,
#             scheduler=None, 
#             start_epoch=starting_epoch, 
#             model_prefix=config.model_prefix)


# # Validation

# In[28]:


# %pdb
# model = model.to(DEVICE)
# validation_criterion = validation_criterion.to(DEVICE)
# trainer_with_tensorboar_inference.test_validation(model=model,
#                                                   validation_criterion = validation_criterion,
#                                                   valid_dataloader = val_loader, 
#                                                   language_model = data_helper.LANG, 
#                                                   DEVICE = DEVICE)


# In[29]:


# 757/64


# In[18]:


# %%time
# model = model.to(DEVICE)
# validation_criterion = validation_criterion.to(DEVICE)
# print(trainer_with_tensorboar_inference.test_validation_random_search(model=model,
#                                                                       validation_criterion = validation_criterion,
#                                                                       valid_dataloader = val_loader, 
#                                                                       language_model = data_helper.LANG, 
#                                                                       DEVICE = DEVICE, 
#                                                                         MAX_SEQ_LEN=500, 
#                                                                         N_CANDIDATES=10))


# # Testing

# In[60]:


# for epoch in [20]:
#     # checkpoint = torch.load("checkpoint.pt")
#     model_prediction = sequence2sequence_Atten_modules_withgumbel.Sequence2Sequence(**parameters) 
#     # proceeding from old models
#     model_path = os.path.join(paths.output_path, 'baseline_s2s_'+str(epoch)+'.pth.tar')
#     print("=> loading checkpoint '{}'".format(model_path))
#     checkpoint = torch.load(model_path)
#     starting_epoch = checkpoint['epoch']+1
#     model_state_dict = checkpoint['model_state_dict']
#     model_prediction.load_state_dict(model_state_dict)
#     # optimizer.load_state_dict(checkpoint['optimizer_label_state_dict'])
#     best_vali_loss = checkpoint['best_vali_loss']
#     best_epoch = checkpoint['best_epoch']
#     print("=> loaded checkpoint '{}' (epoch {})"
#           .format(model_path, checkpoint['epoch']))

#     print('val_loss:%.4f val_distance:%.4f' % (checkpoint['val_loss'], checkpoint['val_distance']))

#     model_prediction = model_prediction.to(DEVICE)
#     references = trainer_with_tensorboar_inference.inference(model_prediction, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500)
#     import pandas as pd
#     varification_results_df = pd.DataFrame({'Id':np.arange(test_loader.dataset.size), 'Predicted':np.array(references)})

#     varification_results_df.to_csv(os.path.join(paths.output_path, 'submission_'+'baseline_greedy_'+str(epoch)+'.csv'),index=False)


# In[30]:


# %pdb
# model = model.to(DEVICE)
# references = trainer_with_tensorboar_inference.inference(model, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500)


# In[ ]:





# In[ ]:


# references_random_search = trainer_with_tensorboar_inference.inference(model, test_dataloader, data_helper.LANG, MAX_SEQ_LEN=500, N_CANDIDATES=100)


# # Evaluation

# In[102]:


# encoder = nn.Sequential(list(model.children())[0]).to(DEVICE)
# decoder = nn.Sequential(list(model.children())[1]).to(DEVICE)


# In[103]:


# test = next(iter(train_loader))


# In[104]:


# utterance = test[0][0:2]
# # utterance.size()


# In[106]:


# utterance[0].size(), utterance[1].size()


# In[107]:


# transcript = test[1][0:2]
# # transcript.size()
# transcript[0].size(), transcript[1].size()


# In[108]:


# y_hat_label, labels_padded, attentions, masked_energy, key_mask  = trainer.evaluate(encoder=encoder,
#                                                          decoder=decoder,
#                                                          lang=data_helper.LANG,
#                                                          utterance=utterance,
#                                                          transcript=transcript,
#                                                          TEACHER_FORCING_Ratio=1)


# In[148]:


# y_hat_label.size()


# In[110]:


# labels_padded.size()


# In[111]:


# labels_padded


# In[112]:


# attentions.size()


# In[113]:


# attentions


# In[114]:


# masked_energy


# In[116]:


# key_mask.size()


# In[132]:


# key_mask[1][1,:]


# In[137]:


# torch.full((1,10), 3)


# In[50]:


# plt.imshow(utterance.cpu().numpy().T, interpolation='nearest')
# plt.show()


# In[51]:


# sentence = data_helper.LANG.indexes2string(labels_padded.cpu().numpy()[0])
# sentence


# In[52]:


# trainer.showAttention(list(sentence), attentions.squeeze(0).cpu().numpy().T)


# In[55]:


# # testing gumbel
# test = torch.randint(0,10,(2,4))
# torch.FloatTensor(test.size()).uniform_(0,1)

# test = nn.Embedding(3,3)

# test(torch.LongTensor([0]))

# batch_size = 5
# length_padded = 10
# nb_digits = 10

# y = torch.LongTensor(length_padded,batch_size, 1).random_() % nb_digits

# # One hot encoding buffer that you create out of the loop and just keep reusing
# y_onehot = torch.FloatTensor(length_padded,batch_size, nb_digits)

# # In your for loop
# y_onehot.zero_()
# y_onehot.scatter_(2, y, 1)

# print(y)
# print(y_onehot)

# class Gumbel_Softmax(nn.Module):
#     """Gumbel_Softmax [Gumbel-Max trick (Gumbel, 1954; Maddison et al., 2014) + Softmax (Jang, E., Gu, S., & Poole, B., 2016)]
#         source: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
#     """
#     def __init__(self):
#         super(Gumbel_Softmax, self).__init__()
    
#     def sample_gumbel_distribution(self, size, eps=1e-20):
#         """sample from Gumbel Distribution (0,1)"""
#         uniform_samples = torch.FloatTensor(size).uniform_(0, 1)
#         return -torch.log(-torch.log(uniform_samples + eps) + eps)

#     def forward(self, logits, temperature):
#         y = logits + self.sample_gumbel_distribution(logits.size(), eps=1e-20)
#         return nn.functional.softmax(y/temperature, dim=-1)

# test_gumbel = Gumbel_Softmax()

# test = test_gumbel(y_onehot, 1)

# y_onehot[0,0,:]


# In[ ]:




