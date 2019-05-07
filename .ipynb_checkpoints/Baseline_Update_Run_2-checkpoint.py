#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
# %matplotlib inline


# In[2]:


from loss import *


# In[3]:


import paths
import config
import util
import importlib
import data
import decay
import s2s_controller
import trainer_with_tensorboar_inference
import char_language_model


# In[4]:


reload_packages = [paths, util, config, data, decay, s2s_controller, trainer_with_tensorboar_inference, char_language_model]
for package in reload_packages:
    importlib.reload(package)
# importlib.reload(data)


# In[5]:


# from tensorboardX import SummaryWriter


# In[6]:


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# # Load Data

# In[7]:


data_helper = data.Data()


# In[8]:


train_loader = data_helper.get_loader(mode='train')
val_loader = data_helper.get_loader(mode='val')
test_loader = data_helper.get_loader(mode='test')


# - [ ] debug loss while validation
#     - why loss is so large for almost the same prediction
# - [x] add teacher force decay
# - [x] add grumbel lr decay
# - [x] try to predict using early models
# - [ ] TODO: in random search inferensing, use a batch of same utterance of size NUM_CANDIDATE instead of looping.
# - [ ] weight initialization according to the paper
# - [ ] init bias with tf
# - [ ] pretrain word embedding

# In[9]:


output_path = os.path.join(paths.output_path, 'Baseline_Try_2')


# # Define model

# In[10]:


data_helper.LANG.n_chars


# In[11]:


parameters = {
    'embed_size' : 40, 
    'hidden_size' : 512, 
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
    'GUMBEL_SOFTMAX' :False,
    'batch_size' : 32, 
    'INIT_STATE' : False
}
# use eos token to pad
# use the vocab_size as the padding value for label embedding and pad_sequence; 
# cannot use -1 for label, which will cause error in criterion


# In[12]:


model = s2s_controller.Sequence2Sequence(**parameters) 


# In[13]:


model


# In[14]:


# initialize the bias of the last linear layer with char prob distribution
model.state_dict().keys()

char_distribution = {i: data_helper.LANG.char2count[i]/sum(data_helper.LANG.char2count.values()) for i in data_helper.LANG.char2count}

decoder_fc2_bias = [np.mean(list(char_distribution.values())), np.mean(list(char_distribution.values()))]

decoder_fc2_bias

decoder_fc2_bias = decoder_fc2_bias + [char_distribution[data_helper.LANG.index2char[i]] for i in range(34) if i not in [0,1]]

len(decoder_fc2_bias)

model.decoder.mlp.fc2.bias = torch.nn.Parameter(torch.tensor(decoder_fc2_bias))
# tie two layers again
# model.decoder.embedding.bias = model.decoder.mlp.fc2.bias


# In[14]:


optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3*0.5, momentum=0.9, nesterov=True)
best_epoch, best_vali_loss, starting_epoch = 0, 400, 0


# In[15]:


# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


# In[16]:


criterion = nn.CrossEntropyLoss(reduction='none')
# criterion = loss.SoftCrossEntropy(reduction='none') # training uses gumbel softmax smoothed ground truth to calculate loss
validation_criterion = nn.CrossEntropyLoss(reduction='none') # validation uses hard label ground truth to calculate loss


# In[17]:


# proceeding from old models
model_path = os.path.join(output_path, 'baseline_s2s_20.pth.tar')
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


# In[18]:


for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()


# In[19]:


init_TEACHER_FORCING_Ratio=1
decay_scheduler = decay.BasicDecay(initial_rate=init_TEACHER_FORCING_Ratio, anneal_rate=1, min_rate=0.6, every_step=5, mode='step')

for i in range(20):
    print(decay_scheduler.get_rate(i))


# In[22]:


# %pdb
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
                            tLog = None,
                            vLog = None,
                            teacher_forcing_scheduler = decay_scheduler,
                            scheduler=None, 
                            start_epoch=starting_epoch, 
                            model_prefix=config.model_prefix,
                            output_path=output_path)


# In[23]:


# # %pdb
# with SummaryWriter(os.path.join(output_path, "tensorboard_logs/train_pytorch"), comment='training') as tLog, SummaryWriter(os.path.join(output_path, "tensorboard_logs/val_pytorch"), comment='testing') as vLog: 
# #     tLog.add_graph(model, test_input, True)
#     trainer_with_tensorboar_inference.run(model = model,
#                                 optimizer = optimizer,
#                                 criterion = criterion,
#                                 validation_criterion = validation_criterion,
#                                 train_dataloader = train_loader,
#                                 valid_dataloader = val_loader,
#                                 language_model = data_helper.LANG,
#                                 best_epoch = best_epoch,
#                                 best_vali_loss = best_vali_loss, 
#                                 DEVICE = DEVICE, 
#                                 tLog = tLog,
#                                 vLog = vLog,
#                                 teacher_forcing_scheduler = decay_scheduler,
#                                 scheduler=scheduler, 
#                                 start_epoch=starting_epoch, 
#                                 model_prefix=config.model_prefix,
#                                 output_path=output_path)


# In[24]:


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

# In[25]:


# %pdb
# model = model.to(DEVICE)
# validation_criterion = validation_criterion.to(DEVICE)
# trainer_with_tensorboar_inference.test_validation(model=model,
#                                                   validation_criterion = validation_criterion,
#                                                   valid_dataloader = val_loader, 
#                                                   language_model = data_helper.LANG, 
#                                                   DEVICE = DEVICE)


# In[26]:


# %%time
# model = model.to(DEVICE)
# validation_criterion = validation_criterion.to(DEVICE)
# print(trainer_with_tensorboar_inference.test_validation_random_search(model=model,
#                                                                       validation_criterion = validation_criterion,
#                                                                       valid_dataloader = val_loader, 
#                                                                       language_model = data_helper.LANG, 
#                                                                       DEVICE = DEVICE, 
#                                                                       MAX_SEQ_LEN=500, 
#                                                                         N_CANDIDATES=100))


# In[27]:


# %%time
# %pdb
# model = model.to(DEVICE)
# validation_criterion = validation_criterion.to(DEVICE)
# print(trainer_with_tensorboar_inference.test_validation_beam_search(model=model,
#                                                                       validation_criterion = validation_criterion,
#                                                                       valid_dataloader = val_loader, 
#                                                                       language_model = data_helper.LANG, 
#                                                                       DEVICE = DEVICE, 
#                                                                       MAX_SEQ_LEN=500, 
#                                                                       beam_size=5, 
#                                                                       num_candidates=1))


# # Testing

# In[16]:


# # %pdb
# for epoch in [21]:
#     # checkpoint = torch.load("checkpoint.pt")
#     model_prediction = s2s_controller.Sequence2Sequence(**parameters) 
#     # proceeding from old models
#     model_path = os.path.join(output_path, 'baseline_s2s_'+str(epoch)+'.pth.tar')
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
# #     references = trainer_with_tensorboar_inference.inference(model_prediction, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500)
#     references = trainer_with_tensorboar_inference.inference_random_search(model_prediction, test_loader,language_model=data_helper.LANG, validation_criterion=validation_criterion, DEVICE=DEVICE, MAX_SEQ_LEN=500, N_CANDIDATES=100, GUMBEL_T=1.5)
#     import pandas as pd
#     varification_results_df = pd.DataFrame({'Id':np.arange(test_loader.dataset.size), 'Predicted':np.array(references)})

#     varification_results_df.to_csv(os.path.join(paths.output_path, 'submission_'+'baseline_random_'+str(epoch)+'.csv'),index=False)


# In[ ]:


# for epoch in [21]:
#     # checkpoint = torch.load("checkpoint.pt")
#     model_prediction = s2s_controller.Sequence2Sequence(**parameters) 
#     # proceeding from old models
#     model_path = os.path.join(output_path, 'baseline_s2s_'+str(epoch)+'.pth.tar')
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
# #     references = trainer_with_tensorboar_inference.inference(model_prediction, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500)
#     references = trainer_with_tensorboar_inference.inference_random_search(model_prediction, test_loader,language_model=data_helper.LANG, validation_criterion=validation_criterion, DEVICE=DEVICE, MAX_SEQ_LEN=500, N_CANDIDATES=100, GUMBEL_T=2.0)
#     import pandas as pd
#     varification_results_df = pd.DataFrame({'Id':np.arange(test_loader.dataset.size), 'Predicted':np.array(references)})

#     varification_results_df.to_csv(os.path.join(paths.output_path, 'submission_'+'baseline_random_'+str(epoch)+'_2.csv'),index=False)


# In[14]:


# # %pdb
# for epoch in [21]:
#     # checkpoint = torch.load("checkpoint.pt")
#     model_prediction = s2s_controller.Sequence2Sequence(**parameters) 
#     # proceeding from old models
#     model_path = os.path.join(output_path, 'backup', 'baseline_s2s_'+str(epoch)+'.pth.tar')
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
# #     references = trainer_with_tensorboar_inference.inference(model_prediction, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500)
#     references = trainer_with_tensorboar_inference.inference_beam_search(model_prediction, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500, beam_size=5, num_candidates=1)
#     import pandas as pd
#     varification_results_df = pd.DataFrame({'Id':np.arange(test_loader.dataset.size), 'Predicted':np.array(references)})

#     varification_results_df.to_csv(os.path.join(paths.output_path, 'submission_'+'baseline_beam_'+str(epoch)+'.csv'),index=False)


# In[15]:


# fast beam search


# In[14]:


# 8 core cpu, beam width 32 will use 1.3 minutes for each utterance


# In[28]:


# # %%time
# # %pdb

# for epoch in [21]:
#     # checkpoint = torch.load("checkpoint.pt")
#     parameters['device'] = 'cuda'
#     model_prediction = s2s_controller.Sequence2Sequence(**parameters) 
#     # proceeding from old models
#     model_path = os.path.join(output_path, 'baseline_s2s_'+str(epoch)+'.pth.tar')
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
#     model_prediction = model_prediction.to(DEVICE)
#     print('val_loss:%.4f val_distance:%.4f' % (checkpoint['val_loss'], checkpoint['val_distance']))
    
#     checkpoint_cpu = torch.load(model_path, map_location='cpu')
#     parameters['device'] = 'cpu'
#     model_reference = s2s_controller.Sequence2Sequence(**parameters) 
#     model_reference.load_state_dict(checkpoint_cpu['model_state_dict'])
# #     references = trainer_with_tensorboar_inference.inference(model_prediction, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500)
#     references = trainer_with_tensorboar_inference.inference_fast_beam_search(model_prediction, model_reference, test_loader,language_model=data_helper.LANG, MAX_SEQ_LEN=500, beam_size=32, num_candidates=1)
#     import pandas as pd
#     varification_results_df = pd.DataFrame({'Id':np.arange(test_loader.dataset.size), 'Predicted':np.array(references)})

#     varification_results_df.to_csv(os.path.join(paths.output_path, 'submission_'+'baseline_beam_'+str(epoch)+'.csv'),index=False)


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




