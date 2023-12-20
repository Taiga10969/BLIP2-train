import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

from lavis.models import load_model_and_preprocess

import wandb

import utils
import configs
from dataset import SciCapDataset

from transformers import T5TokenizerFast

wandb.init(project="BLIP2_scicap_training_fp16_ALL")
project_name = 'scicap_training'

config = configs.Config()
utils.setup_torch_seed(seed=config.seed)

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')

# model_load
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5',
    model_type='pretrain_flant5xl', #pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    is_eval=False,
    device=device
)

#model = nn.DataParallel(model)

model.to(device)

msg = model.load_state_dict(torch.load('/taiga/moonshot/blip2_train_fp16_t5/result/pth/epoch1_t5.pth'))
print("msg : ", msg)


model.max_txt_len = 480
model = model.to(torch.float16)


summary(model)


optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

#lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
#                                                              T_0=config.t_0, 
#                                                              T_mult=config.t_mult)


dataset = SciCapDataset(dataset_path = '/taiga/Datasets/scicap_data', 
                       transform = None,
                       train = True,
                       train_include_val = True,
                       include_subfig = False,
                       image_processor=vis_processors['eval'],
                       tokenizer=None,
                       generate = True,
                       )
print('len(dataset) : ', len(dataset))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)


train_loss_list = []

for epoch in range(1, config.epoch+1, 1):
    start_time = time.time()
    with tqdm(dataloader) as pbar:
        pbar.set_description(f'[train epoch : {epoch}]')
        model.train()
        sum_train_loss = 0.0
        train_loss = 0.0
        train_count = 0

        for images, input_prompts, captions in pbar:
            train_count+=1

            input_datas = {"image": images.to(device, torch.float16), "text_input": input_prompts, "text_output": captions}

            outputs = model(input_datas)

            loss = outputs['loss']
            sum_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ave_loss=sum_train_loss/train_count
            wandb.log({"iter": loss.item()})
            pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=ave_loss, lr = optimizer.param_groups[0]['lr']))

    #lr_scheduler.step()
    end_time = time.time()
    train_loss_list.append(ave_loss)
    print(f"epoch : {epoch}, train_loss : {ave_loss}, time : {end_time-start_time}s")
    wandb.log({"epoch": epoch, "train_loss" : ave_loss, "time" : end_time-start_time})
    
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"result/pth/{project_name}_epoch{epoch}.pth")


# リストをJSONフォーマットに変換
json_data = json.dumps(train_loss_list)
# JSONデータをファイルに書き込む
with open(f'result/{project_name}_train_loss_list.json', 'w') as file:
    file.write(json_data)


wandb.finish()


