import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import StoryGenerator
from embedding.loader import get_w2v, StoryDataset, TEXT_DATA_DIR


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        
        
w2v = get_w2v()

t_ds = StoryDataset('train', w2v, TEXT_DATA_DIR, 82, 80)
v_ds = StoryDataset('valid', w2v, TEXT_DATA_DIR, 82, 80)
        
trainLoader = DataLoader(t_ds, 32)
valLoader = DataLoader(v_ds, 32)
  
sg = StoryGenerator(
    len(wv.wv.vocab), wv.vector_size, 120, 82, 3, 3, 1
)

MODEL_PATH = './model/checkpoint.pt'

optimizer = torch.optim.Adam(sg.parameters(), lr=.01)
loss_func = nn.CrossEntropyLoss()
sg = sg.float()
sg.apply(init_weights)
sg.train()
verbose = True

loss_hist = []
perp_hist = []
val_loss_hist = []
val_perp_hist = []
best_loss = 500
gradient = []
avg_grad = 2
no_improvement = 0
for e in range(10000):
    running_loss = 0
    running_perp = 0
    count = 0
    gradient = []
    avg_grad = 2
    sg.train()
    sg.to(dev)
    for p, s, y in trainLoader:
        t1 = time.time()
        optimizer.zero_grad()
        pred = sg(p.to(dev), s.to(dev))
        loss = loss_func(pred, y.to(dev))
        loss.backward()
        #grad clipping
        total_norm = 0
        if e == 0 and count < 10:
            for p in sg.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient.append(total_norm)
            nn.utils.clip_grad_norm_(sg.parameters(), total_norm)
        elif e==0 and count >= 10:
            avg_grad = np.mean(gradient)
            nn.utils.clip_grad_norm_(sg.parameters(), avg_grad)
        else:
            nn.utils.clip_grad_norm_(sg.parameters(), avg_grad)
        optimizer.step()
        running_loss += loss.item()
        running_perp += math.exp(loss.item())
        count+=1
        print(count, time.time() - t1)
        if count%10 == 0:
            torch.save(sg.state_dict(), MODEL_PATH)
    if e==0:
        avg_grad = np.mean(gradient)
    if verbose:
        print(f'{e}/10000--->Training Loss {running_loss/count}')
        print(f'------------>Training Perp {running_perp/count}')
    loss_hist.append(running_loss/count)
    perp_hist.append(running_perp/count)
    running_loss = 0
    running_perp = 0
    count = 0
    sg.eval()
    for p, s, y in valLoader:
        with torch.no_grad():
            pred = sg(p.to(dev), s.to(dev))
            loss = loss_func(pred, y.to(dev))
        running_loss += loss.item()
        running_perp += math.exp(loss.item())
        count +=1
    val_loss = running_loss/count
    val_perp = running_loss/count
    if verbose:
        print(f'------>Val Loss {val_loss}')
        print(f'------->Val perp {val_perp}')
    val_loss_hist.append(val_loss)
              
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(sg.state_dict(), MODEL_PATH)
        print(f'Checkpointing on Epoch {e}')
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= 100:
            print(f'no improvement for 100 epochs')
            break
