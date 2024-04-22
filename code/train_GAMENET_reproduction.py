import torch
import torch.nn as nn
import torch.optim as optim
import dill
from models import GAMENet
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, train_ddi_scores, valid_ddi_scores, train_accuracies=None, valid_losses=None, valid_accuracies=None):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    width=10
    epochs = np.arange(0,len(train_losses))
	
	# Plot loss curves
	# plt.figure(figsize=(10, 5))
    figure, axis  = plt.subplots(1, 3)
    if train_losses is not None : 
        axis[0].plot(epochs, train_losses, label='Training Loss')
    
    if valid_losses is not None : 
        axis[0].plot(epochs, valid_losses, label='Validation Loss')

    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].set_title('Loss Curves')
    axis[0].legend()

    if train_accuracies is not None : 
        axis[1].plot(epochs, train_accuracies, label='Training Accuracy')
    if valid_accuracies is not None : 
        axis[1].plot(epochs, valid_accuracies, label='Validation Accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy')
    axis[1].set_title('Accuracy Curves')
    axis[1].legend()

    #DDI SCORES
    if train_ddi_scores is not None : 
        axis[2].plot(epochs, train_ddi_scores, label='Training DDI score')
        width = 15
    if valid_ddi_scores is not None : 
        axis[2].plot(epochs, train_ddi_scores, label='Validation DDI Score')
        width = 15

    axis[2].set_xlabel('Epoch')
    axis[2].set_ylabel('DDI_SCORE')
    axis[2].set_title('DDI_SCORE_GRAPH')
    axis[2].legend()


    figure.set_size_inches(w=width, h=5)
    plt.show()

def ddi_rate_score(record, path='../data/ddi_A_final.pkl'):
    '''
    this calculates the ddi and we borrowed this from the original paper
    '''
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt

def validate(data, model, voc_size):

    valid_losses = []  #lower the better
    ddi_scores = [] #lower the better

    model.eval()
    with torch.no_grad():
        for patient, visits in enumerate(data):
            for idx, medical_features in enumerate(visits):
                seq_input = visits[:idx+1]  #get all the visits until the current visit
                
                #target is the medication for the patient
                target = np.zeros((1, voc_size[2]))
                target[:, medical_features[2]] = 1
                # print(target)
                output = model(seq_input)
                loss = F.binary_cross_entropy_with_logits(output, torch.FloatTensor(target).to(device))

                # normalising output
                output = F.sigmoid(output).detach().cpu().numpy()[0]
                output[output >= 0.5] = 1
                output[output < 0.5] = 0
                y_label = np.where(output == 1)[0]
                ddi_score = ddi_rate_score([[y_label]])

                loss_i = float((loss.detach().cpu().numpy()))
                ddi_score_i = float(ddi_score)

                # print("output is ",output)

                valid_losses.append(loss_i)
                ddi_scores.append(ddi_score_i)

    return valid_losses,ddi_scores

def train(model,data,optimizer,voc_size):
    model.train()
    train_losses = []  #lower the better
    ddi_scores = [] #lower the better


    for patient, visits in enumerate(data):
        for idx, medical_features in enumerate(visits):
            seq_input = visits[:idx+1]  #get all the visits until the current visit
            
            #target is the medication for the patient
            target = np.zeros((1, voc_size[2]))
            target[:, medical_features[2]] = 1
            # print(target)
            output, batch_neg_loss = model(seq_input)
            loss = F.binary_cross_entropy_with_logits(output, torch.FloatTensor(target).to(device))

            # normalising output
            output = F.sigmoid(output).detach().cpu().numpy()[0]
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            y_label = np.where(output == 1)[0]
            ddi_score = ddi_rate_score([[y_label]])

            loss_i = float((loss.detach().cpu().numpy()))
            ddi_score_i = float(ddi_score)

            # print("output is ",output)

            train_losses.append(loss_i)
            ddi_scores.append(ddi_score_i)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    return train_losses, ddi_scores        
# Some parameters
NUM_EPOCHS = 2
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

# LOADING DATA
data_path = '../data/records_final.pkl'
voc_path = '../data/voc_final.pkl'

ehr_adj_path = '../data/ehr_adj_final.pkl'
ddi_adj_path = '../data/ddi_A_final.pkl'
device = torch.device('cpu')

ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

# MODEL
model = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)


# DATA SPLIT
split_point = int(len(data) * 8/10)
data_train = data[:split_point]

# TRAINING
#medical_features[0] = diagnoses
#medical_features[1] = procedures
#medical_features[2] = medication

# INIT TRACKERS
average_train_losses = []  #lower the better
average_train_ddi_scores = [] #lower the better
average_valid_losses  = []
average_valid_ddi_scores = []

for epoch in range(NUM_EPOCHS):
    train_losses, train_ddi_scores = train(model=model,data=data_train,optimizer=optimizer,voc_size=voc_size)
    valid_losses, valid_ddi_scores = validate(model=model,data=data_train,voc_size=voc_size)

    average_train_losses.append(np.average(np.array(train_losses)))
    average_train_ddi_scores.append(np.average(np.array(train_ddi_scores)))
    average_valid_losses.append(np.average(np.array(valid_losses)))
    average_valid_ddi_scores.append(np.average(np.array(valid_ddi_scores)))

plot_learning_curves(
    train_losses=average_train_losses,
    valid_losses=average_valid_losses,
    train_ddi_scores=average_train_ddi_scores,
    valid_ddi_scores=average_valid_ddi_scores
    )

