import numpy as np
import torchvision.transforms as transforms
from Bio import SeqIO
from tqdm.notebook import tqdm
from torchvision.models import alexnet, resnet18

import torch
from torch.nn import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import random
np.random.seed(2020)
random.seed(2020)
torch.manual_seed(2020)

class TransferDataset(Dataset):
    def __init__(self, data_list, labels, transform, model_type='cnn'):
        self.transform = transform
        self.data_list = data_list
        self.labels = labels
        self.model_type =model_type
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        seed = np.random.randint(1e9)       
        random.seed(seed)
        np.random.seed(seed)
        if self.model_type == 'cnn':
            singel_image_ = np.load(self.data_list[idx]).astype(np.float32)
            singel_image_ = self.transform(singel_image_) #for image
        else:
            singel_image_ = self.data_list[idx]
            singel_image_ = torch.FloatTensor(singel_image_)#for sequence
        
        label = self.labels[idx]
        # print(label)
        return singel_image_, label

class torch_dataset_func:
    def __init__(self, model_type = 'cnn'):
        self.data_ds = None
        self.data_dl = None
        self.model_type =model_type

    def get_TransferDataset(self, data_list = None, labels = None, batch_size = 1, shuffle = False):
        transformer = transforms.Compose([
                    transforms.ToTensor(),
                    ])     

        self.data_ds = TransferDataset(data_list= data_list, labels= labels, transform= transformer, 
                                                                        model_type = self.model_type)
        self.data_dl = DataLoader(self.data_ds, batch_size= batch_size, 
                        shuffle=shuffle)

        return self.data_dl
        # get func item
    def get_ds(self,):
        return self.data_ds


class sequence_dataprocess:
    def __init__(self, gene_list = '-NACGT' ):
        self.gene_list = gene_list
        self.label_ = []
        self.class_ =None
        self.new_lineage_label = []
        self.new_lineage_label_1000 = []

    def clean(self, x):
        x = x.upper() 
        if x == 'T' or x == 'A' or x == 'G' or x == 'C' or x == '-' or x == 'N':
            return x
        if x == 'U' or x == 'Y':
            return 'T'
        if x == 'K' or x == 'S':
            return 'G'
        if x == 'M' or x == 'R' or x == 'W' or x == 'H' or x=='V' or x=='D':
            return 'A'
        if x== 'B':
            return 'C'

    def convert_gene_index(self, lineage_list):
        dict_search = {}
        for idx, i in enumerate(self.gene_list):

            dict_search[i] = idx
        print(dict_search)

        num_new_sequences =[]
        for k in tqdm(lineage_list):
            temp_store=[]
            for j in k:
                temp_store.append(self.clean(j)) #one hot
                # temp_store.append(dict_search[clean(j)])
            num_new_sequences.append(temp_store)
        total_sequence_array = np.array(num_new_sequences)
        # print(total_sequence_array.shape)
        return total_sequence_array

    def  dataframe_dataloader(self, fasta_data_path, lineage_label, selection_filter):
        # loading fasta data
        # plz check ur pandas dataframe context and dimension 
        # only for one lineage function
        fasta_data = SeqIO.parse(fasta_data_path,"fasta")
        for idx, rna in enumerate(fasta_data):
            if "B.1.617.2" == lineage_label[idx][0]:
            # break
            # print(lineage_label[idx][0].split(' ')[0])
                self.label_.append(lineage_label[idx][1].split(' ')[0])
                self.new_lineage_label.append(str(rna.seq))
                self.new_lineage_label_1000.append(np.array(list(str(rna.seq)))[selection_filter])
        print('filter sample:', len(self.new_lineage_label))
        print('-----sample len-----')
        print('total sequence shape', len(self.new_lineage_label[0]),'  ||','  filter sequence shape', len(self.new_lineage_label_1000[0]))
        # get database class name
        self.class_, _, _, _= np.unique(self.label_,return_counts=True,return_index=True,return_inverse=True)
        print("-----class name-----")
        class_dict_ = {}
        for idx, i in enumerate(self.class_):
            class_dict_[i] = idx
        print(class_dict_)
        num_label = []
        for i in self.label_:
            num_label.append(class_dict_[i])
        return self.new_lineage_label, self.new_lineage_label_1000, self.class_, num_label

    def gene_index_remaker(self, seq_array ,c_type='Integer'):
            if c_type == 'Integer': #ver1
                gene_index ={'A': 2, 'C': 1, 'G': 3, 'T': 0, 'N': -1, '-': -1}
            elif c_type == 'EIIP': #ver2
                gene_index ={'A': 0.1260, 'C': 0.1340, 'G': 0.0806, 'T': 0.1335, 'N': -0.1, '-': -0.1}
            elif c_type == 'Atomic': #ver3
                gene_index ={'A': 70, 'C': 58, 'G': 78, 'T': 66, 'N': -1, '-': -1}
            num_new_sequences =[]
            for k in tqdm(seq_array):
                temp_single_seq_transfer = []
                for j in k:
                    temp_single_seq_transfer.append(gene_index[j])
                num_new_sequences.append(temp_single_seq_transfer)
            total_sequence_array = np.array(num_new_sequences)
            return total_sequence_array

