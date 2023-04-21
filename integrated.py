import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
from torch.utils.data import Dataset, DataLoader
import math
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, Precision, Recall
import csv

import pathlib
import esm
import re
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from os import listdir, mkdir, remove
from shutil import rmtree
import os

""""............................generate data sets........................................"""

def calculate_esm(model_location, fasta_file, output_dir, GPU_name='cuda:0', toks_per_batch=4096, nogpu=False, return_contacts=False, repr_layers=[-1]):
    fasta_file = pathlib.Path(fasta_file)
    output_dir = pathlib.Path(output_dir)
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda(GPU_name)
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device=GPU_name, non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["mean_representations"] = {
                    layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                    for layer, t in representations.items()
                }
                torch.save(
                    result,
                    output_file,
                )
                
def esm_to_np(dir_name, out_put_file):
    """
    convert esm mean_representations to np
    Parameters: dir_name: directory of esm 
                out_put_file: npy name
    """
    li = []
    for i in listdir(dir_name):
        ts = torch.load(dir_name+"/"+i)["mean_representations"][33]
        li.append(ts)
    np.save(out_put_file, np.stack(li, axis=0))
    
# def map_col(col_name):
#     if "Bind" in col_name:
#         return "Bind"
#     elif "Display" in col_name:
#         return "Display"
#     else:
#         return col_name
    
def generate_training_dataset_from_csv(filepath, sequence_col_name, output_dirc, GPU_name='cuda'):
    """
    Args: 
        filepath: The filepath of csv file
        sequence_col_name: Sequence column name
        output_dirc: output dirctory that stores training datasets
        GPU_name: GPU for esm embedding
    """
    if not os.path.exists(output_dirc):
        mkdir(output_dirc)
    df = pd.read_csv(filepath)
    generating(df, sequence_col_name, output_dirc, GPU_name=GPU_name)
    
    
def generate_training_dataset_from_excel(filepath, sequence_col_name, output_dirc, GPU_name='cuda', sheet_name=0):
    """
    Args: 
        filepath: The filepath of excel file
        sequence_col_name: Sequence column name
        output_dirc: output dirctory that stores training datasets
        GPU_name: GPU for esm embedding
        sheet_name: should be used when having multi sheets in one excel file.
            Strings are used for sheet names. Integers are used in zero-indexed sheet positions (chart sheets do not count as a sheet position)
    """
    if not os.path.exists(output_dirc):
        mkdir(output_dirc)
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    generating(df, sequence_col_name, output_dirc, GPU_name=GPU_name)

def generating(df, sequence_col_name, output_dirc, GPU_name):
    df = df[df[sequence_col_name].apply(lambda x: type(x) == str)]
    df = df[df[sequence_col_name].map(lambda x: False == bool(re.search('[^ARNDCEQGHILKMFPSTWYV]', x)))]
    df.to_csv(output_dirc+"/refined.csv")
    
    #write fasta file
    i = 0
    f = open(output_dirc+"/refined.fas",'w')
    for index, item in df[sequence_col_name].items():
        if i == 0:
            f.write(">"+str(i)+"\n")
            f.write(item)
        else:
            f.write("\n>"+str(i)+"\n")
            f.write(item)            
        i += 1
    f.close()
    
    calculate_esm("esm1b_t33_650M_UR50S", output_dirc+"/refined.fas", output_dirc+"/emb_esm1b", GPU_name=GPU_name)
    esm_to_np(output_dirc+"/emb_esm1b", output_dirc+"/esm.npy")
    rmtree(output_dirc+"/emb_esm1b")
    remove(output_dirc+"/refined.fas")

""""............................antibody cls........................................"""

class antibody_cls:
    def __init__(self, dataset_dirc, mutation_count_column=None, D_col="Display", B_col="Bind", bin_num=2, using_chain_as_feature=False, name=None):
        """Load antibody dataset generated by generate_training_dataset function into one object
        
        Can handle two or four gates binding data.
        If experiments have four gates, csv file need 'Bin1','Bin2','Bin3','Bin4' as reads column names
        If experiments have two gates, column names of display and binding reads need to be specified
        
        Args: 
            dataset_dirc: the name of dirctory generated by generate_training_dataset function
            mutation_count_column: Mutation count column name, used for finding parental sequence
                If not provided, the parental index default 0.
            D_col: Display reads column name, only for 2 gates experiment data
            B_col: Binding reads column name, only for 2 gates experiment data
            bin_num: The number of gates in experiments, should be 2 or 4
            using_chain_as_feature: Add HC or LC as one input feature or not.
            name: only used when using chain feature, distinguish HC or LC by checking whether having "HC" or "LC" in name
                (if deal with CCR8, name will be used for cells count reference, it should be "35A6_HC", "35A6_LC", "35B6_HC", or "35B6_LC")
        """
        self.esm = np.load(dataset_dirc+"/esm.npy")
        self.n_sequences = self.esm.shape[0]
        self.df = pd.read_csv(dataset_dirc+"/refined.csv")
        self.bin_num = bin_num
        self.name = name
        self.D_col = D_col 
        self.B_col = B_col
        self.using_chain_as_feature = using_chain_as_feature
        if using_chain_as_feature:
            if 'HC' in name:
                self.chain = 1
            else:
                self.chain = 0
        
        if mutation_count_column:
            self.update_parent_index(mutation_count_column)
        else:
            self.parent_index=0
        
    
    def generate_training_array(self):
        self.cal_binding()
        self.cal_ln_weight()
        self.generate_relative_data()
        self.calculate_training_array()
            
    
    def update_parent_index(self, mutation_count_column):
        """Update the parental index using mutation count"""
        self.parent_index = self.df[self.df[mutation_count_column]==0].index[0]
    
    #cell counts for CCR8
    counts = {'35A6_HC': {'cells_bins': [42835, 257193, 611967, 861289], 'reads_bins': [288091, 519296, 708095, 1174710]},
              '35A6_LC': {'cells_bins': [4231,   90008,   311152,   559060], 'reads_bins': [445436,   671442,   1013472,   1387306]},
              '35B6_HC': {'cells_bins': [127359,   304017,   447249,   617209], 'reads_bins': [327636,   538160,   809533,   922335]},
              '35B6_LC': {'cells_bins': [26841,   231633,   339631,   412200], 'reads_bins': [431283,   570909,   932746,   1204253]}}
    
    def cal_binding(self):
        """Calculate binding affinity and add the binding affinity as a column 'binding' to date frame
        
        When 4 gates:
            Calculate the weighted average using 'Bin1','Bin2','Bin3','Bin4' columns and antibody_cls.counts
            
        When 2 gates:
            Calculate the ratio of B_frac/D_frac
        
        """
        if self.bin_num==4:
            df = self.df[['Bin1','Bin2','Bin3','Bin4']]
            df = df/antibody_cls.counts[self.name]['reads_bins']*antibody_cls.counts[self.name]['cells_bins']
            # df = df/sum(antibody_cls.counts[self.name]['cells_bins'])
            # self.df['binding'] = df.apply(lambda x: antibody_cls.func1(x['Bin1'], x['Bin2'], x['Bin3'], x['Bin4']), axis=1)
            self.df['binding'] = df.apply(lambda x: antibody_cls.func2(x['Bin1'], x['Bin2'], x['Bin3'], x['Bin4']), axis=1)
            
        elif self.bin_num==2:
            sum_bind = self.df[self.B_col].sum()
            sum_display = self.df[self.D_col].sum()
            pop_frac_bind = self.df[self.B_col]/sum_bind
            pop_frac_display = self.df[self.D_col]/sum_display
            self.df['binding'] = pop_frac_bind/pop_frac_display
            
    @staticmethod
    def func1(a,b,c,d):
        return (a*1+b*2+c*3+d*4) / (a+b+c+d)
    
    @staticmethod
    def func2(a,b,c,d):
        return (a+b)/(c+d) if (c+d) != 0 else float('inf')
    
    def cal_ln_weight(self):
        """Calculate sample weight and add the sample weight as a column 'sample_weight' to date frame
        
        When 4 gates:
            Calculate sample weight basing on reads and cell counts, sample_weight = ln(sample_cell_count + 1)
            
        When 2 gates:
            sample_weight = ln(B_col+D_col)
            
        """
        if self.bin_num==4:
            # df = self.df[['Bin1','Bin2','Bin3','Bin4']]
            # df = df/antibody_cls.counts[self.name]['reads_bins']*antibody_cls.counts[self.name]['cells_bins']
            # self.df['cells_count'] = df.sum(axis=1)
            # self.df['sample_weight'] = np.log(self.df['cells_count'] + 1)
            self.df['sample_weight'] = np.log(self.df['Bin1']+self.df['Bin2']+self.df['Bin3']+self.df['Bin4'])
        elif self.bin_num==2:
            self.df['sample_weight'] = np.log(self.df[self.B_col]+self.df[self.D_col])
        
    def classify(self):
        """Classify numerical binding data and add the results as a column 'class' to date frame
        
        Note: this methods does not use parental binding data
        Used for numerical binding data calculated by weighted average of gate reads
        """
        self.df["class"] = self.df['binding'].apply(self.map_1)
            
    def generate_relative_data(self):
        """Calculate relative esm representation and classify binding affinity comparing to the parental sequence
        
        After running cal_ln_weight and cal_binding
        Generate self.df_relative dataframe, including a column "class"
        Generate self.esm_relative array with a shape of (n_sequences-1, dimension_of_ESM_repres)
        """
        parent_series = self.df.loc[self.parent_index]
        self.df_relative = self.df.drop(index=self.parent_index)
        self.up_20 = parent_series['binding'] * 1.2
        self.parent_bind = parent_series['binding']
        self.down_20 = parent_series['binding'] * 0.8
        self.df_relative["class"] = self.df_relative['binding'].apply(self.map_2)
        count = np.bincount(self.df_relative["class"].to_numpy())
        self.classes_ratio = count[0]/count[1]
            
        parent_repres = self.esm[self.parent_index]
        self.esm_relative =  np.delete(self.esm, self.parent_index, axis=0)
        self.esm_relative = self.esm_relative - parent_repres
        
    @staticmethod
    def map_1(x):
        """convert numerical binding affinity (1~4) to class [2,1,0]"""
        if x < 2:
            return 2
        elif x < 3:
            return 1
        else:
            return 0
             
    def map_2(self,x):
        if x < self.down_20:
            return 0
        else:
            return 1    
        
    def calculate_training_array(self, relative_data = True):
        """Store all training necessory data in one array, self.arr_relative or self.arr"""
        if relative_data:
            df_array = self.df_relative[['sample_weight','class']].to_numpy()
            if self.using_chain_as_feature:
                chain_feature = np.array( [[self.chain]] * (self.n_sequences -1) )
                self.arr_relative = np.concatenate((chain_feature, self.esm_relative, df_array), axis=1)
            else: 
                self.arr_relative = np.concatenate((self.esm_relative, df_array), axis=1)
                
            del self.esm
            del self.esm_relative
        else:
            df_array = self.df[['sample_weight','class']].to_numpy()
            self.arr = np.concatenate((self.esm, df_array), axis=1)
            del self.esm
            
class antibody_cls_from_BLADE(antibody_cls):
    """Load antibody dataset generated by generate_training_dataset function into one object
    
    Can handle dataset designed for BLADE, the dataset should have "Observations" and "Response" columns, and has parent sequence in the first row.
    
    Args: 
        dataset_dirc: the name of dirctory generated by generate_training_dataset function
        using_chain_as_feature: Add HC or LC as one input feature or not.
        name: only used when using chain feature, distinguish HC or LC by checking whether having "HC" or "LC" in name
            (if deal with CCR8, name will be used for cells count reference, it should be "35A6_HC", "35A6_LC", "35B6_HC", or "35B6_LC")
    """
    
    def __init__(self, dataset_dirc, using_chain_as_feature=False, name=None):
        """
        """
        super().__init__(dataset_dirc=dataset_dirc, using_chain_as_feature=using_chain_as_feature, name=name)
        self.df.rename(columns = {'Response':'binding'}, inplace = True)
    
    def generate_training_array(self):
        self.cal_ln_weight()
        self.generate_relative_data()
        self.calculate_training_array()
        
    def cal_ln_weight(self):
        self.df['sample_weight'] = np.log(self.df["Observations"])
        
        
""""............................Random Forest classifier........................................"""


def cal_class_weight_np(array, ratio=1):
    """Calculate class weights"""
    if ratio == None:
        return None
    r = [1]
    r.append(ratio)
    li = np.bincount(array[:,-1].astype(int), weights=array[:,-2])
    li = li/li.max()
    li = 1/li
    li = li * r
    return dict(enumerate(li))

def print_metrics(y_test, y_pred, sample_weight):
    print("F1_score for [lower affinity, maintain or increase] class:",metrics.f1_score(y_test, y_pred, sample_weight=sample_weight, average=None))
        
def train_rf(antibody_list=[], verbose=0, n_jobs=25, max_depth=20, tune_hyper=False, parameters={}, fbeta=1):
    """Train and test a RandomForestClassifier using sklearn
    
    Args:
        antibody_list: A list of antibody objects
        verbose: Controls the verbosity when fitting, set 2 for detials
        n_jobs: The number of jobs to run in parallel
        max_depth: The max depth of trees
        tune_hyper: GridSearchCV or not
        parameters: param_griddict or list of dictionaries. Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of parameter settings.
        fbeta: Determines the weight of recall in the F score, beta < 1 lends more weight to precision, while beta > 1 favors recall 
            (beta -> 0 considers only precision, beta -> +inf only recall).
        
    Return:
        RandomForestClassifier object
    """
    array = np.concatenate([i.arr_relative for i in antibody_list])

    # if use_class_weight:
    #     class_weight = cal_class_weight(array, ratio=class_ratio)
    # else:
    #     class_weight = None
        
    X = array[:,:-2]
    w = array[:,-2]
    y = array[:,-1]
    
    # X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    #                                                     X, y, w, test_size=test_size, random_state=0) # random number from np
    
    if tune_hyper:
        class_weight = cal_class_weight_np(array)
        rfc = RFC(n_jobs=n_jobs, class_weight = class_weight)
        scorer = make_scorer(fbeta_score, beta=fbeta, pos_label=1)
        clf = GridSearchCV(rfc, parameters, scoring=scorer, verbose=verbose)
    else:
        class_weight = cal_class_weight_np(array)
        clf = RFC(class_weight = class_weight, max_depth=max_depth, verbose=verbose, n_jobs=n_jobs)
        
    X = array[:,:-2]
    w = array[:,-2]
    y = array[:,-1]
    del(array)
    
    clf.fit(X, y, sample_weight=w)
    
    # if test:
    #     y_pred=clf.predict(X_test)
    #     print("metrics on the test set:")
    #     print_metrics(y_test,y_pred,w_test)
        
    return clf

def test_rf(clf, antibody_list):
    """Test a trained random forest on an antibody list
    
    Args:
        clf: A trained classifier
        antibody_list: A list of antibody objects
    """
    array = np.concatenate([i.arr_relative for i in antibody_list])
    X = array[:,:-2]
    w = array[:,-2]
    y = array[:,-1]
    
    y_pred=clf.predict(X)
    print_metrics(y, y_pred, w)
        
def F1score_rf(clf, antibody):
    y=antibody.arr_relative[:, -1]
    y_pred=clf.predict(antibody.arr_relative[:,:-2])
    sample_weight=antibody.arr_relative[:, -2]
    return metrics.f1_score(y, y_pred, sample_weight=sample_weight)

def recall_rf(clf, antibody):
    y=antibody.arr_relative[:, -1]
    y_pred=clf.predict(antibody.arr_relative[:,:-2])
    sample_weight=antibody.arr_relative[:, -2]
    return metrics.recall_score(y, y_pred, sample_weight=sample_weight)

def precision_rf(clf, antibody):
    y=antibody.arr_relative[:, -1]
    y_pred=clf.predict(antibody.arr_relative[:,:-2])
    sample_weight=antibody.arr_relative[:, -2]
    return metrics.precision_score(y, y_pred, sample_weight=sample_weight)
    
""""............................NN classifier........................................"""
    
def train_NN(antibody_list, record_name, batch_size, lr, NN_architecture=None,
              num_epochs=1000, test_size=0.1, num_shuffle=1, early_stopping=True, rolling_average_range=70, 
              DataLoader_num_workers=0, GPU_name="cuda", reload_from_checkpoint=False):
    """
    Train a neural network using Pytorch.
    Instead of returning a classifier, this function stores everything in 'metrics_NN/', 'tensorboard_NN/', 'Checkpoints_NN/' and 'Models/' folders. 
    
    Args:
        antibody_list: A list of antibody objects
        record_name: a string, the function will use this name to record everything in 'metrics_NN/', 'tensorboard_NN/', 'Checkpoints_NN/' and 'Models/' folders. 
        batch_size: the batch size for training process (3000 to 10000)
        lr: learning rate (0.0001 to 0.003)
        NN_architecture: a list of integer, e.g., [1280,100,100,2] creates a NN which has 1280 as the inputs size, two hiden layers with 100 nodes, 2 as the outputs size. The first number should be 1280 for ESM embedding. The last should be 2 for two class classifier.
        num_epochs: number of epochs
        test_size: percentage of data holding out for validation
        num_shuffle: e.g., if the num_shuffle is 5, the function will shuffle test_size data 5 times and train NN 5 times
        early_stopping: early stop or not. It is designed to avoid overfitting. If it's true, this function will calculate the F1 score on the test_size data every epoch and then calculate the averaged F1 score over "rolling_average_range" epoch. (For example, if rolling_average_range is 70, the average F1 will be calcualted every 70 epochs.) If the current averaged F1 score is lower than the last averaged F1 score, the training process stops.
        rolling_average_range: to define how many epochs averaged F1 is calcualted once.
        DataLoader_num_workers: To avoid blocking computation code with data loading, PyTorch provides an easy switch to perform multi-process data loading by simply setting the argument num_workers to a positive integer. However, it does not work well on skywalkers.
        GPU_name: GPU for training
        reload_from_checkpoint: whether reload a training process or not.
    """

    if not os.path.exists('metrics_NN/'):
        os.makedirs('metrics_NN/')
    if not os.path.exists('tensorboard_NN/'):
        os.makedirs('tensorboard_NN/')
    if not os.path.exists('Checkpoints_NN/'):
        os.makedirs('Checkpoints_NN/')   
    if not os.path.exists('Models/'):
        os.makedirs('Models/')   
        
    f = open('metrics_NN/'+record_name, 'w', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['F1','Precision','Recall'])
    
    device = torch.device(GPU_name)
    
    metrics = {}
    metrics['F1'] = F1Score(num_classes=2, average=None).to(device)
    metrics['Precision'] = Precision(num_classes=2, average=None).to(device)
    metrics['Recall'] = Recall(num_classes=2, average=None).to(device)
    metrics_names = ['F1', 'Precision', 'Recall']

    for i in range(num_shuffle):
        model = NeuralNet(NN_architecture)
        model = model.to(device)
        
        writer = SummaryWriter('tensorboard_NN/{}_r{}'.format(record_name,i+1))
        record_name_loop = '{}_r{}'.format(record_name, i+1)
        tensor = load_data(antibody_list)
        class_weight = cal_class_weight_tensor(tensor).to(device)
        tensor_train, tensor_test = train_test_split(tensor, test_size=test_size)
        del(tensor)
        test_sample_len = tensor_test.shape[0]
        
        trainingset = TrainingSet(tensor_train)
        Training_dataloader = DataLoader(dataset=trainingset, batch_size=batch_size, shuffle=True, num_workers=DataLoader_num_workers, drop_last=True)
        
        criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        if reload_from_checkpoint:
            loaded_checkpoint = torch.load("Checkpoints_NN/{}.pth".format(record_name_loop))
            reloaded_epoch = loaded_checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optim_state'])
        else:
            reloaded_epoch = 0
        
        # training loop
        n_step = len(Training_dataloader)
        running_f1_score = 0.0
        last_f1 = 0.0
        
        # Early stopping
        current_score = {}
        rolling_score = {}
        current_average_score = {}
        last_average_score = {}
        last_average_score['F1'] = 0.0
        for name in metrics_names:
            rolling_score[name] = 0.0
        
        # with torch.profiler.profile(
        #      schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        #      on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard/{}_profiler'.format(record_name_loop)),
        #      record_shapes=True
        #  ) as prof:

        for epoch in range(num_epochs):
            for i, (x, w, y) in enumerate(Training_dataloader):      
                x, w, y = x.to(device), w.to(device), y.to(device)
                #forward
                outputs = model(x)
                loss = criterion(outputs, y)
                loss = loss * w
                loss = loss.mean()
                
                #backwards
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs, 1)
                running_f1_score += metrics['F1'](predicted, y)[1].item()
                
                # prof.step()
            
            # Track on training
            current_f1 = running_f1_score / n_step
            writer.add_scalar('training F1', current_f1, epoch+reloaded_epoch)
            # if current_f1 + 0.1 < last_f1:
            #     print("Bad leanring process")
                # break
            # last_f1 = current_f1
            running_f1_score = 0.0
                               
            # Early stopping
            if early_stopping:
                for name in metrics_names:
                    current_score[name] = 0.0             
                for i in range(5):
                    shuffle_index = torch.randint(high=test_sample_len, size=(3000,))
                    shufflled_testset = tensor_test[shuffle_index].to(device)
                    outputs = model(shufflled_testset[:,:-2])
                    _, predicted = torch.max(outputs, 1)
                    for name in metrics_names:
                        current_score[name] += metrics[name](predicted, shufflled_testset[:,-1].long())[1].item()                    
                for name in metrics_names:
                    current_score[name] = current_score[name] /5 
                               
                writer.add_scalar('test F1', current_score['F1'], epoch+reloaded_epoch)
                for name in metrics_names:
                    rolling_score[name] += current_score[name]
                               
                if (epoch+1) % rolling_average_range == 0:
                    for name in metrics_names:
                        current_average_score[name] = rolling_score[name] / rolling_average_range
                    print(f'epoch {epoch+1}, test_F1 = {current_average_score["F1"]:.4f}')
                    
                    if current_average_score["F1"] < last_average_score["F1"]:
                        torch.save(checkpoint, "Checkpoints_NN/{}.pth".format(record_name_loop))
                        model.load_state_dict(checkpoint["model_state"])
                        torch.save(model, "Models/{}.pth".format(record_name_loop))
                        csv_writer.writerow([last_average_score['F1'],last_average_score['Precision'],last_average_score['Recall']]) 
                        del(model, optimizer)
                        break
                    else:
                        checkpoint = {
                               "epoch": epoch + reloaded_epoch,
                               "model_state": model.state_dict(),
                               "optim_state": optimizer.state_dict()
                               }
                        for name in metrics_names:
                            last_average_score[name] = current_average_score[name]                        
                    for name in metrics_names:
                        rolling_score[name] = 0.0
                        
    
    f.close()                           


def load_data(antibody_list):
    array = np.concatenate([i.arr_relative for i in antibody_list])
    tensor = torch.from_numpy(array).float()
    return tensor

def cal_class_weight_tensor(tensor):
    """Calculate class weights"""
    li = torch.bincount(tensor[:,-1].long(), weights=tensor[:,-2])
    li = li/torch.max(li)
    li = 1/li
    return li

class NeuralNet(nn.Module):
    """
    Args: 
        NN_architecture: a list of int, for example: [input_size, hidden_layer1_size, ..., output_size]
    """
    def __init__(self, NN_architecture):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(NN_architecture) - 1
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(NN_architecture[i], NN_architecture[i+1]))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for n, layer in enumerate(self.layers):
            if n == self.num_layers - 1:
                x = layer(x)
            else:
                x = layer(x)
                x = self.relu(x)
        return x
    
    
class TrainingSet(Dataset):
    def __init__(self, tensor):
        self.x = tensor[:,:-2]
        self.w = tensor[:,-2]
        self.y = tensor[:,-1].long()
        self.n_samples = tensor.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.w[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

def test_NN(model_name, antibody_list, map_location='cpu'):
    """Test a trained NN on an antibody list
    
    Args:
        model_name: the name of a trained NN model stored in Models/ dirctory.
        antibody_list: A list of antibody objects
    """
    metrics = {}
    metrics['F1'] = F1Score(num_classes=2, average=None)
    metrics['Precision'] = Precision(num_classes=2, average=None)
    metrics['Recall'] = Recall(num_classes=2, average=None)
    metrics_names = ['F1', 'Precision', 'Recall']
    
    model = torch.load('Models/{}'.format(model_name), map_location=map_location)
    model.eval()
    tensor = load_data(antibody_list)
    
    outputs = model(tensor[:,:-2])
    _, predicted = torch.max(outputs, 1)
    
    scores = {}
    for metric_name in metrics_names:
        scores[metric_name] =  metrics[metric_name](predicted, tensor[:,-1].long())[1].item()
        print("{}: {:.3f}".format(metric_name, scores[metric_name]))
    del(tensor, model)
    
def embeding_seqs(seqs):
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    
    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = seqs
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    mutants = torch.stack(sequence_representations[1:])
    parent = sequence_representations[0]
    delt = mutants - parent
    return delt
    