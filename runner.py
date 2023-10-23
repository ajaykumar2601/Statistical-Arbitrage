import sys
import yaml
sys.path.append(r'/Users/rana/Desktop/Ajay Kumar/Term 3/ZA/stats_arb_with excess returns/Boson/models') 


from train_test import test
import numpy as np
from train_test import train
import pandas as pd
from preprocess import preprocess_ou
from preprocess import preprocess_fourier
from preprocess import preprocess_cumsum
from OUFFN import OUFFN
from CNNTransformer import CNNTransformer
from FourierFFN import FourierFFN
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator

current_directory = os.getcwd()

base = "/Users/rana/Desktop/Ajay Kumar/Term 3/ZA/stats_arb_with excess returns/Boson/boson data/"


random_seed_list = [42]


ou_obj = OUFFN(logdir = current_directory,
                lookback = 30,
                random_seed=42, 
                device = "cpu",
                hidden_units = [4,4,4,4], 
                dropout = 0.01)


fft_obj = FourierFFN(logdir = current_directory,
  lookback = 30,
  random_seed=42, 
  device = "cpu",
  hidden_units = [30, 16, 8, 4], 
  dropout = 0.01)

cnn_obj = CNNTransformer(logdir = current_directory,
              random_seed = 42, 
              lookback = 30,
              device = "opencl", # other options for device are e.g. "cuda:0"
              normalization_conv = True, 
              filter_numbers = [1,8], 
              attention_heads = 4, 
              use_convolution = True,
              hidden_units = 2*8, 
              hidden_units_factor = 2,
              dropout = 0.01, 
              filter_size = 2, 
              use_transformer = True)

# logdir,
#              lookback = 30,
#              random_seed=0, 
#              device = "cpu",
#              hidden_units = [30, 16, 8, 4], 
#              dropout = 0.25)


c=sys.path.append('./configs')

#vzl = train(model=ou_obj, preprocess = preprocess_ou, data_train=np.array(data), log_dev_progress=False, device="cpu",
 #           output_path = current_directory)



    
excess_ret = pd.read_csv(base+"FX_Excess_Returns_final.csv",index_col=('Date')).rename_axis('Date')
excess_ret.index = excess_ret.index.rename("Date")
excess_ret.fillna(0,inplace=True)
excess_ret.index = pd.to_datetime(excess_ret.index)

# %%

epoch_num = 250

l = [1,3,5]
objectives = ['meanvar','sharpe']
model_tags = ['OUFFN','FourierFFN','CNNTransformer'
              ]
configs_list = ['ouffn-full.yaml','fourierffn-full.yaml','cnntransformer-full.yaml'
                ]
preprocess_list = [preprocess_ou,preprocess_fourier,
                   preprocess_cumsum]
model_list = [ou_obj,fft_obj,
              cnn_obj]

for idx,preprocess1 in enumerate(preprocess_list):
    
    print (f"starting {preprocess1}")
    model_tag1 = model_tags[idx]
    config1 = configs_list[idx]
    model1 = model_list[idx]
    
    with open(fr'/Users/rana/Desktop/Ajay Kumar/Term 3/ZA/stats_arb_with excess returns/Boson/configs/{config1}', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    for obj1 in objectives:
        print (f"starting{obj1}")
        for _ in l:
            print (f"starting factor {_}")
        
            data=pd.read_csv(base+f"FX_PCA_{_}_FACTOR_RESID_DAILY.csv",index_col=('Date')).rename_axis('Date')
            data.fillna(0,inplace=True)
            data.index = pd.to_datetime(data.index)
            
            factors_num = _
            
            vari = f'FX {obj1} With Cost {factors_num} factor'
            
            vzl_test = test(Data=np.array(data), 
                        daily_dates= list(data.index),
                        model=model1,
                        preprocess=preprocess1,
                        config= config,
                        #objective_secondary=None,
                        objective_secondary = "non information_ratio", # valid choices are "information_ratio" and anything else
                        excess_returns_data = np.array(excess_ret),
                        num_epochs=epoch_num,
                        device='cpu',
                        model_tag=model_tag1, batchsize=100,
                        trans_cost = 0.0005,
                        hold_cost = 0.0001,
                        vari = vari,
                        objective = obj1
                        #output_path =r"C:\Users\Rohit Chaturvedi\Documents\Goal 2022\Berkeley MFE\Courses\Term 3\230 ZA\Boson\logdir"
                        )
            
            
            vari = f'FX {obj1} With No Cost {factors_num} factor'
            
            vzl_test = test(Data=np.array(data), 
                        daily_dates= list(data.index),
                        model=model1,
                        preprocess=preprocess1,
                        config= config,
                        #objective_secondary=None,
                        objective_secondary = "non information_ratio", # valid choices are "information_ratio" and anything else
                        excess_returns_data = np.array(excess_ret),
                        num_epochs=epoch_num,
                        device='cpu',
                        model_tag=model_tag1, batchsize=100,
                        trans_cost = 0.000,
                        hold_cost = 0.000,
                        vari = vari,
                        objective = obj1
                        #output_path =r"C:\Users\Rohit Chaturvedi\Documents\Goal 2022\Berkeley MFE\Courses\Term 3\230 ZA\Boson\logdir"
                        )
        
        
 






 
# # %%
# # vzl_test_fft = test(Data=np.array(data),
# #                     daily_dates= list(data.index),
# #                     model=fft_obj,
# #                     preprocess=preprocess_fourier,
# #                     config= config,
# #                     objective_secondary = "Non ir",
# #                     excess_returns_data = np.array(excess_ret),
# #                     num_epochs=10,
# #                     device='cpu',
# #                     model_tag='FourierFFN',batchsize=50,
# #                     trans_cost = 0.0005,
# #                     hold_cost = 0.0002,
# #                     vari='fftffn'
# #                     #output_path =r"C:\Users\Rohit Chaturvedi\Documents\Goal 2022\Berkeley MFE\Courses\Term 3\230 ZA\Boson\logdir"
# #                     )

