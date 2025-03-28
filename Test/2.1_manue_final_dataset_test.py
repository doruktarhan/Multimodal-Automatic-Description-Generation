#testscript.py
import os
import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.custom_dataloader import CustomDataLoader
from Data.dataset import RealEstateDataset, collate_fn
from Data.preprocessor import Preprocessor
from Model.model_utils import load_tokenizer
import torch
from torch.utils.data import DataLoader
import json

data_path = "Data/funda_scrapped_amsterdam_sample.json"
model_name = "Qwen/Qwen2.5-3B-Instruct"

#stream batch size and train batch size

stream_batch_size = 10
train_batch_size = 5

#custom data loader from efficient streaming
custom_loader = CustomDataLoader(data_path)
#load the preprocessor and the tokenizer
preprocessor = Preprocessor()
tokenizer = load_tokenizer(model_name)
#max input and output length for the tokenizer
max_input_length: int = 32768
max_output_length: int = 32768
#streaming the data 
for stream_batch in custom_loader.stream_data(stream_batch_size):
    print(len(stream_batch))
    #Creating dataset and preprocessing the examples for stream batch
    dataset = RealEstateDataset.from_preprocessor( 
        stream_batch,
        preprocessor,
        tokenizer,
        max_input_length,
        max_output_length
    )

    #create a dataloader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    #loop over the dataloader batches
    for batch in dataloader:
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['labels'].shape)
        break



    



