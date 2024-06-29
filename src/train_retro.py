import argparse

from datasets import load_from_disk
from datasets import load_from_disk

from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from retro_pytorch import RETRO
from training import TrainingWrapper

#import lovely_tensors as lt
#lt.monkey_patch()

#ds = load_dataset("wikimedia/wikipedia", "20231101.en")
#ds.save_to_disk('/media/drdo/DATA/Datasets/wikipedia_20231101_en')

def train(common_args, training_args, data_args, accelerator):

    retro = RETRO(
        max_seq_len = 2048,                      # max sequence length
        enc_dim = 896,                           # encoder model dimension
        enc_depth = 3,                           # encoder depth
        dec_dim = 768,                           # decoder model dimensions
        dec_depth = 12,                          # decoder depth
        dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
        heads = 8,                               # attention heads
        dim_head = 64,                           # dimension per head
        dec_attn_dropout = 0.25,                 # decoder attention dropout
        dec_ff_dropout = 0.25                    # decoder feedforward dropout
    )#.cuda()

    dataset_dict = load_from_disk(data_args.dataset_path)
    dataset = dataset_dict['train']

    processed_data_path = common_args.repo_dir+'/data/processed'

    wrapper = TrainingWrapper(
    retro = retro,       
    frozen_model_path = training_args.frozen_model_path,                      
    knn = training_args.knn,    # 2                       
    chunk_size = training_args.chunk_size,  # 64         
    #documents_path = './text_folder',           
    dataset = dataset,
    tokenizer_path = data_args.tokenizer_path,     
    #glob = '**/*.txt',                          
    chunks_memmap_path =  processed_data_path+'/train.chunks.dat',  # path to chunks
    seqs_memmap_path =  processed_data_path+'/train.seq.dat',   # path to sequence data
    # path to document ids per chunk (used for filtering neighbors belonging to same document)
    doc_ids_memmap_path =  processed_data_path+'/train.doc_ids.dat',  
    processed_stats_json_path = processed_data_path+'/processed-stats.json',
    faiss_index_filename = processed_data_path+'/knn.index',
    force_reprocess = data_args.force_reprocess,
    max_chunks = data_args.max_chunks,                   
    max_seqs = data_args.max_seqs,            
    knn_extra_neighbors = training_args.knn_extra_neighbors,   
    chunks_to_embeddings_batch_size = training_args.chunks_to_embeddings_batch_size,   
    max_index_memory_usage = data_args.max_index_memory_usage,
    current_memory_available = data_args.current_memory_available,
)



def run():

    common_parser = argparse.ArgumentParser()
    training_parser = argparse.ArgumentParser()
    data_parser = argparse.ArgumentParser()

    # common args
    common_parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    common_parser.add_argument(
        "--repo_dir",
        default='/home/drdo/retro-cramming',
        type=str,
    )


    # training args
    training_parser.add_argument(
        "--frozen_model_path",
        default='bert-base-cased', 
    )
    training_parser.add_argument(
        "--chunk_size", 
        default=64,
        type=int,
    )
    training_parser.add_argument(
        "--knn", 
        default=2,
        type=int,
    )
    training_parser.add_argument(
        "--knn_extra_neighbors", 
        default=100,
        type=int,
    )
    training_parser.add_argument(
        "--mixed_precision", # choose from no, fp16, bf16 or fp8
        default='fp16',
        type=str,
    )
    training_parser.add_argument(
        "--chunks_to_embeddings_batch_size", 
        default=16,
        type=int,
    )
    training_parser.add_argument(
        "--per_device_train_batch_size", 
        default=16,
        type=int,
    )
    training_parser.add_argument(
        "--train_steps",
        default='100000',
        type=int,
    )
    training_parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    training_parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )


    # data args
    data_parser.add_argument(
        "--dataset_path",
        default='/media/drdo/DATA/Datasets/wikipedia_20231101_en',
        type=str,
    )
    data_parser.add_argument(
        "--tokenizer_path",
        default='bert-base-cased',
        type=str,
    )
    data_parser.add_argument(
        "--max_chunks",
        default=1_000_000,
        type=int,
    )
    data_parser.add_argument(
        "--max_seqs",
        default=100_000,
        type=int,
    )
    data_parser.add_argument(
        "--max_index_memory_usage",
        default='100m',
        type=str,
    )
    data_parser.add_argument(
        "--current_memory_available",
        default='1G',
        type=str,
    )
    data_parser.add_argument(
        "--force_reprocess",
        action="store_true",
    )


    # parse args
    common_args = common_parser.parse_args()
    training_args = training_parser.parse_args()
    data_args = data_parser.parse_args()

    # set seed
    set_seed(common_args.seed)

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=common_args.repo_dir+'/src',
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": training_args.lr,
        "train_steps": training_args.train_steps,
        "seed": common_args.seed,
        "train_batch_size": training_args.per_device_train_batch_size,
    }
    # run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)


    # train function
    train(common_args, training_args, data_args, accelerator)

    # end logging
    #accelerator.end_training()


if __name__ == "__main__":

    run()