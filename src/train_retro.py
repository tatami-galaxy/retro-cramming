from dataclasses import dataclass, field
import typing
from typing import List
from tqdm.auto import tqdm
import timeit

from datasets import load_from_disk
from datasets import load_from_disk

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import HfArgumentParser

import torch
from retro_pytorch import RETRO
from training import TrainingWrapper

#import lovely_tensors as lt
#lt.monkey_patch()

#ds = load_dataset("wikimedia/wikipedia", "20231101.en")
#ds.save_to_disk('/media/drdo/DATA/Datasets/wikipedia_20231101_en')


@dataclass
class CommonArguments:

    seed: int = field(default=2)
    repo_dir: str = field(default='/home/drdo/retro-cramming')


@dataclass
class TrainingArguments:

    # frozen encoder
    frozen_model_path: str = field(
        default="bert-base-cased",
        metadata={"help": "frozen encoder model. If changing model from bert make sure to update special tokens"},
    )

    # retro
    max_seq_len_retro: int = field(default=2048)
    enc_dim: int = field(default=896)          
    enc_depth: int = field(default=3)                         
    dec_dim: int = field(default=768)                           
    dec_depth: int = field(default=12)
    # decoder cross attention layers (with causal chunk cross attention)   
    dec_cross_attn_layers: List[int] = field(default_factory=lambda: [1, 3, 6, 9])
    heads: int = field(default=8) 
    dim_head: int = field(default=64, metadata={"help": "must be greater than 32"})   # MIN_DIM_HEAD = 32
    dec_attn_dropout: float = field(default=0.25)   # 0.
    dec_ff_dropout: float = field(default=0.25) # 0.
    enc_attn_dropout: float = field(default=0.25)   # 0.
    enc_ff_dropout: float = field(default=0.25) # 0.   
    norm_klass: typing.Any = field(default=None, metadata={"help": "RMSNorm by default. Change in retro_pytorch.py"})  
    gated_rmsnorm: bool = field(default=False)
    use_deepnet: bool = field(default=False)
    
    # chunk and knn
    chunk_size: str = field(default=64)
    knn: int = field(default=2)
    knn_extra_neighbors: int = field(default=100, metadata={"help": "some neighbors will get removed (-1) for being in the same doc"})
    chunks_to_embeddings_batch_size: int = field(default=16)
    
    # other training args
    mixed_precision: str = field(
        default="fp16",
        metadata={"help": "choose from : ['no', 'fp16', 'bf16', 'fp8']"}
    )
    per_device_train_batch_size: int = field(default=2)
    train_time_in_hours: int = field(default=24)
    gradient_accumulation_steps: int = field(default=1)
    lr: float = field(default=3e-4)
    wd: float = field(default=1e-1)

    def __post_init__(self):
        pass


@dataclass
class DataArguments:

    dataset_path: str = field(
        default='/media/drdo/DATA/Datasets/wikipedia_20231101_en',
    )
    tokenizer_path: str = field(default='bert-base-cased')
    max_chunks: int = field(default=10_000_000) # 10m chunks
    max_seqs: int = field(default=1_000_000)    # 1m seqs
    max_index_memory_usage: str = field(default='100m')
    current_memory_available: str = field(default='1G')
    force_reprocess: bool = field(default=False)
    max_rows_per_file: int = field(default=500, metadata={"help": "max rows per file for embedding files"})
    pad_id: int = field(default=0)
    add_continuations: bool = field(default=True, metadata={"help": "whether to retrieve continuations of chunks"})


def train(common_args, training_args, data_args, accelerator):

    retro = RETRO(
        frozen_model_path = training_args.frozen_model_path,
        max_seq_len = training_args.max_seq_len_retro,  
        enc_dim = training_args.enc_dim,
        enc_depth = training_args.enc_depth,
        dec_dim = training_args.dec_dim,
        dec_depth = training_args.dec_depth,
        dec_cross_attn_layers = training_args.dec_cross_attn_layers,
        heads = training_args.heads,
        dim_head = training_args.dim_head,
        enc_attn_dropout = training_args.enc_attn_dropout,
        enc_ff_dropout = training_args.enc_ff_dropout,
        dec_attn_dropout = training_args.dec_attn_dropout,
        dec_ff_dropout = training_args.dec_ff_dropout,
        chunk_size = training_args.chunk_size,
        pad_id = data_args.pad_id,
        norm_klass = training_args.norm_klass,
        gated_rmsnorm = training_args.gated_rmsnorm,
        use_deepnet = training_args.use_deepnet,
    )#.cuda()

    dataset_dict = load_from_disk(data_args.dataset_path)
    dataset = dataset_dict['train']

    processed_data_path = common_args.repo_dir+'/data/processed'

    wrapper = TrainingWrapper(

        # retro and encoder #
        retro = retro,       
        frozen_model_path = training_args.frozen_model_path,                      
                
        # data #          
        dataset = dataset,
        tokenizer_path = data_args.tokenizer_path,         
        chunk_size = training_args.chunk_size,  # 64                   
        chunks_memmap_path =  processed_data_path+'/train.chunks.dat',  # path to chunks
        seqs_memmap_path =  processed_data_path+'/train.seq.dat',   # path to sequence data
        # path to document ids per chunk (used for filtering neighbors belonging to same document)
        doc_ids_memmap_path =  processed_data_path+'/train.doc_ids.dat',  
        processed_stats_json_path = processed_data_path+'/processed-stats.json',
        max_chunks = data_args.max_chunks,                   
        max_seqs = data_args.max_seqs,    
        pad_id = data_args.pad_id,
        add_continuations = data_args.add_continuations,        
        

        # faiss and embeddings #
        chunks_to_embeddings_batch_size = training_args.chunks_to_embeddings_batch_size,   
        embeddings_folder = processed_data_path+'/embeddings',
        max_rows_per_file = data_args.max_rows_per_file,
        faiss_index_filename = processed_data_path+'/knn.index',
        index_infos_file = processed_data_path+'index_infos.json',
        max_index_memory_usage = data_args.max_index_memory_usage,
        current_memory_available = data_args.current_memory_available,

        # knn #
        knn = training_args.knn,    # 2           
        knn_extra_neighbors = training_args.knn_extra_neighbors, 

        # others #
        force_reprocess = data_args.force_reprocess,

    )


    # get the dataloader and optimizer (AdamW with all the correct settings)
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    #train_dl = iter(wrapper.get_dataloader(
    train_dl = wrapper.get_dataloader(
        batch_size = training_args.per_device_train_batch_size,
        shuffle = True
    )
    optim = wrapper.get_optimizer(lr = training_args.lr, wd = training_args.wd)

    # now do your training
    # ex. one gradient step

    # prepare everything for accelerator
    retro, optim, train_dl = accelerator.prepare(retro, optim, train_dl)

    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval
    total_time_elapsed = 0
    new_time = 0
    train_time_in_secs = training_args.train_time_in_hours*60*60

    progress_bar = tqdm(range(global_step, train_time_in_secs), disable=not accelerator.is_main_process, position=0)

    while True:

        retro.train()
        # seq : b x max_seq_len_retro + 1)  -> 1 extra token since split by seq[:, :-1], seq[:, 1:] for autoregression
        # retrieved : b x seq_num_chunks (32) x knn (2) x (2*chunk_size) -> 128 since chunk + continuation, each 64 tokens
        #seq, retrieved = map(lambda t: t.cuda(), next(train_dl))
        for seq, retrieved in train_dl:

            # get start time
            step_start_time = timeit.default_timer()

            with accelerator.accumulate(retro):
                loss = retro(seq,retrieved)
                accelerator.backward(loss)
                optim.step()
                optim.zero_grad()

            # get time elapsed
            elapsed = timeit.default_timer() - step_start_time
            new_time = new_time + elapsed

            # checks if the accelerator has performed an optimization step behind the scenes
            # update progress bar
            if accelerator.sync_gradients:
                if new_time - total_time_elapsed >= 1:
                    progress_bar.update(int(new_time) - int(total_time_elapsed))
                    total_time_elapsed = new_time

            #TODO : eval, need to calc time elapsed after eval

            #TODO : save model

            if int(total_time_elapsed) >= train_time_in_secs:
                return
        


def run():

    # parse cl arguments
    parser = HfArgumentParser((CommonArguments, TrainingArguments, DataArguments))
    common_args, training_args, data_args = parser.parse_args_into_dataclasses()

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
        "train_time_in_hours": training_args.train_time_in_hours,
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