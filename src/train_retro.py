from dataclasses import dataclass, field
from typing import List

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
        metadata={"help": "frozen encoder model"},
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
    dim_head: int = field(default=64) 
    dec_attn_dropout: float = field(default=0.25)
    dec_ff_dropout: float = field(default=0.25)  
    
    # chunk and knn
    chunk_size: str = field(default=64)
    knn: int = field(default=2)
    knn_extra_neighbors: int = field(default=100)
    chunks_to_embeddings_batch_size: int = field(default=16)

    # other training args
    mixed_precision: str = field(
        default="fp16",
        metadata={"help": "choose from : ['no', 'fp16', 'bf16', 'fp8']"}
    )
    per_device_train_batch_size: int = field(default=2)
    train_steps: int = field(default=100000)
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
    max_chunks: int = field(default=1_000_000)
    max_seqs: int = field(default=100_000)
    max_index_memory_usage: str = field(default='100m')
    current_memory_available: str = field(default='1G')
    force_reprocess: bool = field(default=False)



def train(common_args, training_args, data_args, accelerator):

    retro = RETRO(
        max_seq_len = training_args.max_seq_len_retro,  
        enc_dim = training_args.enc_dim,
        enc_depth = training_args.enc_depth,
        dec_dim = training_args.dec_dim,
        dec_depth = training_args.dec_depth,
        dec_cross_attn_layers = training_args.dec_cross_attn_layers,
        heads = training_args.heads,
        dim_head = training_args.dim_head,
        dec_attn_dropout = training_args.dec_attn_dropout,
        dec_ff_dropout = training_args.dec_ff_dropout,
    ).cuda()

    dataset_dict = load_from_disk(data_args.dataset_path)
    dataset = dataset_dict['train']

    processed_data_path = common_args.repo_dir+'/data/processed'

    wrapper = TrainingWrapper(
        retro = retro,       
        frozen_model_path = training_args.frozen_model_path,                      
        knn = training_args.knn,    # 2           
        knn_extra_neighbors = training_args.knn_extra_neighbors,            
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
        index_infos_file = processed_data_path+'index_infos.json',
        force_reprocess = data_args.force_reprocess,
        max_chunks = data_args.max_chunks,                   
        max_seqs = data_args.max_seqs,             
        chunks_to_embeddings_batch_size = training_args.chunks_to_embeddings_batch_size,   
        embeddings_folder = processed_data_path+'/embeddings',

        max_index_memory_usage = data_args.max_index_memory_usage,
        current_memory_available = data_args.current_memory_available,
    )


    # get the dataloader and optimizer (AdamW with all the correct settings)
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_dl = iter(wrapper.get_dataloader(
        batch_size = training_args.per_device_train_batch_size,
        shuffle = True
    ))
    optim = wrapper.get_optimizer(lr = training_args.lr, wd = training_args.wd)

    # now do your training
    # ex. one gradient step

    # seq       - (2, 2049)         - 1 extra token since split by seq[:, :-1], seq[:, 1:]
    # retrieved - (2, 32, 2, 128)   - 128 since chunk + continuation, each 64 tokens
    seq, retrieved = map(lambda t: t.cuda(), next(train_dl))

    loss = retro(
        seq,
        retrieved,
        return_loss = True
    )

    # one gradient step
    loss.backward()
    optim.step()
    optim.zero_grad()


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