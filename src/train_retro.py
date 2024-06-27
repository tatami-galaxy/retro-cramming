import argparse

from datasets import load_from_disk
from datasets import load_from_disk

from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
from retro_pytorch import RETRO
from training import TrainingWrapper

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

    wrapper = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    #documents_path = './text_folder',             # path to folder of text
    dataset = dataset,
    tokenizer_path = data_args.tokenizer_path,     # path to tokenizer
    #glob = '**/*.txt',                            # text glob
    chunks_memmap_path = './train.chunks.dat',     # path to chunks
    seqs_memmap_path = './train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    max_chunks = data_args.max_chunks,                        # maximum cap to chunks
    max_seqs = data_args.max_seqs,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = data_args.max_index_memory_usag,
    current_memory_available = data_args.current_memory_available,
)



def run():

    common_parser = argparse.ArgumentParser()
    training_parser = argparse.ArgumentParser()
    data_parser = argparse.ArgumentParser()

    common_parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    common_parser.add_argument(
        "--output_dir",
        default='./',
        type=str,
    )

    training_parser.add_argument(
        "--encoder_model",
        default='bert-base-uncased', 
    )
    training_parser.add_argument(
        "--mixed_precision", # choose from no, fp16, bf16 or fp8
        default='fp16',
        type=str,
    )
    training_parser.add_argument(
        "--per_device_train_batch_size", 
        default='16',
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

    data_parser.add_argument(
        "--dataset_path",
        default='/media/drdo/DATA/Datasets/wikipedia_20231101_en',
        type=str,
    )
    data_parser.add_argument(
        "--tokenizer_path",
        default='bert-base-uncased',
        type=str,
    )
    data_parser.add_argument(
        "--max_chunks",
        default=10000000,
        type=int,
    )
    data_parser.add_argument(
        "--max_seqs",
        default=1000000,
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
        project_dir=common_args.output_dir
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