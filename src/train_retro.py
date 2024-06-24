import torch
from retro_pytorch import RETRO
from training import TrainingWrapper

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
)

wrapper = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = './text_folder',              # path to folder of text
    glob = '**/*.txt',                             # text glob
    chunks_memmap_path = './train.chunks.dat',     # path to chunks
    seqs_memmap_path = './train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    max_chunks = 1_000_000,                        # maximum cap to chunks
    max_seqs = 100_000,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '100m',
    current_memory_available = '1G'
)