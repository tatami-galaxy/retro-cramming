from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import faiss
from autofaiss import build_index

from transformers import AutoConfig, AutoTokenizer, AutoModel

from utils import memmap, reset_folder_

# singleton globals
MODEL = None
TOKENIZER = None
SOS_ID = None
EOS_ID = None

# helper functions
def exists(val):
    return val is not None

def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr


# indexing helper functions
def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

def get_tokenizer(tokenizer_path):
    global TOKENIZER
    global SOS_ID
    global EOS_ID
    if not exists(TOKENIZER):
        #TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', tokenizer_path)
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
        SOS_ID = TOKENIZER.cls_token
        EOS_ID = TOKENIZER.sep_token
    return TOKENIZER


def get_bert(frozen_model_path):
    global MODEL
    if not exists(MODEL):
        #MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', frozen_model_path)
        MODEL = AutoModel.from_pretrained(frozen_model_path)
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL

# tokenize
def tokenize(texts, tokenizer_path, add_special_tokens = True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer(tokenizer_path)

    encoding = tokenizer(
        texts,
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )

    token_ids = encoding.input_ids
    return token_ids

# text to chunks

def doc_text_to_chunks_and_seq_indices(
    *,
    doc_text,
    tokenizer_path,
    seq_len,
    chunk_size,
    pad_id,
):
    assert (seq_len % chunk_size) == 0, 'sequence length must be divisible by chunk size'

    # 1 x total text length
    ids = tokenize(doc_text, tokenizer_path)
    # has sos and eos tokens (cls and sep for bert)
    ids = rearrange(ids, '1 ... -> ...')

    text_len = ids.shape[-1]

    # pad to multiple of chunk size with an extra token
    padding = chunk_size - ((text_len - 1) % chunk_size)
    ids = F.pad(ids, (pad_id, padding))

    # split out very last token (of last chunk)
    # this is a pad token if n not a multiple of chunk_size
    ids, last_token = ids[:-1], ids[-1:]

    # rearrange all tokens into chunks
    ids = rearrange(ids, '(n c) -> n c', c = chunk_size)

    # first tokens of chunk [2:] and on will become the last token of chunk [1:]
    last_token_per_chunk = ids[1:, 0]
    all_last_tokens = torch.cat((last_token_per_chunk, last_token), dim = 0)
    all_last_tokens = rearrange(all_last_tokens, 'n -> n 1')

    # append all last tokens to ids for (num_chunks, chunk_size + 1) 
    # chunk size = l+1
    chunks_with_extra_token = torch.cat((ids, all_last_tokens), dim = -1)

    # calculate chunk indices starting at 0, spaced number of chunks of seq len apart
    total_chunks = ids.shape[0]
    num_chunks_per_seq = seq_len // chunk_size
    # seq starting points in doc
    seq = torch.arange(0, total_chunks, num_chunks_per_seq)

    return chunks_with_extra_token, seq


def text_dataset_to_chunks_(
    *,
    #folder,
    dataset,
    tokenizer_path,
    chunks_memmap_path,
    seqs_memmap_path,
    doc_ids_memmap_path,
    max_chunks,
    max_seqs,
    chunk_size,
    pad_id,
    seq_len,
    #glob = '**/*.txt',
):
    #paths = sorted([*Path(folder).glob(glob)])

    total_chunks = 0
    total_docs = 0
    total_seqs = 0

    chunks_shape = (max_chunks, chunk_size + 1)
    seqs_shape = (max_seqs,)
    doc_ids_shape = (max_chunks,)

    with memmap(chunks_memmap_path, shape = chunks_shape, dtype = np.int32, mode = 'w+') as chunks_memmap\
        , memmap(seqs_memmap_path, shape = seqs_shape, dtype = np.int32, mode = 'w+') as seqs_memmap\
        , memmap(doc_ids_memmap_path, shape = doc_ids_shape, dtype = np.int32, mode = 'w+') as doc_ids_memmap:

        bar = tqdm(range(max_chunks), desc="Text dataset to token chunks")  

        for example in dataset:

            chunks, seq = doc_text_to_chunks_and_seq_indices(
                doc_text = example['text'],
                tokenizer_path = tokenizer_path,
                chunk_size = chunk_size,
                seq_len = seq_len,
                pad_id = pad_id,
            )

            # how many chunks in doc
            doc_chunk_len = chunks.shape[0]
            # how many seqs in doc
            doc_seq_len = seq.shape[0]

            # if max chunks reached stop
            if total_chunks + doc_chunk_len > max_chunks:
                break
            # if max seqs reached stop
            if total_seqs + doc_seq_len > max_seqs:
                break

            # store chunks, seqs, doc_ids
            # chunks (token ids), bound by max_chunks
            chunks_memmap[total_chunks: (total_chunks + doc_chunk_len)] = chunks.numpy() 
            # seq starting positions in corpus (all docs)
            # bound by max_seqs
            seqs_memmap[total_seqs: (total_seqs + doc_seq_len)] = seq.numpy() + total_chunks
            # doc id for each chunk
            doc_ids_memmap[total_chunks: (total_chunks + doc_chunk_len)] = np.full((doc_chunk_len,), total_docs)

            total_chunks += doc_chunk_len
            total_seqs += doc_seq_len
            total_docs += 1

            bar.update(doc_chunk_len)


    return dict(
        chunks = total_chunks,
        docs = total_docs,
        seqs = total_seqs
    )


# embedding function
@torch.no_grad()
def bert_embed(
    frozen_model_path,
    token_ids,
    return_cls_repr = False,
    eps = 1e-8,
    pad_id = 0.
):
    model = get_bert(frozen_model_path)
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    # b x chunk_size x dim
    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]               # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim = 1)

    mask = mask[:, 1:]                          # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, 'b n -> b n 1')

    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    masked_mean =  numer / (denom + eps)
    return masked_mean


# chunks to knn
# stored in mmaped file
def chunks_to_embeddings_(
    *,
    frozen_model_path,
    num_chunks,
    chunks_memmap_path,
    embeddings_memmap_path,
    embed_dim,
    chunk_size,
    batch_size,
    use_cls_repr = False,
    pad_id = 0.
):
    
    chunks_shape = (num_chunks, chunk_size + 1)
    embed_shape = (num_chunks, embed_dim)

    bar = tqdm(range(num_chunks), desc="Chunks to embeddings")

    with memmap(chunks_memmap_path, shape = chunks_shape, dtype = np.int32) as chunks\
        , memmap(embeddings_memmap_path, shape = embed_shape, dtype = np.float32, mode = 'w+') as embeddings:

        for dim_slice in range_chunked(num_chunks, batch_size = batch_size):

            # b x chunk_size+1
            batch_chunk_npy = chunks[dim_slice]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            # add sos token to each chunk in batch -> why?
            #cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            #batch_chunk = torch.cat((cls_tokens, batch_chunk), dim = 1)

            # omit last token, the first token of the next chunk
            # used for autoregressive training
            # not needed during embedding computation
            # b x chunk_size
            batch_chunk = batch_chunk[:, :-1]

            # b x dim
            batch_embed = bert_embed(
                frozen_model_path,
                batch_chunk,
                return_cls_repr = use_cls_repr
            )

            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
            #print(f'embedded {dim_slice.stop} / {num_chunks}')
            bar.update(dim_slice.stop - dim_slice.start)


def memmap_file_to_chunks_(
    memmap_path,
    *,
    folder,
    shape,
    dtype,
    max_rows_per_file = 500
):
    rows, _ = shape

    bar = tqdm(range(shape[0]), desc="Splitting embedding file into smaller chunks")

    with memmap(memmap_path, shape = shape, dtype = dtype, mode = 'r') as f:
        reset_folder_(folder)

        for ind, dim_slice in enumerate(range_chunked(rows, batch_size = max_rows_per_file)):
            filename = folder +'/'+ f'{ind:05d}.npy'
            data_slice = f[dim_slice]

            np.save(str(filename), f[dim_slice])
            #print(f'saved {str(filename)}')
            bar.update(dim_slice.stop - dim_slice.start)


def index_embeddings(
    embeddings_folder,
    *,
    index_file,
    index_infos_file,
    max_index_memory_usage,
    current_memory_available,
):

    build_index(
        embeddings = embeddings_folder,
        index_path = index_file,
        index_infos_path = index_infos_file,
        metric_type = "l2",
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available,
        make_direct_map = True,
        should_be_memory_mappable = False,
        use_gpu = torch.cuda.is_available(),
    )

    index = faiss_read_index(index_file)
    return index


def chunks_to_index_and_embed(
    *,
    frozen_model_path,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    index_file,
    index_infos_file,
    embeddings_folder,
    max_index_memory_usage,
    current_memory_available,
    max_rows_per_file,
    chunks_to_embeddings_batch_size,
    use_cls_repr = False,
    **index_kwargs
):
    # get embed_dim
    config = AutoConfig.from_pretrained(frozen_model_path)
    embed_dim = config.hidden_size

    embedding_path = f'{chunk_memmap_path}.embedded'
    embed_shape = (num_chunks, embed_dim)

    chunks_to_embeddings_(
        frozen_model_path = frozen_model_path,
        num_chunks = num_chunks,
        chunk_size = chunk_size,
        chunks_memmap_path = chunk_memmap_path,
        embeddings_memmap_path = embedding_path,
        use_cls_repr = use_cls_repr,
        batch_size = chunks_to_embeddings_batch_size,
        embed_dim = embed_dim
    )
    # split embedding file into smaller chunks (files)
    memmap_file_to_chunks_(
        embedding_path,
        shape = embed_shape,
        dtype = np.float32,
        folder = embeddings_folder,
        max_rows_per_file = max_rows_per_file
    )
    # build faiss index
    index = index_embeddings(
        embeddings_folder = embeddings_folder,
        index_file = index_file,
        index_infos_file = index_infos_file,
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available,
        **index_kwargs
    )

    embeddings = np.memmap(embedding_path, shape = embed_shape, dtype = np.float32, mode = 'r')
    return index, embeddings


def chunks_to_precalculated_knn_(
    *,
    frozen_model_path,
    num_nearest_neighbors,  # 2
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    doc_ids_memmap_path,
    embeddings_folder,
    index_file,
    index_infos_file,
    max_index_memory_usage,
    current_memory_available,
    max_rows_per_file,
    chunks_to_embeddings_batch_size,
    num_extra_neighbors,    # 100
    force_reprocess,
    use_cls_repr = False,
    **index_kwargs
):
    
    chunk_path = Path(chunk_memmap_path)
    knn_path = chunk_path.parents[0] / f'{chunk_path.stem}.knn{chunk_path.suffix}'

    # early return knn path and faiss index
    # unless if force_reprocess is True
    if index_file is not None and knn_path.exists() and not force_reprocess:
        print(f'preprocessed knn found at {str(knn_path)}, faiss index reconstituted from {str(index_file)}')
        index = faiss_read_index(index_file)
        return knn_path, index

    # fetch the faiss index and calculated embeddings for the chunks
    index, embeddings = chunks_to_index_and_embed(
        frozen_model_path = frozen_model_path,
        num_chunks = num_chunks,
        chunk_size = chunk_size,
        chunk_memmap_path = chunk_memmap_path,
        index_file = index_file,
        index_infos_file = index_infos_file,
        chunks_to_embeddings_batch_size = chunks_to_embeddings_batch_size,
        embeddings_folder = embeddings_folder,
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available,
        max_rows_per_file = max_rows_per_file,
        **index_kwargs
    )

    # pre-compute knns for training set
    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1  # why +1?

    bar = tqdm(range(num_chunks), desc="Calculating knns")

    with memmap(knn_path, shape = (num_chunks, num_nearest_neighbors), dtype = np.int32, mode = 'w+') as knns\
        , memmap(doc_ids_memmap_path, shape = (num_chunks,), dtype = np.int32, mode = 'r') as doc_ids:

        for dim_slice in range_chunked(num_chunks, batch_size = max_rows_per_file):

            # max_rows_per_file x embed_dim
            query_vector = embeddings[dim_slice]

            distances, indices = index.search(query_vector, k = total_neighbors_to_fetch)

            # remove self from distances and indices
            distances = distances[:, 1:]
            indices = indices[:, 1:]

            # mask out any neighbors that belong to the same document to -1
            query_doc_ids = doc_ids[dim_slice]
            neighbor_doc_ids = doc_ids[indices]
            neighbor_from_same_doc = query_doc_ids[..., None] == neighbor_doc_ids

            # np.where example : 
            # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            # np.where(a < 5, a, 10*a)
            # array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
            indices = np.where(neighbor_from_same_doc, -1, indices)
            distances = np.where(neighbor_from_same_doc, 1e3, distances)    # large distance for same doc (1000)

            # re-sort indices by updated distances
            # since some of the neighbors were potentially removed for being in the same doc
            indices = np.take_along_axis(indices, np.argsort(distances, axis = 1), axis = 1)

            # store nearest neighbors to knn memmap
            knns[dim_slice] = indices[:, :num_nearest_neighbors]

            bar.update(dim_slice.stop - dim_slice.start)

    print(f'knns saved to {knn_path}')
    return knn_path, index
