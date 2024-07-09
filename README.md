# retro-cramming

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

RETRO implementation under the cramming setup

#### Follow through steps

- Dataset
    - wikipedia dataset

- Init RETRO class (retro_pytorch)
    - Init encoder, decoder

- Init TrainingWrapper (training)
    - text_dataset_to_chunks_ (retrieval) 
        - doc_text_to_chunks_and_seq_indices
            - tokenize
            - chunk
            - 1 token overlap between adjacent chunks (chunk_size + 1) -> why?
        - store chunks, seqs, doc_ids for each chunk

- Faiss index
    - chunks_to_precalculated_knn_ (retrieval)
        - chunks_to_index_and_embed (retrieval)
            - chunks_to_embeddings_ (retrieval)
                add sos token to each chunk in batch -> not adding
            - memmap_file_to_chunks_ (retrieval)
                - partition embeddings to small files
            - index_embeddings (retrieval)
            - pre-compute knns
                - mask out any neighbors that belong to the same document
                - redo argmax

- RETRODataset for retrieving training seqs
    - Extend pytorch dataset
    - get dataloader over it
    - __getitem__ method of RETRODataset (data)
        - remove the last token, except for last token of last chunk -> why did we add it?
        - disallow having more than 1 document in a sequence, as it would break RETRO's CCA
        - knn_to_retrieved_chunks (data)
            - derive mask for no neighbors found (-1)
            - get neighbor and continuation chunks
            - use presence of [EOS] in chunk as way to detect document boundaries
            - combine neighbors with continuations
            - mask out any nearest neighbor chunks that was -1 (not found at index time) to padding id
            
- RETRO
    - how do rotary embeddings work?
    - forward()
        - decode()
