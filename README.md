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
        - store chunks, seqs, doc_ids for each chunk
- Faiss index
    - chunks_to_precalculated_knn_ (retrieval)
        - chunks_to_index_and_embed (retrieval)
            - chunks_to_embeddings_ (retrieval)
            - memmap_file_to_chunks_ (retrieval)
                - partition embeddings to small files
            - index_embeddings (retrieval)
            - pre-compute knns
- RETRODataset for retrieving training seqs
    - Extend pytorch dataset
    - get dataloader over it
    - __getitem__ method of RETRODataset (data)