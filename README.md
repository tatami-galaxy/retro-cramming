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
    - text_dataset_to_chunks_ (retrieval)  -> need to parallelize
        - doc_text_to_chunks_and_seq_indices
            - tokenize
            - chunk
        - store chunks, seqs, doc_ids for each chunk
