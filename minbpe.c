#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INITIAL_VOCAB_SIZE 256
#define MAX_TEXT_SIZE 1024

// Define the structures
typedef struct {
    int first;
    int second;
} IntPair;

typedef struct {
    IntPair pair;
    int idx;
} Merge;

typedef struct {
    Merge *merges;
    size_t num_merges;
    unsigned char **vocab;
    size_t vocab_size;
} BasicTokenizer;


BasicTokenizer* create_tokenizer();
void clean_tokenizer(BasicTokenizer *tokenizer);
void train(BasicTokenizer *tokenizer, const char *text, size_t vocab_size, int verbose);
void encode(BasicTokenizer *tokenizer, const char *text, int *ids, size_t *ids_size);
void decode(const BasicTokenizer *tokenizer, const int *ids, size_t ids_size, char *text);
size_t find_pair_index(Merge *merges, size_t merges_size, IntPair pair);
void token_counts(const int *ids, size_t ids_size, size_t *pair_counts, size_t *pair_counts_size);
void merge(int *ids, size_t *ids_size, IntPair pair, int idx);


/*
* @brief creates a new BasicTokenizer.
*
* Initializes a new BasicTokenizer with an empty merge list and a vocabulary
* containing the first 256 ASCII characters.
*
* @return A pointer to the newly created BasicTokenizer, or NULL if allocation fails.
*/
BasicTokenizer* create_tokenizer() {
    BasicTokenizer *tokenizer = (BasicTokenizer*)malloc(sizeof(BasicTokenizer));
    tokenizer->merges = NULL;
    tokenizer->num_merges = 0;
    tokenizer->vocab = (unsigned char**)malloc(INITIAL_VOCAB_SIZE * sizeof(unsigned char*));
    for (int i = 0; i < INITIAL_VOCAB_SIZE; ++i) {
        tokenizer->vocab[i] = (unsigned char*)malloc(sizeof(unsigned char));
        tokenizer->vocab[i][0] = i;
    }
    tokenizer->vocab_size = INITIAL_VOCAB_SIZE;
    return tokenizer;
}

/*
* @brief Frees all resources associated with a BasicTokenizer.
*
* @param tokenizer Pointer to the BasicTokenizer to be cleaned up.
*/
void clean_tokenizer(BasicTokenizer *tokenizer) {
    for (size_t i = 0; i < tokenizer->vocab_size; ++i) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->merges);
    free(tokenizer);
}

/*
* @brief the tokenizer on the given text.
*
* Performs byte pair encoding (BPE) on the input text to learn merges
* and expand the vocabulary up to the specified size.
*
* @param tokenizer Pointer to the BasicTokenizer to be trained.
* @param text The input text to train on.
* @param vocab_size The desired final vocabulary size.
* @param verbose If non-zero, print progress information during training.
*/
void train(BasicTokenizer *tokenizer, const char *text, size_t vocab_size, int verbose) {
    size_t num_merges = vocab_size - INITIAL_VOCAB_SIZE;
    size_t text_size = strlen(text);
    int *ids = (int*)malloc(text_size * sizeof(int));
    for (size_t i = 0; i < text_size; ++i) {
        ids[i] = (unsigned char)text[i];
    }

    tokenizer->merges = (Merge*)malloc(num_merges * sizeof(Merge));

    for (size_t i = 0; i < num_merges; ++i) {
        size_t pair_counts[MAX_TEXT_SIZE * 3];
        size_t pair_counts_size;
        token_counts(ids, text_size, pair_counts, &pair_counts_size);

        size_t max_count = 0;
        IntPair best_pair = { 0, 0 };
        for (size_t j = 0; j < pair_counts_size * 3; j += 3) {
            IntPair pair = { pair_counts[j], pair_counts[j+1] };
            size_t count = pair_counts[j+2];
            if (count > max_count) {
                max_count = count;
                best_pair = pair;
            }
        }

        if (max_count == 0) {
            break; // No more pairs to merge
        }

        int idx = INITIAL_VOCAB_SIZE + i;
        merge(ids, &text_size, best_pair, idx);
        tokenizer->merges[i] = (Merge){ best_pair, idx };

        tokenizer->vocab = (unsigned char**)realloc(tokenizer->vocab, (idx + 1) * sizeof(unsigned char*));
        tokenizer->vocab[idx] = (unsigned char*)malloc(2 * sizeof(unsigned char));
        tokenizer->vocab[idx][0] = best_pair.first;
        tokenizer->vocab[idx][1] = best_pair.second;

        if (verbose) {
            printf("Merge %zu/%zu: (%d, %d) -> %d\n", i + 1, num_merges, best_pair.first, best_pair.second, idx);
        }
    }

    free(ids);
}

/*
* @brief Encodes the given text into token IDs using the trained tokenizer.
*
* @param tokenizer Pointer to the trained BasicTokenizer.
* @param text The input text to encode.
* @param ids Output array to store the resulting token IDs.
* @param ids_size Pointer to store the number of token IDs generated.
*/
void encode(BasicTokenizer *tokenizer, const char *text, int *ids, size_t *ids_size) {
    size_t text_size = strlen(text);
    *ids_size = text_size;
    for (size_t i = 0; i < text_size; ++i) {
        ids[i] = (unsigned char)text[i];
    }

    while (*ids_size >= 2) {
        size_t pair_counts[MAX_TEXT_SIZE * 3];
        size_t pair_counts_size;
        token_counts(ids, *ids_size, pair_counts, &pair_counts_size);

        IntPair best_pair = { -1, -1 };
        size_t best_idx = 0;
        size_t min_merge_idx = SIZE_MAX;

        for (size_t i = 0; i < pair_counts_size * 3; i += 3) {
            IntPair pair = { pair_counts[i], pair_counts[i+1] };
            size_t idx = find_pair_index(tokenizer->merges, tokenizer->num_merges, pair);
            if (idx < tokenizer->num_merges && idx < min_merge_idx) {
                min_merge_idx = idx;
                best_pair = pair;
                best_idx = idx;
            }
        }

        if (best_pair.first == -1) {
            break;
        }

        merge(ids, ids_size, best_pair, tokenizer->merges[best_idx].idx);
    }
}

/*
* @brief Decodes a list of token IDs back into text.
*
* @param tokenizer Pointer to the BasicTokenizer used for decoding.
* @param ids Array of token IDs to decode.
* @param ids_size Number of token IDs in the array.
* @param text Output buffer to store the decoded text.
*/
void decode(const BasicTokenizer *tokenizer, const int *ids, size_t ids_size, char *text) {
    for (size_t i = 0; i < ids_size; ++i) {
        text[i] = tokenizer->vocab[ids[i]][0];
    }
    text[ids_size] = '\0';
}


/* @brief Decodes a list of token IDs back into text.
*
* @param tokenizer Pointer to the BasicTokenizer used for decoding.
* @param ids Array of token IDs to decode.
* @param ids_size Number of token IDs in the array.
* @param text Output buffer to store the decoded text.
*/
size_t find_pair_index(Merge *merges, size_t merges_size, IntPair pair) {
    for (size_t i = 0; i < merges_size; ++i) {
        if (merges[i].pair.first == pair.first && merges[i].pair.second == pair.second) {
            return i;
        }
    }
    return merges_size;
}

/*
* @brief Counts the frequencies of consecutive token pairs in the given ID sequence.
*
* @param ids Array of token IDs.
* @param ids_size Number of token IDs in the array.
* @param pair_counts Output array to store pair counts (format: [first, second, count, ...]).
* @param pair_counts_size Pointer to store the number of unique pairs found.
*/
void token_counts(const int *ids, size_t ids_size, size_t *pair_counts, size_t *pair_counts_size) {
    *pair_counts_size = 0;
    for (size_t i = 0; i < ids_size - 1; ++i) {
        IntPair pair = { ids[i], ids[i + 1] };
        size_t index = find_pair_index((Merge *)pair_counts, *pair_counts_size, pair);
        if (index == *pair_counts_size) {
            // New pair
            pair_counts[*pair_counts_size * 3] = pair.first;
            pair_counts[*pair_counts_size * 3 + 1] = pair.second;
            pair_counts[*pair_counts_size * 3 + 2] = 1; // Initialize count to 1
            (*pair_counts_size)++;
        } else {
            // Increment count
            pair_counts[index * 3 + 2]++;
        }
    }
}


/*
* @brief Applies a merge operation to the given sequence of token IDs.
*
* @param ids Array of token IDs to be merged.
* @param ids_size Pointer to the size of the ids array (will be updated after merging).
* @param pair The pair of tokens to be merged.
* @param idx The new token ID to replace the merged pair.
*/
void merge(int *ids, size_t *ids_size, IntPair pair, int idx) {
    int new_ids[MAX_TEXT_SIZE];
    size_t new_ids_size = 0;
    for (size_t i = 0; i < *ids_size; ++i) {
        if (ids[i] == pair.first && i < *ids_size - 1 && ids[i + 1] == pair.second) {
            new_ids[new_ids_size++] = idx;
            ++i;  // Skip the next element
        } else {
            new_ids[new_ids_size++] = ids[i];
        }
    }
    memcpy(ids, new_ids, new_ids_size * sizeof(int));
    *ids_size = new_ids_size;
}


int main() {
    BasicTokenizer *tokenizer = create_tokenizer();
    
    const char *text = "hello world the sky is blue";
    size_t vocab_size = 300;

    printf("Input Text:%s\n",text);
    train(tokenizer, text, vocab_size, 1);

    // Encode the text
    int ids[MAX_TEXT_SIZE];
    size_t ids_size = 0;
    encode(tokenizer, text, ids, &ids_size);
    
    // Decode the ids
    char decoded_text[MAX_TEXT_SIZE];
    decode(tokenizer, ids, ids_size, decoded_text);
    
    printf("Encoded IDs:\n");
    for (size_t i = 0; i < ids_size; ++i) {
        printf("%d ", ids[i]);
    }
    printf("\nDecoded text: %s\n", decoded_text);
    
    clean_tokenizer(tokenizer);

    return 0;
}