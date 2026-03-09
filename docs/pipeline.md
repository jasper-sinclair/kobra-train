# NNUE Pipeline

check_selfplay_perspective_features.py
     ↓
training.txt
     ↓
normalize_dataset.py
     ↓
verify_training_txt.py
     ↓
shuffle_training_txt.py
     ↓
convert_to_sparse.py
     ↓
verify_sparse_features.py
     ↓
verify_sparse_structure.py
     ↓
train.py (training_sparse.bin)



## Step-by-step:

1️⃣ check_selfplay_perspective_features.py

Purpose:

validate feature extraction

confirm white/black perspective logic

catch feature indexing bugs early

Output:

training.txt

✔



2️⃣ normalize_dataset.py

Purpose:

unify formats

convert evals → probability

truncate FEN to 4 fields

deduplicate positions

Output example:

FEN | result

Example:

rnbqkbnr/... w KQkq - | 0.53

This ensures the verify script sees only one format.

✔



3️⃣ verify_training_txt.py

Now the dataset is standardized:

FEN | result

Verification becomes simple:

Checks typically include:

valid FEN

side to move valid

result ∈ [0,1]

both kings exist

no corrupted lines

✔



4️⃣ shuffle_training_txt.py

Shuffling after normalization and verification is ideal.

You shuffle clean, deduplicated data.

Output:

training_shuffled.txt

✔



5️⃣ convert_to_sparse.py

Converts:

FEN | result

into:

training_sparse.bin

Binary format:

uint8  n_white
uint8  n_black
uint16 white_indices[]
uint16 black_indices[]
float32 result

✔


6️⃣ verify_sparse_features.py

Checks:

feature index < 768

counts valid entries

perspective symmetry

✔


Many pipelines skip this.

7️⃣ verify_sparse_structure.py

Checks binary integrity:

record alignment

feature counts

float parsing

EOF correctness

✔


8️⃣ train.py

Uses:

training_sparse.bin

✔


The production pipeline has three validation layers:

Stage	Purpose
verify_training_txt.py	text dataset validation
verify_sparse_features.py	feature correctness
verify_sparse_structure.py	binary file integrity
