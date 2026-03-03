The scripts in this folder are for building a minimal 768-input piece-square NNUE.

Feature Encoding:
6 piece types × 64 squares × 2 perspectives
Total input size: 768
Perspective-relative encoding (board flipped for black)

Features:
“Piece type on square from side-to-move perspective”
No king-relative indexing is used.

Architecture:
Input:   768
Hidden:  128 (shared first layer)
Output:  1 scalar evaluation

Forward pass:
Linear layer (768 → 128)
Clipped ReLU (0–1)
Squared activation (x²)
Concatenate white & black accumulators
Final linear layer → evaluation
Parameter Count
~100,000 parameters



