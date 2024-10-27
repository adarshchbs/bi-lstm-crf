import torch
from typing import List, Tuple


def concat_with_padding(
    sequences: List[torch.Tensor], padding_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad and concatenate sequences of token embeddings.
    All sequences must have the same embedding dimension (last dimension).

    Args:
        sequences (List[torch.Tensor]): List of token embedding sequences
                                      Each tensor has shape [sequence_length, embedding_dim]
        padding_value (float): Value to use for padding (default: 0.0)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_sequences: Tensor of shape [batch_size, max_seq_length, embedding_dim]
            - mask: Boolean tensor of shape [batch_size, max_seq_length]
              where True indicates valid (non-padded) positions

    Raises:
        ValueError: If sequences have different embedding dimensions or list is empty
    """
    if not sequences:
        raise ValueError("Empty sequence list provided")

    # Get embedding dimension from first sequence
    embedding_dim = sequences[0].size(-1)

    # Validate all sequences have same embedding dimension
    for i, seq in enumerate(sequences):
        if seq.size(-1) != embedding_dim:
            raise ValueError(
                f"Sequence {i} has embedding dimension {seq.size(-1)}, "
                f"expected {embedding_dim}"
            )

    # Get max sequence length
    max_seq_length = max(seq.size(0) for seq in sequences)
    batch_size = len(sequences)

    # Initialize padded tensor and mask
    padded_sequences = torch.full(
        (batch_size, max_seq_length, embedding_dim),
        padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    mask = torch.zeros(
        (batch_size, max_seq_length), dtype=torch.bool, device=sequences[0].device
    )

    # Fill padded tensor and mask
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len] = seq
        mask[i, :seq_len] = True

    return padded_sequences, mask


# Example usage
if __name__ == "__main__":
    # Create example sequences with same embedding dimension
    embedding_dim = 4
    seq1 = torch.randn(3, embedding_dim)  # 3 tokens
    seq2 = torch.randn(5, embedding_dim)  # 5 tokens
    seq3 = torch.randn(2, embedding_dim)  # 2 tokens

    sequences = [seq1, seq2, seq3]
    padded_sequences, mask = concat_with_padding(sequences)

    print("Padded sequences shape:", padded_sequences.shape)
    print("Mask shape:", mask.shape)
    print("\nMask (True indicates valid tokens):")
    print(mask)

    # Example of invalid input
    try:
        invalid_seq = torch.randn(3, embedding_dim + 1)  # Different embedding dim
        concat_with_padding([seq1, invalid_seq])
    except ValueError as e:
        print("\nExpected error for invalid input:")
        print(e)
