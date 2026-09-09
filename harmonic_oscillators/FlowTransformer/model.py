from dataclasses import dataclass

import torch
from torch import nn

INPUT_LENGTH = 7       # time and six initial-state coordinates
OUTPUT_LENGTH = 6     # six future-state coordinates
SEQUENCE_LENGTH = 13   # seven inputs, START, and five preceding output tokens


@dataclass(frozen=True)
class ModelConfig:
    """Network settings. A dataclass creates the initializer from these fields."""

    num_bins: int = 256
    width: int = 64  #token embedding dimension
    layers: int = 2
    heads: int = 4

    def validate(self):
        if not 2 <= self.num_bins <= 4096:
            raise ValueError("num_bins must lie between 2 and 4096")
        if self.width < 1 or self.heads < 1 or self.width % self.heads != 0:
            raise ValueError("width must be positive and divisible by positive heads")
        if self.layers < 1:
            raise ValueError("layers must be positive")


class TransformerBlock(nn.Module):
    """Attention and a small feed-forward network, each with a residual sum."""

    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.width,
            num_heads=config.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(config.width)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.width, 2 * config.width),
            nn.GELU(),
            nn.Linear(2 * config.width, config.width),
        )

    def forward(self, token_vectors, causal_mask):
        # PyTorch creates queries, keys, and values and combines the heads.
        normalized = self.attention_norm(token_vectors)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        token_vectors = token_vectors + attended

        normalized = self.feed_forward_norm(token_vectors)
        transformed = self.feed_forward(normalized)
        return token_vectors + transformed


class FlowTransformer(nn.Module):
    """P(output bins | input bins), predicting one coordinate at a time.

    Each bin ID becomes a learned lookup vector. Position vectors identify the
    time/coordinate slot.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.start_token_id = config.num_bins
        self.token_embedding = nn.Embedding(config.num_bins + 1, config.width)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, config.width)
        self.blocks = nn.ModuleList()
        for _ in range(config.layers):
            self.blocks.append(TransformerBlock(config))
        self.final_norm = nn.LayerNorm(config.width)
        self.output_layer = nn.Linear(config.width, config.num_bins)

        # Small initial output scores give a distribution close to uniform.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    @property
    def parameter_count(self):
        """Computed count of the numbers learned during training."""
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_tokens, previous_output_tokens):
        """Return scores of shape (batch, predicted coordinates, num_bins).

        Training passes output_tokens[:, :-1] as previous_output_tokens:
        sequence = [seven inputs, START, output_1, ..., output_5].
        START predicts output_1; output_1 predicts output_2, and so on.
        The target at a position cannot see its own answer or later answers.
        """
        if input_tokens.ndim != 2 or input_tokens.shape[1] != INPUT_LENGTH:
            raise ValueError("input_tokens must have shape (batch, 7)")
        if previous_output_tokens.ndim != 2:
            raise ValueError("previous_output_tokens must have shape (batch, length)")
        if len(input_tokens) != len(previous_output_tokens):
            raise ValueError("input and output batches must have the same size")
        if previous_output_tokens.shape[1] >= OUTPUT_LENGTH:
            raise ValueError("supply zero to five preceding output tokens")

        batch_size = len(input_tokens)
        start_tokens = torch.full(
            (batch_size, 1), self.start_token_id,
            dtype=torch.long, device=input_tokens.device,
        )
        sequence = torch.cat((input_tokens, start_tokens, previous_output_tokens), dim=1)
        sequence_length = sequence.shape[1]

        positions = torch.arange(sequence_length, device=sequence.device)
        token_vectors = self.token_embedding(sequence)

        position_vectors = self.position_embedding(positions).unsqueeze(0)
        token_vectors = token_vectors + position_vectors

        # True blocks attention in nn.MultiheadAttention. Only later positions
        # are blocked: each position can read itself and preceding positions.
        causal_mask = torch.ones(
            sequence_length, sequence_length,
            dtype=torch.bool, device=sequence.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        for block in self.blocks:
            token_vectors = block(token_vectors, causal_mask)

        # Discard input positions: we charge loss only for the six outputs.
        output_vectors = token_vectors[:, INPUT_LENGTH:]
        output_vectors = self.final_norm(output_vectors)
        transformer_scores = self.output_layer(output_vectors)
        #output has shape (batch_size, number_of_prediction_positions, num_bins)
        return transformer_scores

    @torch.no_grad()
    def generate(self, input_tokens, generator=None, greedy=False):
        """
        Return six generated bin IDs and the scores used to choose them.
        With greedy=True, choose the most probable bin at each step.
        With greedy=False, sample from the full categorical distribution.
        """
        was_training = self.training
        self.eval()
        generated_tokens = torch.empty(
            len(input_tokens), 0, dtype=torch.long, device=input_tokens.device,
        )  #holds generated output bin IDs
        scores_by_coordinate = []   #store logits for generated bin ID

        for _ in range(OUTPUT_LENGTH):
            scores = self(input_tokens, generated_tokens) #self.forward(input_tokens, generated_tokens)
            next_scores = scores[:, -1, :]   #take logits from final sequence position
            scores_by_coordinate.append(next_scores)
            if greedy == True:
                next_token = torch.argmax(next_scores, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_scores, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)

            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

        self.train(was_training)
        generated_scores = torch.stack(scores_by_coordinate, dim=1)
        return generated_tokens, generated_scores
