from dataclasses import dataclass
import math

import torch


def quantize(values: torch.Tensor, low, high, num_bins: int) -> torch.Tensor:
    """Return the index of the equal-width bin containing each value.

    low and high are the interval endpoints a_j and c_j. Scalars apply to every
    coordinate; vectors give one endpoint per coordinate, across every sample.

        index_j = floor(num_bins * (value_j - a_j) / (c_j - a_j))

    Clamp to [0, num_bins - 1], including at the right endpoint. The result has
    the same shape as values and contains torch.long integers.
    Example: quantize(torch.tensor([0.30]), -1, 1, 8) returns tensor([5]).
    """
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer of at least 2")
    values = torch.as_tensor(values, dtype=torch.float64)
    low = torch.as_tensor(low, dtype=torch.float64, device=values.device)
    high = torch.as_tensor(high, dtype=torch.float64, device=values.device)
    for tensor in (values, low, high):
        if not torch.isfinite(tensor).all():
            raise ValueError("values and interval endpoints must be finite")
    if not (high > low).all():
        raise ValueError("each right endpoint must exceed its left endpoint")

    fraction_along_interval = (values - low) / (high - low)
    indices = torch.floor(num_bins * fraction_along_interval)
    indices = indices.clamp(min=0, max=num_bins - 1)
    return indices.long()


def dequantize(indices: torch.Tensor, low, high, num_bins: int) -> torch.Tensor:
    """Return each indexed bin's midpoint, preserving the input shape.

    The midpoint is low + (index + 0.5) * bin_width. For original values within
    the interval, quantize followed by dequantize has error <= bin_width / 2.
    That bound does not apply to an incorrectly predicted bin.
    """
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer of at least 2")
    indices = torch.as_tensor(indices)
    if not torch.isfinite(indices).all():
        raise ValueError("bin indices must be finite")
    if not (indices == indices.long()).all():
        raise ValueError("bin indices must be integers")
    if not ((indices >= 0) & (indices < num_bins)).all():
        raise ValueError("bin indices must lie between 0 and num_bins - 1")

    low = torch.as_tensor(low, dtype=torch.float64, device=indices.device)
    high = torch.as_tensor(high, dtype=torch.float64, device=indices.device)
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("interval endpoints must be finite")
    if not (high > low).all():
        raise ValueError("each right endpoint must exceed its left endpoint")

    bin_width = (high - low) / num_bins
    midpoints = low + (indices.double() + 0.5) * bin_width
    return midpoints


def flow(time: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
    """Evaluate Psi(t, q, v); state order is (q1, q2, q3, v1, v2, v3).

    time can be a scalar, a vector of sample times, or a column of sample times.
    Physical calculations use float64; the model uses float32.
    """
    initial_state = torch.as_tensor(initial_state, dtype=torch.float64)
    if initial_state.shape[-1] != 6:
        raise ValueError("the state must have six coordinates")
    time = torch.as_tensor(time, dtype=torch.float64, device=initial_state.device)
    if time.ndim == initial_state.ndim and time.shape[-1] == 1:
        time = time.squeeze(-1)

    cosine = time.cos().unsqueeze(-1)
    sine = time.sin().unsqueeze(-1)
    position = initial_state[..., :3]
    velocity = initial_state[..., 3:]

    future_position = position * cosine + velocity * sine
    future_velocity = -position * sine + velocity * cosine
    return torch.cat((future_position, future_velocity), dim=-1)


def sample_ball(size: int, radius: float, generator: torch.Generator) -> torch.Tensor:
    """Sample uniformly by volume inside the six-dimensional ball of radius R."""
    if size <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("size and radius must be positive; radius must be finite")
    directions = torch.randn(size, 6, dtype=torch.float64, generator=generator)
    direction_lengths = directions.norm(dim=1, keepdim=True)
    unit_directions = directions / direction_lengths.clamp_min(1e-300)

    # In dimension 6, the volume inside radius r is proportional to r**6.
    uniform_draws = torch.rand(size, 1, dtype=torch.float64, generator=generator)
    distances_from_origin = radius * uniform_draws.pow(1.0 / 6.0)
    return unit_directions * distances_from_origin


def tokenize_inputs(times, initial_states, tokenization_radius, num_bins):
    """
    Build seven integer tokens: time, q1, q2, q3, v1, v2, v3.
    tokenization_radius determines the fixed coordinate interval [-R,R] used
    for all state tokens.
    """
    time_indices = quantize(times.reshape(-1, 1), 0, 2 * math.pi, num_bins)
    state_indices = quantize(initial_states, -tokenization_radius, tokenization_radius, num_bins)
    return torch.cat((time_indices, state_indices), dim=1)


def sample_inputs(size: int,
                sampling_radius: float,
                tokenization_radius: float,
                num_bins: int,
                seed: int,
                ):
    """
    Sample states from a ball of radius sampling_radius, but quantize 
    them using the fixed interval [-tokenization_radius, tokenization_radius]
    """
    if sampling_radius > tokenization_radius:
        raise ValueError(
            "sampling_radius must not exceed tokenization_radius"
        )
    generator = torch.Generator().manual_seed(seed)
    initial_states = sample_ball(size, sampling_radius, generator)
    uniform_times = torch.rand(size, dtype=torch.float64, generator=generator)
    times = 2 * math.pi * uniform_times
    input_tokens = tokenize_inputs(times, initial_states, tokenization_radius, num_bins)

    return times, initial_states, input_tokens


@dataclass
class FlowData:
    """Named arrays for one dataset; dataclass supplies their initializer."""

    times: torch.Tensor
    initial_states: torch.Tensor
    future_states: torch.Tensor
    input_tokens: torch.Tensor
    output_tokens: torch.Tensor

    def __len__(self):
        return len(self.times)


def make_data(
                size: int, 
                sampling_radius: float,
                tokenization_radius: float,
                num_bins: int,
                seed: int,
            ) -> FlowData:
    """
    Generate a dataset.

    Continuous initial states are sampled from a ball of radius
    sampling_radius.

    Both input and output states are quantized using the fixed
    coordinate interval [-tokenization_radius, tokenization_radius].
    """
    times, initial_states, input_tokens = sample_inputs(size, sampling_radius, 
                                                        tokenization_radius, num_bins, seed,
                                                        )
    #compute exact continuous flow
    future_states = flow(times, initial_states)
    output_tokens = quantize(future_states, -tokenization_radius, tokenization_radius, num_bins)
    return FlowData(times, initial_states, future_states, input_tokens, output_tokens)


def midpoint_flow(input_tokens, tokenization_radius, num_bins):
    """Exact flow evaluated at input-bin midpoints."""
    times = dequantize(input_tokens[:, 0], 0, 2 * math.pi, num_bins)
    initial_states = dequantize(input_tokens[:, 1:], -tokenization_radius, tokenization_radius, num_bins)
    return flow(times, initial_states)

