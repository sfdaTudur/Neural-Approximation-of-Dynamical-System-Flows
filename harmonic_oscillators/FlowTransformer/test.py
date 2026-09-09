# test.py

import math

import pytest
import torch

from data import (
    quantize,
    dequantize,
    flow,
    sample_ball,
    tokenize_inputs,
    sample_inputs,
    make_data,
    midpoint_flow,
)

from model import (
    ModelConfig,
    FlowTransformer,
    INPUT_LENGTH,
    OUTPUT_LENGTH,
)

from training import (
    autoregressive_loss,
    train_model,
)


# ============================================================
# quantize()
# ============================================================


def test_quantize_example():
    values = torch.tensor([0.30])

    result = quantize(
        values,
        low=-1.0,
        high=1.0,
        num_bins=8,
    )

    assert result.item() == 5
    assert result.dtype == torch.long


def test_quantize_shape_preserved():
    values = torch.tensor(
        [
            [-0.5, 0.0, 0.5],
            [0.1, 0.2, 0.3],
        ]
    )

    result = quantize(
        values,
        low=-1.0,
        high=1.0,
        num_bins=16,
    )

    assert result.shape == values.shape


def test_quantize_endpoints():
    values = torch.tensor(
        [-1.0, 1.0],
        dtype=torch.float64,
    )

    result = quantize(
        values,
        low=-1.0,
        high=1.0,
        num_bins=8,
    )

    assert result[0].item() == 0
    assert result[1].item() == 7


def test_quantize_clamps_outside_interval():
    values = torch.tensor(
        [-2.0, 2.0],
        dtype=torch.float64,
    )

    result = quantize(
        values,
        low=-1.0,
        high=1.0,
        num_bins=8,
    )

    assert result[0].item() == 0
    assert result[1].item() == 7


def test_quantize_rejects_invalid_num_bins():
    with pytest.raises(ValueError):
        quantize(
            torch.tensor([0.0]),
            -1.0,
            1.0,
            1,
        )


def test_quantize_rejects_invalid_interval():
    with pytest.raises(ValueError):
        quantize(
            torch.tensor([0.0]),
            1.0,
            -1.0,
            8,
        )


def test_quantize_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        quantize(
            torch.tensor([float("nan")]),
            -1.0,
            1.0,
            8,
        )


# ============================================================
# dequantize()
# ============================================================


def test_dequantize_midpoints():
    indices = torch.tensor([0, 1, 2, 3])

    result = dequantize(
        indices,
        low=0.0,
        high=4.0,
        num_bins=4,
    )

    expected = torch.tensor(
        [0.5, 1.5, 2.5, 3.5],
        dtype=torch.float64,
    )

    assert torch.allclose(result, expected)


def test_quantize_dequantize_error_bound():
    num_bins = 256

    values = torch.linspace(
        -0.999,
        0.999,
        1000,
        dtype=torch.float64,
    )

    indices = quantize(
        values,
        -1.0,
        1.0,
        num_bins,
    )

    reconstructed = dequantize(
        indices,
        -1.0,
        1.0,
        num_bins,
    )

    bin_width = 2.0 / num_bins

    error = torch.abs(
        values - reconstructed
    )

    assert torch.all(
        error <= bin_width / 2 + 1e-12
    )


def test_dequantize_rejects_out_of_range_indices():
    with pytest.raises(ValueError):
        dequantize(
            torch.tensor([8]),
            -1.0,
            1.0,
            8,
        )


def test_dequantize_rejects_noninteger_indices():
    with pytest.raises(ValueError):
        dequantize(
            torch.tensor([1.5]),
            -1.0,
            1.0,
            8,
        )


# ============================================================
# flow()
# ============================================================


def test_flow_at_zero_is_identity():
    generator = torch.Generator().manual_seed(0)

    states = torch.randn(
        20,
        6,
        dtype=torch.float64,
        generator=generator,
    )

    times = torch.zeros(
        20,
        dtype=torch.float64,
    )

    result = flow(times, states)

    assert torch.allclose(
        result,
        states,
        atol=1e-12,
        rtol=1e-12,
    )


def test_flow_is_periodic():
    generator = torch.Generator().manual_seed(1)

    states = torch.randn(
        20,
        6,
        dtype=torch.float64,
        generator=generator,
    )

    times = torch.rand(
        20,
        dtype=torch.float64,
        generator=generator,
    )

    first = flow(
        times,
        states,
    )

    second = flow(
        times + 2 * math.pi,
        states,
    )

    assert torch.allclose(
        first,
        second,
        atol=1e-12,
        rtol=1e-12,
    )


def test_flow_preserves_norm():
    generator = torch.Generator().manual_seed(2)

    states = torch.randn(
        100,
        6,
        dtype=torch.float64,
        generator=generator,
    )

    times = (
        2
        * math.pi
        * torch.rand(
            100,
            dtype=torch.float64,
            generator=generator,
        )
    )

    future_states = flow(
        times,
        states,
    )

    initial_norms = states.norm(dim=1)
    future_norms = future_states.norm(dim=1)

    assert torch.allclose(
        initial_norms,
        future_norms,
        atol=1e-12,
        rtol=1e-12,
    )


def test_flow_half_period_changes_sign():
    generator = torch.Generator().manual_seed(3)

    states = torch.randn(
        10,
        6,
        dtype=torch.float64,
        generator=generator,
    )

    times = torch.full(
        (10,),
        math.pi,
        dtype=torch.float64,
    )

    result = flow(times, states)

    assert torch.allclose(
        result,
        -states,
        atol=1e-12,
        rtol=1e-12,
    )


def test_flow_rejects_wrong_state_dimension():
    states = torch.randn(
        10,
        5,
        dtype=torch.float64,
    )

    times = torch.zeros(
        10,
        dtype=torch.float64,
    )

    with pytest.raises(ValueError):
        flow(times, states)


# ============================================================
# sample_ball()
# ============================================================


def test_sample_ball_shape():
    generator = torch.Generator().manual_seed(0)

    samples = sample_ball(
        size=100,
        radius=1.0,
        generator=generator,
    )

    assert samples.shape == (100, 6)
    assert samples.dtype == torch.float64


def test_sample_ball_samples_inside_radius():
    generator = torch.Generator().manual_seed(1)

    radius = 0.5

    samples = sample_ball(
        size=1000,
        radius=radius,
        generator=generator,
    )

    norms = samples.norm(dim=1)

    assert torch.all(
        norms <= radius + 1e-12
    )


def test_sample_ball_reproducible():
    generator1 = torch.Generator().manual_seed(123)
    generator2 = torch.Generator().manual_seed(123)

    samples1 = sample_ball(
        100,
        1.0,
        generator1,
    )

    samples2 = sample_ball(
        100,
        1.0,
        generator2,
    )

    assert torch.equal(
        samples1,
        samples2,
    )


def test_sample_ball_rejects_bad_radius():
    generator = torch.Generator().manual_seed(0)

    with pytest.raises(ValueError):
        sample_ball(
            10,
            -1.0,
            generator,
        )


# ============================================================
# tokenize_inputs()
# ============================================================


def test_tokenize_inputs_shape_and_range():
    generator = torch.Generator().manual_seed(0)

    states = sample_ball(
        size=20,
        radius=0.5,
        generator=generator,
    )

    times = (
        2
        * math.pi
        * torch.rand(
            20,
            dtype=torch.float64,
            generator=generator,
        )
    )

    num_bins = 32

    tokens = tokenize_inputs(
        times,
        states,
        tokenization_radius=1.0,
        num_bins=num_bins,
    )

    assert tokens.shape == (20, 7)
    assert tokens.dtype == torch.long

    assert torch.all(tokens >= 0)
    assert torch.all(tokens < num_bins)


# ============================================================
# sample_inputs()
# ============================================================


def test_sample_inputs_shapes():
    times, states, tokens = sample_inputs(
        size=50,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=32,
        seed=42,
    )

    assert times.shape == (50,)
    assert states.shape == (50, 6)
    assert tokens.shape == (50, 7)


def test_sample_inputs_respects_sampling_radius():
    _, states, _ = sample_inputs(
        size=1000,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=32,
        seed=42,
    )

    assert torch.all(
        states.norm(dim=1) <= 0.5 + 1e-12
    )


def test_sample_inputs_rejects_sampling_radius_too_large():
    with pytest.raises(ValueError):
        sample_inputs(
            size=10,
            sampling_radius=2.0,
            tokenization_radius=1.0,
            num_bins=32,
            seed=0,
        )


# ============================================================
# make_data()
# ============================================================


def test_make_data_shapes():
    data = make_data(
        size=100,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=32,
        seed=0,
    )

    assert len(data) == 100
    assert data.times.shape == (100,)
    assert data.initial_states.shape == (100, 6)
    assert data.future_states.shape == (100, 6)
    assert data.input_tokens.shape == (100, 7)
    assert data.output_tokens.shape == (100, 6)


def test_make_data_future_states_are_exact_flow():
    data = make_data(
        size=100,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=32,
        seed=0,
    )

    expected = flow(
        data.times,
        data.initial_states,
    )

    assert torch.allclose(
        data.future_states,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_make_data_output_tokens_match_quantization():
    num_bins = 32

    data = make_data(
        size=100,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=num_bins,
        seed=0,
    )

    expected = quantize(
        data.future_states,
        -1.0,
        1.0,
        num_bins,
    )

    assert torch.equal(
        data.output_tokens,
        expected,
    )


# ============================================================
# midpoint_flow()
# ============================================================


def test_midpoint_flow_shape():
    data = make_data(
        size=100,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=32,
        seed=0,
    )

    result = midpoint_flow(
        data.input_tokens,
        tokenization_radius=1.0,
        num_bins=32,
    )

    assert result.shape == (100, 6)


def test_midpoint_flow_matches_manual_computation():
    num_bins = 32

    data = make_data(
        size=50,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=num_bins,
        seed=0,
    )

    result = midpoint_flow(
        data.input_tokens,
        tokenization_radius=1.0,
        num_bins=num_bins,
    )

    midpoint_times = dequantize(
        data.input_tokens[:, 0],
        0,
        2 * math.pi,
        num_bins,
    )

    midpoint_states = dequantize(
        data.input_tokens[:, 1:],
        -1.0,
        1.0,
        num_bins,
    )

    expected = flow(
        midpoint_times,
        midpoint_states,
    )

    assert torch.allclose(
        result,
        expected,
    )


# ============================================================
# ModelConfig
# ============================================================


def test_model_config_valid():
    config = ModelConfig(
        num_bins=32,
        width=16,
        layers=2,
        heads=4,
    )

    config.validate()


def test_model_config_rejects_width_not_divisible_by_heads():
    config = ModelConfig(
        num_bins=32,
        width=15,
        layers=2,
        heads=4,
    )

    with pytest.raises(ValueError):
        config.validate()


def test_model_config_rejects_invalid_num_bins():
    config = ModelConfig(
        num_bins=1,
        width=16,
        layers=2,
        heads=4,
    )

    with pytest.raises(ValueError):
        config.validate()


# ============================================================
# FlowTransformer.forward()
# ============================================================


def make_small_model():
    config = ModelConfig(
        num_bins=16,
        width=16,
        layers=1,
        heads=4,
    )

    return FlowTransformer(config)


def test_model_parameter_count_positive():
    model = make_small_model()

    assert model.parameter_count > 0


def test_model_forward_shape():
    model = make_small_model()

    batch_size = 8

    input_tokens = torch.randint(
        0,
        16,
        (batch_size, INPUT_LENGTH),
    )

    output_tokens = torch.randint(
        0,
        16,
        (batch_size, OUTPUT_LENGTH),
    )

    logits = model(
        input_tokens,
        output_tokens[:, :-1],
    )

    assert logits.shape == (
        batch_size,
        OUTPUT_LENGTH,
        16,
    )


def test_model_forward_with_no_previous_outputs():
    model = make_small_model()

    batch_size = 8

    input_tokens = torch.randint(
        0,
        16,
        (batch_size, INPUT_LENGTH),
    )

    previous_outputs = torch.empty(
        batch_size,
        0,
        dtype=torch.long,
    )

    logits = model(
        input_tokens,
        previous_outputs,
    )

    assert logits.shape == (
        batch_size,
        1,
        16,
    )


def test_model_rejects_wrong_input_shape():
    model = make_small_model()

    bad_inputs = torch.randint(
        0,
        16,
        (8, 6),
    )

    previous_outputs = torch.empty(
        8,
        0,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(
            bad_inputs,
            previous_outputs,
        )


# ============================================================
# Causal autoregressive behavior
# ============================================================


def test_first_prediction_does_not_depend_on_future_outputs():
    """
    Changing Y2,...,Y5 should not change the logits used to
    predict Y1 because of the causal attention mask.
    """
    torch.manual_seed(0)

    model = make_small_model()
    model.eval()

    input_tokens = torch.randint(
        0,
        16,
        (1, INPUT_LENGTH),
    )

    previous_a = torch.tensor(
        [[1, 2, 3, 4, 5]],
        dtype=torch.long,
    )

    previous_b = torch.tensor(
        [[1, 9, 10, 11, 12]],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits_a = model(
            input_tokens,
            previous_a,
        )

        logits_b = model(
            input_tokens,
            previous_b,
        )

    # Prediction of Y1 is produced at the START position.
    assert torch.allclose(
        logits_a[:, 0, :],
        logits_b[:, 0, :],
        atol=1e-7,
        rtol=1e-7,
    )


# ============================================================
# generate()
# ============================================================


def test_greedy_generate_shape_and_range():
    torch.manual_seed(0)

    model = make_small_model()

    input_tokens = torch.randint(
        0,
        16,
        (10, INPUT_LENGTH),
    )

    generated_tokens, generated_scores = model.generate(
        input_tokens,
        greedy=True,
    )

    assert generated_tokens.shape == (
        10,
        OUTPUT_LENGTH,
    )

    assert generated_scores.shape == (
        10,
        OUTPUT_LENGTH,
        16,
    )

    assert torch.all(
        generated_tokens >= 0
    )

    assert torch.all(
        generated_tokens < 16
    )


def test_greedy_generation_is_deterministic():
    torch.manual_seed(0)

    model = make_small_model()

    input_tokens = torch.randint(
        0,
        16,
        (10, INPUT_LENGTH),
    )

    first_tokens, _ = model.generate(
        input_tokens,
        greedy=True,
    )

    second_tokens, _ = model.generate(
        input_tokens,
        greedy=True,
    )

    assert torch.equal(
        first_tokens,
        second_tokens,
    )


def test_stochastic_generation_reproducible_with_seed():
    torch.manual_seed(0)

    model = make_small_model()

    input_tokens = torch.randint(
        0,
        16,
        (10, INPUT_LENGTH),
    )

    generator1 = torch.Generator().manual_seed(123)
    generator2 = torch.Generator().manual_seed(123)

    tokens1, _ = model.generate(
        input_tokens,
        generator=generator1,
        greedy=False,
    )

    tokens2, _ = model.generate(
        input_tokens,
        generator=generator2,
        greedy=False,
    )

    assert torch.equal(
        tokens1,
        tokens2,
    )


def test_generate_restores_training_mode():
    model = make_small_model()
    model.train()

    input_tokens = torch.randint(
        0,
        16,
        (4, INPUT_LENGTH),
    )

    model.generate(
        input_tokens,
        greedy=True,
    )

    assert model.training


# ============================================================
# autoregressive_loss()
# ============================================================


def test_autoregressive_loss_is_scalar_and_finite():
    torch.manual_seed(0)

    model = make_small_model()

    input_tokens = torch.randint(
        0,
        16,
        (8, INPUT_LENGTH),
    )

    output_tokens = torch.randint(
        0,
        16,
        (8, OUTPUT_LENGTH),
    )

    loss = autoregressive_loss(
        model,
        input_tokens,
        output_tokens,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_autoregressive_loss_backward():
    torch.manual_seed(0)

    model = make_small_model()

    input_tokens = torch.randint(
        0,
        16,
        (8, INPUT_LENGTH),
    )

    output_tokens = torch.randint(
        0,
        16,
        (8, OUTPUT_LENGTH),
    )

    loss = autoregressive_loss(
        model,
        input_tokens,
        output_tokens,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        for gradient in gradients
    )


# ============================================================
# train_model()
# ============================================================


def test_train_model_runs_one_epoch():
    """
    Integration test: generate a very small dataset and make sure
    one full training epoch completes successfully.
    """
    torch.manual_seed(0)

    data = make_data(
        size=64,
        sampling_radius=0.5,
        tokenization_radius=1.0,
        num_bins=16,
        seed=0,
    )

    model = make_small_model()

    loss_history = train_model(
        model,
        data,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        device="cpu",
    )

    assert len(loss_history) == 1
    assert math.isfinite(
        loss_history[0]
    )