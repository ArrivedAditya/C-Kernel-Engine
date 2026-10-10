import math

import torch as th
import torch.nn.functional as F

v0 = None


def timemix(params, x, vprime_0, shift_state, wkv_state, layer_id):
    B, T, C = x.shape()  # B: batch_size, T: sequence_length, C: d_model
    N = wkv_state.shape[-1]  # N: head_size
    H = C // N  # H: head_count

    # Weights prepartion logic
    x_shifted = th.cat([shift_state, x[:, -1, :]], dim=1)
    shift_state = x[:, -1, :]

    # BTC
    x_receptance = th.lerp(x, x_shifted, params.mu_r)
    x_decay = th.lerp(x, x_shifted, params.mu_d)
    x_key = th.lerp(x, x_shifted, params.mu_k)
    x_value = th.lerp(x, x_shifted, params.mu_v)
    x_iclr = th.lerp(x, x_shifted, params.mu_a)
    x_gate = th.lerp(x, x_shifted, params.mu_g)

    r = x_receptance @ params.W_receptance
    d = params.decay_lora(x_decay)
    k = x_key @ params.W_key
    vprime = x_value @ params.W_value
    gate = params.gate_lora(x_gate)
    iclr = params.iclr_lora(x_iclr).sigmoid()

    # Layer 0: no interpolation and return value to be use by after 0.
    if layer_id == 0:
        v = vprime_0 = vprime
    else:
        value_residual_gate = th.sigmoid(params.nu_lora(x_value))
        v = th.lerp(vprime, vprime_0, value_residual_gate)

    decay = th.exp(-math.exp(0.5) * d.to(th.float).sigmoid())
    removal_k = k + params.removal_key_multiplier
    removal_k = F.normalize(removal_k.view(B, T, H, -1), dim=1).view(B, T, C)
    replacement_k = th.lerp(k, k + iclr, params.iclr_mix_amt)

    # recurrence relation
    out = th.empty_like(x).view(B, T, C)
    for t in range(T):
        # One step for wkv_state translation
        decay_t = decay[:, t].view(B, H, N, 1)
        iclr_t = iclr[:, t].view(B, H, N, 1)
        removal_k_t = removal_k[:, t].view(B, H, N, 1)
        replacement_k_t = replacement_k[:, t].view(B, H, N, 1)
        v_t = v[:, t].view(B, H, N, 1)
        r_t = r[:, t].view(B, H, N, 1)

    wkv_state = (
        wkv_state + decay_t.mT - wkv_state @ removal_k_t @ (iclr_t + removal_k_t).mT
    )
    wkv_state = wkv_state + v_t @ replacement_k_t.mT
    y = wkv_state @ r_t  # BHVK mathmul BHKI = BHVI
    out[:, t] = y.view[B, T, C]

    # normalization
    y = F.group_norm(
        y.view(B * T, -1),
        num_groups=H,
        weight=params.ln_x.weight,
        bias=params.ln_x.bias,
        eps=H * 1e-5,
    ).view(B, T, -1)

    bonus = (r * k * params.bonus_multiplier).sum(dim=-1, keepdim=True) * v
    bonus = bonus.view(B, T, C)
    out = out + bonus
    out = (out * gate) @ params.W_output  # BTC heads

    return out, vprime, shift_state, wkv_state


def channelmix(params, x, shift_state):
    x_shifted = th.cat([shift_state, x[:, -1, :]], dim=1)
    shift_state = x[:, -1, :]
    xk = th.lerp(x, x_shifted, params.mu_x)
    k = params.W_k @ xk
    v = params.W_v @ th.relu(k).square()
    return v, shift_state


def rwkv_model(params, input_idx, state):
    x = params.embedding(input_idx)
    x = params.layer_norm_pre(x)

    v0 = None
    layer_id = -1

    for layer in params.layers:
        layer_id = layer_id + 1
        dx, v0, state.timemix_shiftstate, state.timemix_wkvstate = timemix(
            layer.time_mix,
            layer.time_norm_timemix(x),
            v0,
            state.timemix_shiftstate,
            state.timemix_wkvstate,
            layer_id,
        )
        x = x + dx
        dx, state.channelmix_shiftstate = channelmix(
            layer.channelmix, layer.layer_norm_channelmix(x), state.channelmix_shiftcase
        )
        x = x + dx

    x = params.layer_norm_out(x)
    logits = params.head(x)
    return logits, state
