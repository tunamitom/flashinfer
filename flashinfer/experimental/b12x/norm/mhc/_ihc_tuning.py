"""iHC tuning surface: query/config payloads and the b12x TuningContract.

Mirrors how mHC's fused kernels register in ``_tuning.py``: every per-size
launch geometry row lives in ``IHC_TUNING`` (whose 6144 row serves Hy4), and
``TUNING`` exposes the legal launch choices to the b12x preparation/race
machinery. The iHC knob space is small (Triton chunk sizes and warp count);
the tile rules are the mHC fused-Gram rules the kernels inherit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation._efficiency import AutotuneEffort
from b12x.preparation.tuning import (
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

IHC_MULT = 4
IHC_MIXES = 2 * IHC_MULT  # 8 rows: 4 pre + 4 post; mHC's comb rows are absent
# The mHC fused-kernel whitelist is (4096, 5120, 7168); Hy4 needs 6144, which
# already satisfies the kernel tile divisibility below.
IHC_SUPPORTED_HIDDEN_SIZES = (4096, 5120, 6144, 7168)
IHC_SOURCE_TILE_H = 128  # same source tile as the mHC Gram kernel
IHC_GRAM_BLOCK_H = 1024  # same Gram block as the mHC Gram kernel
IHC_DEFAULT_MAGNITUDE = 2.0  # hc_magnitude in the Hy4 config
IHC_DEFAULT_HC_EPS = 1.0e-6  # hc_eps in the Hy4 config
IHC_DEFAULT_NORM_EPS = 1.0e-5  # rms_norm_eps in the Hy4 config
IHC_OPERATIONS = ("pre", "post", "post_pre", "head")
IHC_BACKENDS = ("triton_fused", "torch")


@dataclass(frozen=True, kw_only=True)
class IhcConfig:
    """Tuned launch geometry for one hidden size.

    ``block_k`` is the projection K-loop chunk (must divide ``4 * hidden``);
    ``block_d`` is the collapse/update D-loop chunk (must divide ``hidden``);
    ``num_warps`` is the Triton warp count. Tile divisibility mirrors the mHC
    fused Gram kernel (``IHC_SOURCE_TILE_H`` / ``IHC_GRAM_BLOCK_H``).
    """

    hidden_size: int
    block_k: int
    block_d: int
    num_warps: int
    backend: str = "triton_fused"

    @classmethod
    def from_config(cls, payload: Mapping) -> "IhcConfig":
        expected = frozenset(item.name for item in fields(cls))
        if frozenset(payload) != expected:
            raise ValueError(f"iHC config fields must be {sorted(expected)}")
        return cls(
            hidden_size=int(payload["hidden_size"]),
            block_k=int(payload["block_k"]),
            block_d=int(payload["block_d"]),
            num_warps=int(payload["num_warps"]),
            backend=str(payload["backend"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "block_k": self.block_k,
            "block_d": self.block_d,
            "num_warps": self.num_warps,
            "backend": self.backend,
        }


# Tuning table. One row per whitelisted hidden size; the 6144 row serves Hy4.
IHC_TUNING: Mapping[int, IhcConfig] = MappingProxyType(
    {
        4096: IhcConfig(hidden_size=4096, block_k=1024, block_d=256, num_warps=4),
        5120: IhcConfig(hidden_size=5120, block_k=1024, block_d=256, num_warps=4),
        6144: IhcConfig(hidden_size=6144, block_k=1024, block_d=512, num_warps=8),
        7168: IhcConfig(hidden_size=7168, block_k=1024, block_d=512, num_warps=8),
    }
)


def tuning_for(hidden_size: int) -> IhcConfig:
    """Return the tuned launch configuration for a whitelisted hidden size."""
    hidden_size = int(hidden_size)
    config = IHC_TUNING.get(hidden_size)
    if config is None:
        raise ValueError(
            f"iHC hidden size must be one of {IHC_SUPPORTED_HIDDEN_SIZES} "
            f"(mHC's 4096/5120/7168 plus Hy4's 6144); got {hidden_size}"
        )
    _validate_tile_divisibility(config)
    return config


def _validate_tile_divisibility(config: IhcConfig) -> None:
    hidden_size = int(config.hidden_size)
    if (
        hidden_size % IHC_SOURCE_TILE_H
        or hidden_size % IHC_GRAM_BLOCK_H
        or (IHC_MULT * hidden_size) % config.block_k
        or hidden_size % config.block_d
    ):
        raise ValueError(
            f"hidden size {hidden_size} violates the iHC tile divisibility "
            f"({IHC_SOURCE_TILE_H} / {IHC_GRAM_BLOCK_H} / block_k / block_d)"
        )


@dataclass(frozen=True, kw_only=True)
class IhcQuery:
    """One prepared iHC invocation (the query half of the tuning contract)."""

    dtype: str
    max_tokens: int
    hidden_size: int
    operation: str
    magnitude: float = IHC_DEFAULT_MAGNITUDE
    hc_eps: float = IHC_DEFAULT_HC_EPS
    norm_eps: float = IHC_DEFAULT_NORM_EPS
    layernorm_eps: float = IHC_DEFAULT_NORM_EPS
    has_norm_weight: bool = False
    norm_weight_dtype: str = "float32"
    output_mode: str = "provided"
    autotune_effort: str = "default"


def _encode(query: IhcQuery) -> dict[str, object]:
    return {name: getattr(query, name) for name in query.__dataclass_fields__}


def _validate_query(query: IhcQuery, _device: DeviceIdentity | None) -> None:
    if not isinstance(query, IhcQuery):
        raise TypeError("query must be IhcQuery")
    AutotuneEffort(query.autotune_effort)
    if query.operation not in IHC_OPERATIONS:
        raise ValueError(f"unknown iHC operation {query.operation!r}")
    if query.dtype != "bfloat16":
        raise ValueError("iHC requires bfloat16 state")
    if any(
        type(value) is not int or value <= 0
        for value in (query.max_tokens, query.hidden_size)
    ):
        raise ValueError("iHC dimensions must be positive integers")
    if query.hidden_size not in IHC_SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            f"iHC supports hidden sizes {IHC_SUPPORTED_HIDDEN_SIZES}; "
            f"got {query.hidden_size}"
        )
    if not math.isfinite(query.magnitude) or query.magnitude <= 0:
        raise ValueError("post-gate magnitude must be finite and positive")
    for name in ("hc_eps", "norm_eps", "layernorm_eps"):
        value = getattr(query, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if query.operation in ("pre", "post_pre"):
        if (query.hc_eps, query.norm_eps) != (IHC_DEFAULT_HC_EPS, IHC_DEFAULT_NORM_EPS):
            raise ValueError(
                "iHC epsilon values differ from the Hy4 numerical contract"
            )
    if type(query.has_norm_weight) is not bool:
        raise TypeError("has_norm_weight must be a boolean")
    if query.has_norm_weight and query.operation not in ("pre", "post_pre"):
        raise ValueError("RMSNorm folding applies only to pre and post_pre")
    if query.norm_weight_dtype not in ("bfloat16", "float32"):
        raise ValueError("norm weight dtype must be bfloat16 or float32")
    if query.output_mode not in ("functional", "provided"):
        raise ValueError("unknown iHC output form")


def _validate(
    query: IhcQuery,
    config: IhcConfig,
    _device: DeviceIdentity | None,
) -> None:
    if not isinstance(config, IhcConfig):
        raise TypeError("config must be IhcConfig")
    if any(
        type(getattr(config, name)) is not int
        for name in ("hidden_size", "block_k", "block_d", "num_warps")
    ):
        raise TypeError("iHC launch geometry fields must be integers")
    if config.backend not in IHC_BACKENDS:
        raise ValueError(f"unsupported iHC backend {config.backend!r}")
    if config.hidden_size != query.hidden_size:
        raise ValueError("iHC config hidden size differs from the query")
    if (
        config.block_k <= 0
        or (IHC_MULT * config.hidden_size) % config.block_k
        or config.block_d <= 0
        or config.hidden_size % config.block_d
    ):
        raise ValueError("iHC launch chunks must divide the flattened width")
    if config.num_warps not in (1, 2, 4, 8):
        raise ValueError("iHC num_warps must be one of (1, 2, 4, 8)")
    _validate_tile_divisibility(config)


def _tuning_parameters(query: IhcQuery, device: DeviceIdentity | None) -> ParameterSpace:
    hidden = query.hidden_size
    total_k = IHC_MULT * hidden
    values = {
        "backend": IHC_BACKENDS,
        "block_k": tuple(k for k in (256, 512, 1024) if total_k % k == 0),
        "block_d": tuple(d for d in (128, 256, 512) if hidden % d == 0),
        "num_warps": (1, 2, 4, 8),
    }
    return ParameterSpace.create(
        TUNING.knobs,
        values=values,
        autotune_effort=AutotuneEffort(query.autotune_effort),
    )


def _materialize_tuning(
    query: IhcQuery, _device: DeviceIdentity | None, choice: FrozenMapping
) -> IhcConfig:
    return IhcConfig.from_config(
        FrozenMapping({"hidden_size": query.hidden_size, **dict(choice)})
    )


TUNING = TuningContract(
    component_id="norm.mhc.ihc",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(IhcQuery.__dataclass_fields__),
    config_fields=frozenset(IhcConfig.__dataclass_fields__),
    encode_query=_encode,
    encode_config=IhcConfig.to_dict,
    decode_config=IhcConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate,
    default_config=lambda query, device: IHC_TUNING[query.hidden_size],
    candidate_contract_version=1,
    knobs=(
        Knob(
            name="backend",
            values=IHC_BACKENDS,
            binding=ParameterBinding.COMPILE,
        ),
        Knob(name="block_k", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="block_d", values=None, binding=ParameterBinding.COMPILE),
        Knob(name="num_warps", values=(1, 2, 4, 8), binding=ParameterBinding.COMPILE),
    ),
    parameters=_tuning_parameters,
    materialize=_materialize_tuning,
)


__all__ = [
    "IHC_BACKENDS",
    "IHC_DEFAULT_HC_EPS",
    "IHC_DEFAULT_MAGNITUDE",
    "IHC_DEFAULT_NORM_EPS",
    "IHC_GRAM_BLOCK_H",
    "IHC_MIXES",
    "IHC_MULT",
    "IHC_OPERATIONS",
    "IHC_SOURCE_TILE_H",
    "IHC_SUPPORTED_HIDDEN_SIZES",
    "IHC_TUNING",
    "IhcConfig",
    "IhcQuery",
    "TUNING",
    "tuning_for",
]
