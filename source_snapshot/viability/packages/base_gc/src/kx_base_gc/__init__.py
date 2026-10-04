"""Causal Base-GC structural reconstruction.

The package deliberately does not describe itself as an exact replication.  The
paper does not publish enough preprocessing and optimization detail for that
claim.  :func:`registered_contract` exposes every supplied and registered
default as a hash-bound contract.
"""

from .bars import BAR_FEATURES, build_causal_timeframes
from .contract import (
    ACTIONS,
    AgentSpec,
    BaselineContract,
    ExecutionSpec,
    OptimizationSpec,
    contract_sha256,
    load_contract_yaml,
    registered_contract,
)
from .ensemble import (
    EnsembleTraceProduct,
    LoadedEnsembleBundle,
    VerifiedSeedRun,
    aggregate_seed_traces,
    create_ensemble_trace_product,
    load_empirical_ensemble_bundle,
    load_verified_seed_run,
    majority_action,
    write_empirical_ensemble_bundle,
    validate_ensemble_trace_frame,
)
from .empirical import (
    EmpiricalPreparedError,
    PreparedBuildResult,
    build_empirical_timeframes,
    produce_empirical_prepared_product,
    replay_arrival_aligned_fills,
    scan_timeframe_support,
    scan_timeframe_support_cutoffs,
)
from .support_scan import SupportScanResult, run_development_support_scan
from .orchestration import (
    AgentSupport,
    SeedHierarchyResult,
    train_seed_hierarchy,
    write_seed_hierarchy_result,
)
from .prepared import (
    PreparedBarsEvidence,
    PreparedHierarchyProduct,
    load_certified_prepared_bars,
    load_certified_prepared_product,
    synthetic_prepared_product,
    verify_prepared_manifest,
)
from .state import (
    StateRecord,
    align_inherited_position,
    build_decision_calendar,
    build_state_records,
    encode_window,
    observation_dimension,
)
from .sessions import (
    GLOBEX_TIME_ZONE,
    globex_session_date,
    globex_session_dates,
    is_registered_globex_bar_step,
    require_exact_globex_session_dates,
)
from .trace import (
    BaseTraceProduct,
    ModelManifest,
    create_trace_product,
    trace_content_sha256,
    validate_trace_frame,
)
from .transitions import (
    ActionLock,
    DelayedTransitionBuilder,
    Transition,
    transition_builder_for,
)

__all__ = [
    "ACTIONS",
    "BAR_FEATURES",
    "ActionLock",
    "AgentSpec",
    "BaseTraceProduct",
    "BaselineContract",
    "DelayedTransitionBuilder",
    "ExecutionSpec",
    "GLOBEX_TIME_ZONE",
    "EnsembleTraceProduct",
    "EmpiricalPreparedError",
    "LoadedEnsembleBundle",
    "ModelManifest",
    "OptimizationSpec",
    "PreparedBarsEvidence",
    "PreparedBuildResult",
    "PreparedHierarchyProduct",
    "SeedHierarchyResult",
    "AgentSupport",
    "StateRecord",
    "SupportScanResult",
    "Transition",
    "VerifiedSeedRun",
    "aggregate_seed_traces",
    "align_inherited_position",
    "build_causal_timeframes",
    "build_decision_calendar",
    "build_empirical_timeframes",
    "build_state_records",
    "contract_sha256",
    "create_ensemble_trace_product",
    "create_trace_product",
    "encode_window",
    "globex_session_date",
    "globex_session_dates",
    "is_registered_globex_bar_step",
    "load_contract_yaml",
    "load_empirical_ensemble_bundle",
    "load_verified_seed_run",
    "load_certified_prepared_bars",
    "load_certified_prepared_product",
    "majority_action",
    "observation_dimension",
    "produce_empirical_prepared_product",
    "registered_contract",
    "replay_arrival_aligned_fills",
    "require_exact_globex_session_dates",
    "trace_content_sha256",
    "synthetic_prepared_product",
    "scan_timeframe_support",
    "scan_timeframe_support_cutoffs",
    "train_seed_hierarchy",
    "run_development_support_scan",
    "transition_builder_for",
    "validate_trace_frame",
    "validate_ensemble_trace_frame",
    "verify_prepared_manifest",
    "write_empirical_ensemble_bundle",
    "write_seed_hierarchy_result",
]
