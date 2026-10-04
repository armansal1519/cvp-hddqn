# CVP-HDDQN: Research software for hierarchical reinforcement learning in gold futures

Version 1.0.0 · Arman Salehi · [Software DOI](https://doi.org/10.5281/zenodo.23142380)

This source-only distribution provides the CVP-HDDQN neural, feature, learner and execution-ledger implementations. A 15-minute information publisher supplies expiring advice to a five-minute trader that controls inventory. The installable core preserves seven original implementation modules byte-for-byte and uses lightweight package initializers.

No financial outcomes, result tables, generated figures, market-data records, trained weights or original repository history are distributed. Synthetic checks establish software-interface and accounting invariants; they do not establish trading performance.

## Install and inspect

Use Python 3.12 in a fresh environment from this repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[check]'
python scripts/verify_manifest.py
python scripts/check_core.py
python examples/inspect_model.py
```

On Windows, activate `.venv\Scripts\activate`. The check script calls exactly eight selected original unit tests using synthetic inputs, disables gradient recording and rejects optimizer updates. The example initializes an untrained model and prints output shapes, parameter counts and an artificial message-expiry example.

## Contents

- `src/`: installable neural, causal-feature, learning and execution-ledger API.
- `source_snapshot/`: selected original Python implementation and test sources, plus original package declarations. These historical sources are for inspection; their external research contracts are not supplied.
- `scripts/`: synthetic core checks and source-integrity verification.
- `examples/inspect_model.py`: deterministic synthetic forward-pass example.
- `docs/SCOPE.md`: component map and external prerequisites.
- `provenance/source_files.json`: hashes and packaging-only adaptations, with no financial outcomes.

The historical snapshot is not installed by the root package. It includes support code and earlier comparison utilities; inclusion does not designate any algorithm as a published benchmark. It cannot independently rerun the historical research workflow: licensed market data, weights, external configuration/contracts and original bound receipts are absent. This release does not bypass those guards or substitute artificial performance evidence.

## Citation

Salehi, A. (2026). *CVP-HDDQN: Research software for hierarchical reinforcement learning in gold futures* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23142380

Machine-readable metadata is provided in `CITATION.cff`.

## Reuse

Copyright 2026 Arman Salehi. Code is licensed under [PolyForm Noncommercial 1.0.0](LICENSE); project documentation is licensed under [CC BY-NC 4.0](DOCS_LICENSE). Commercial use is not licensed. Dependencies retain their own licenses. This is source-available software with noncommercial restrictions.

Report installation or interface issues at [GitHub Issues](https://github.com/armansal1519/cvp-hddqn/issues), without attaching licensed data, weights or credentials.
