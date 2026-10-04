# Source scope and execution boundaries

The installable `kx_simple_hmadrl` package contains the original V6 networks, observation definitions and causal feature builders, DDQN learner implementation, execution ledger and environment. `kx_base_gc.contract` supplies the original contract definitions used by the ledger. The two release package initializers are deliberately inert; all seven implementation modules retain their original bytes.

The historical Python snapshot retains the original relative source paths for the V6 acquisition/training/replay orchestration, structural reconstruction, component implementations and supporting baseline code. Original test sources are included so the selected synthetic checks can run against the installable core. Original `pyproject.toml` files document historical dependencies; they are not the release installation entry point. Referenced historical README files and runtime contracts have intentionally not been copied. The legacy baseline report generator and its dedicated test are also omitted; archived report commands that reference them are not supported.

Historical entry points locate contract files relative to their module paths and verify source, dataset and checkpoint hashes. They also depend on bound data manifests and research access receipts. Those external prerequisites are not in this code-only distribution. The original path and integrity checks remain in the archived Python; they have not been weakened for portability. Do not add the entire historical snapshot to `PYTHONPATH` and assume its workflows are ready to execute.

The root package supports model and learner imports, feature/environment interfaces, untrained synthetic inference and the selected unit checks. Importing learner code does not execute an optimizer. The default checks use no real market records and perform no fitting or financial replay. The full historical test suite has not been declared portable or run as part of this release.

## Check an installed wheel

For a wheel installed into a separate package directory, verify that imports come from that directory rather than the local `src` tree:

```sh
python scripts/check_core.py --expect-package-root /path/to/installed-packages
```

The test source is still read from this repository's snapshot. The flag switches and checks the imported implementation location. A hash check verifies file integrity, not scientific validity or performance.

## Inputs outside the release

Historical fitting and replay require separately obtained GC trade and book observations, pretraining inputs, data/normalization manifests, model checkpoints and the original runtime contracts/receipts. This distribution grants no rights to those external materials. No result exports, financial outcome arrays, generated figures, trained parameters or account configuration are included.
