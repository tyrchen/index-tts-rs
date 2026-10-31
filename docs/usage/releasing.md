# Releasing index-tts-rs

Release validation runs on Apple Silicon macOS with full Xcode / Metal tooling,
CMake, the pinned stable Rust toolchain, nightly rustfmt, and cached model weights
(`itts models pull`). It never uploads
packages or pushes git refs. The workspace has five packages at one shared version.

## Prepare and verify

1. Review the English [README](../../README.md), [Chinese README](../../README.zh-CN.md),
   and [changelog](../../CHANGELOG.md). Keep the language versions aligned. Benchmark
   numbers are historical unless a new report records the environment and results.
2. Change `workspace.package.version` and the three exact internal dependency versions
   in the root `Cargo.toml` together, then refresh `Cargo.lock` as needed. Internal
   dependencies specify both a development path and a registry version so packaged
   consumers can resolve them without the repository checkout.
3. Install validation tools if missing:

   ```bash
   rustup toolchain install nightly --profile minimal --component rustfmt
   cargo install cargo-audit cargo-deny --locked
   ```

4. Run the full local preflight:

   ```bash
   make release-check
   ```

   During an uncommitted review, `make release-check PACKAGE_ARGS=--allow-dirty`
   allows packaging local changes. This does not bypass compilation or tests.
   Final publication must use a clean, committed working tree.

`release-check` runs build, unit / integration / documentation tests, strict linting,
model-dependent verification, dependency and license audits, instruction / skill
synchronization, API docs, CLI
smoke checks, and Cargo's
workspace package verification. `make package` also resolves the packaged workspace
against a local staging registry and rebuilds each source archive. It does not use
`--no-verify`. Archives are placed under `target/package-check/package` by default.

The core crate links `config/default.yaml` to the canonical root configuration.
Cargo packages the file contents, so its configuration example and tests remain
self-contained after extraction, without maintaining two copies of the YAML.
Keep the symlink when checking out the repository.

The audit policy contains one explicit maintenance exception:
[`RUSTSEC-2024-0436`](https://rustsec.org/advisories/RUSTSEC-2024-0436) for
`paste 1.0.15`, the compile-time macro dependency of `mlx-rs 0.33.0`. There is no
patched upstream version for this maintenance notice. The exception does not
suppress vulnerability advisories; review it when updating MLX. `cargo audit`
continues to print the maintenance warning.

## Verify model-dependent behavior

Public CI does not download the approximately 5 GB model repository. Several golden
tests return early when cached weights are absent, so a green CI run alone does not
prove current model parity. On a prepared Mac:

```bash
cargo build --release --workspace --locked
# Use the built executable, adjusting for your configured Cargo target directory.
# These commands assume that this executable is installed as itts.
itts models pull
itts doctor --output json
make verify-models
```

`verify-models` checks that both fixtures and a resolved local model exist before
running the golden checks and explicitly ignored tests. Confirm the expected snapshot
is cached; individual tests can otherwise return early without reporting a skip.
Run a CLI synthesis using a permitted reference recording and verify the resulting
WAV. For performance changes, collect repeated release-mode measurements, explicit
hardware / OS / model / commit information, and separate daemon RSS from client RSS.
Use `make bench` for CPU microbenchmarks. The existing benchmark report does not
contain enough environment metadata to certify performance on other machines.

Complete native builds before starting model checks. MLX can regenerate its shared
Metal library cache under `~/.mlx/lib`; overlapping a rebuild with inference can
interrupt a test while that resource is being replaced. `release-check` runs its
steps sequentially.

## Collect source artifacts

```bash
make release-artifacts
(cd dist && shasum -a 256 -c SHA256SUMS)
```

This produces five verified `.crate` source archives and `SHA256SUMS`. The `.crate`
files are gzip-compressed source tar archives, not executable downloads. Use a fresh
`DIST_DIR` when collecting a different version, to avoid retaining older archives:

```bash
make release-artifacts DIST_DIR=/tmp/index-tts-rs-0.2.0
```

The embedded emotion prototypes and model-derived fixtures have separate upstream
terms. The core archive must retain `assets/README.md`, `assets/LICENSE-INDEXTTS`,
and the project's MIT license. Downloaded full model weights are not bundled.
Do not describe the entire distribution as exclusively MIT licensed.

A bare `itts` executable is insufficient as a portable binary release: MLX uses
`mlx.metallib` resources and native platform libraries. This release workflow
intentionally distributes verified source archives. A future binary workflow needs
to collect the matching Metal library, retain dependency licenses, and test on a
clean machine without the builder's MLX cache before advertising standalone binaries.

## Publish crates and create a GitHub release

These are explicit maintainer actions; preflight and CI never publish to crates.io.
Authenticate to crates.io using Cargo's credential provider, then, from a clean tree:

```bash
# One final packaging / upload simulation.
cargo publish --workspace --dry-run --locked

# Actual registry upload, following local release gates.
make publish
```

Cargo publishes the dependency graph in order: core → director → service → CLI and
server. See the [Cargo publishing reference](https://doc.rust-lang.org/cargo/commands/cargo-publish.html).
Check registry state after a partial failure before retrying; a published version
cannot be replaced. Once published, users can install the CLI with
`cargo install index-tts-rs-cli --version 0.2.0 --locked` and the HTTP server with
`cargo install index-tts-rs-server --version 0.2.0 --locked` on a supported Mac.
The README uses source installation so it does not imply registry availability
before publication.

After reviewing the committed release, create an annotated tag matching every
package version and push it explicitly:

```bash
git tag -a v0.2.0 -m 'Release 0.2.0'
git push origin v0.2.0
```

The tag workflow validates the version, runs all CI gates, collects source archives,
and creates a **draft** GitHub release with checksums and generated notes. Review
its notes, asset contents, and model-license notices before making it public. It
does not automatically publish crates. Pull requests receive read-only repository
permissions; only the tag release job has permission to write release assets.

## API documentation

Generate locally on a supported Mac:

```bash
cargo doc --workspace --no-deps --locked --open
```

The MLX backend depends on native Apple tooling. docs.rs uses Linux builders and
cross-compiles other targets, so it is not a supported API documentation host for
this configuration. Package metadata points to the repository documentation entry.
See [MLX Rust documentation](https://github.com/oxiglade/mlx-rs#documentation) and
[docs.rs build constraints](https://docs.rs/about/builds) for the upstream limitations.
