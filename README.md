# Interpreting Coding Agents Through Their Behavioral Trajectories

Code for the paper by Ibragim Badertdinov and Alexander Golubev, presented at the [1st Workshop on Interpreting Agent Behavior (IAB) at NeurIPS 2026](https://iab-agents.github.io/).

## Contents

- `parser.py`, `extract_metrics.py` — deterministic parser and per-trajectory metrics.
- `python_pipeline.py`, `python_command_methods.py`, `config/python_analysis.json` — Python-panel metrics, fixed table selections and support counts.
- `multilingual_pipeline.py` — multilingual profile and retrieval analysis.
- `audit_pipeline.py` — model-consensus agreement and parser-score summaries.
- `python_historical/` — original matched-pair, split-half and cross-repository research code; see its README for required inputs and limitations.
- `codebook.md`, `config/`, `requirements.txt`, `tests/` — metric definitions, reproducibility configuration, direct dependency versions and synthetic tests.

## Data and artifact availability

No trajectories, Parquet files, consensus labels or generated analysis outputs are included in the current checkout. The Python pipeline downloads the compressed Parquet files for `python_spring2026` from [`nebius/SWE-rebench-leaderboard-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard-trajectories). The dataset was marked **private** on 2026-11-09 and requires authorized access. It contains 18,468 Python trajectories on 110 tasks; its dataset card specifies CC BY 4.0.

The multilingual pipeline downloads the existing compressed JSONL shards from the public [`ibragim-bad/swe_rebench_07_2026_trajectories`](https://huggingface.co/datasets/ibragim-bad/swe_rebench_07_2026_trajectories) dataset. Both dataset revisions are pinned in `config/`; cached or generated files are not committed.

The 134,329-trajectory corpus and frozen registries used for the original matched/null and cross-repository analyses are not released. Historical code requires that original workspace and is not runnable from this checkout alone. The audit command also requires external consensus and prediction files, which are not included.

## Usage

The parser and audit code use the Python standard library (Python 3.10+):

```bash
python3 parser.py /path/to/trajectory.json --schema auto --include-metrics
python3 extract_metrics.py /path/to/trajectory.json --output metrics.json
python3 -m unittest discover -s tests -v
```

Supply a trusted repository root in the trajectory or with `--repository-root` for file-role-dependent metrics.

For the Python panel, install optional dependencies in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 python_pipeline.py --require-complete-panel --output outputs/python
```

Authenticate with an account authorized to access the dataset, or pass `--input /path/to/parquet-directory` to use local files. The pinned revision, expected population and table selections are in `config/python_analysis.json`; `--revision` can override the revision. The pipeline records input hashes and writes redacted rollout metrics, task-equal profiles, `generation_effort.csv`, `scaffold.csv` and support counts. `--require-complete-panel` checks the configured trajectory, task and configuration counts. Generated outputs, cached data and Parquet files are ignored by Git.

The multilingual analysis can be run separately:

```bash
python3 multilingual_pipeline.py all --output outputs/multilingual --cache .cache/multilingual
```

Given external audit consensus and parser-score CSV files, run:

```bash
python3 audit_pipeline.py --consensus /path/to/consensus.csv --scores /path/to/parser_scores.csv --output outputs/audit/summary.json
```

## Scope and limitations

The Python-panel program is not an exact reproduction of the paper tables. The checked-in parser has an observation-pairing correction relative to the research-time `behavioral-parser-4.0.0+git.4ec73bdbd306`, and historical command-method labels used separate frozen artifacts. The parser does not process normalized `file_change` events: affected rollouts are counted and excluded from profile aggregation, and their per-rollout metrics are partial. Undefined values remain missing.

In a prior comparison against parser-4 replacement tables, 31 of 100 display-rounded cells differed, including Codex comparisons with partial support. The original matched/null and cross-repository numbers were not rerun. Model-consensus labels are surrogate annotations, not human ground truth. A parsed mutation or lack of an observed failure does not establish that the filesystem changed.

Code is released under the MIT license; external data remain subject to their own terms.

## Citation

```bibtex
@inproceedings{badertdinov2026interpreting,
  title     = {Interpreting Coding Agents Through Their Behavioral Trajectories},
  author    = {Badertdinov, Ibragim and Golubev, Alexander},
  booktitle = {1st Workshop on Interpreting Agent Behavior (IAB) at NeurIPS 2026},
  year      = {2026}
}
```
