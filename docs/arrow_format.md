# S2APLER Arrow Bundle Format

S2APLER supports an indexed Arrow IPC bundle as an alternative on-disk format for
`papers.json` and `clusters.json`. The built-in converter writes this format:

```bash
uv run python scripts/convert_to_arrow.py \
  --papers data/papers.json \
  --clusters data/clusters.json \
  --output-dir data/arrow \
  --overwrite
```

`PDData("data/arrow", ...)` loads the full paper dataset, just like
`PDData("data/papers.json", ...)`. Clusters are explicit, also like JSON:

```python
from s2apler.data import PDData

dataset = PDData(
    "data/arrow",
    clusters="data/arrow",
    name="paper_clustering_dataset",
    balanced_pair_sample=False,
)
```

## Bundle Layout

An Arrow bundle is a directory containing `manifest.json` plus Arrow IPC files
and JSON indexes. The loader also accepts a direct path to `manifest.json`.

Required files:

- `manifest.json`
- `papers.arrow`
- `paper_authors.arrow`
- `block_index.json`
- `block_counts.json`
- `papers_batch_index.json`
- `paper_authors_batch_index.json`

Optional file:

- `clusters.arrow`

The Arrow files must be Arrow IPC file-format files readable by
`pyarrow.ipc.open_file`. The current validator checks exact schema equality, so
field names, field order, and Arrow types must match the schemas below.

## `manifest.json`

`manifest.json` identifies the bundle schema, maps logical file names to bundle
files, and stores counts used by validation and profiling.

Required top-level fields:

```json
{
  "schema": "s2apler_arrow_bundle_v2",
  "files": {
    "papers": "papers.arrow",
    "paper_authors": "paper_authors.arrow",
    "block_index": "block_index.json",
    "block_counts": "block_counts.json",
    "papers_batch_index": "papers_batch_index.json",
    "paper_authors_batch_index": "paper_authors_batch_index.json",
    "clusters": "clusters.arrow"
  },
  "counts": {
    "paper_count": 0,
    "paper_author_count": 0,
    "block_count": 0,
    "cluster_count": 0,
    "cluster_membership_count": 0
  },
  "largest_block": {
    "block": "",
    "size": 0
  },
  "batch_sizes": {
    "papers": 16384,
    "paper_authors": 65536,
    "clusters": 65536
  },
  "batch_counts": {
    "papers": 0,
    "paper_authors": 0,
    "clusters": 0
  }
}
```

`files.clusters` is optional. The other `files` entries are required. Extra
metadata, such as `created_at_utc` or source paths, is allowed.

## `papers.arrow`

One row per paper. `paper_id` is the S2APLER sourced paper id as a string. The
bundle loader reconstructs this table into the same raw dictionary shape as
`papers.json`.

| Field | Arrow type |
| --- | --- |
| `paper_id` | `string` |
| `title` | `string` |
| `abstract` | `string` |
| `venue` | `string` |
| `journal_name` | `string` |
| `year` | `int64` |
| `corpus_paper_id` | `int64` |
| `doi` | `string` |
| `pmid` | `string` |
| `source_id` | `string` |
| `pdf_hash` | `string` |
| `source` | `string` |
| `block` | `string` |
| `source_uris` | `list<string>` |

Nulls are accepted for nullable paper attributes. `block` should be populated
when you want block-based prediction or profiling.

## `paper_authors.arrow`

One row per paper author. Authors are grouped by `paper_id` during load in the
same order they were written from `papers.json`; `position` is preserved as data
but is not used to reorder authors.

| Field | Arrow type |
| --- | --- |
| `paper_id` | `string` |
| `position` | `int64` |
| `first` | `string` |
| `middle` | `list<string>` |
| `last` | `string` |
| `suffix` | `string` |
| `author_info_first` | `string` |
| `author_info_middle` | `string` |
| `author_info_last` | `string` |
| `author_info_suffix` | `string` |

The `first`/`middle`/`last`/`suffix` fields match the JSON author shape used by
S2APLER. The `author_info_*` fields preserve alternate source name fields when
present.

## `clusters.arrow`

This file is optional. Include it when you want the bundle to replace
`clusters.json` for training or evaluation.

One row per cluster object. `cluster_key` preserves the top-level JSON object
key, and `cluster_json` stores the full cluster object as compact JSON. This
keeps load behavior identical to `clusters.json`, including empty clusters,
numeric membership ids, and cases where the top-level key differs from the
inner `cluster_id`.

| Field | Arrow type |
| --- | --- |
| `cluster_key` | `string` |
| `cluster_json` | `string` |

On load, rows are restored to `clusters.json` shape:

```json
{
  "outer_cluster_key": {
    "cluster_id": "inner_cluster_id",
    "sourced_paper_ids": ["paper_id_1", "paper_id_2"],
    "model_version": 1
  }
}
```

## JSON Indexes

The JSON indexes make block and paper subset loading possible without scanning
every Arrow record batch.

`block_index.json` maps each block key to the ordered paper ids in that block:

```json
{
  "smithj": ["100", "101", "102"]
}
```

`block_counts.json` maps each block key to its paper count:

```json
{
  "smithj": 3
}
```

`papers_batch_index.json` maps each paper id to the zero-based record batch in
`papers.arrow` that contains its paper row:

```json
{
  "100": 0,
  "101": 0,
  "500000": 31
}
```

`paper_authors_batch_index.json` maps each paper id to the zero-based record
batch or batches in `paper_authors.arrow` that contain its author rows:

```json
{
  "100": 0,
  "101": 0,
  "500000": 78
}
```

The built-in converter keeps all author rows for a paper in one batch. Compatible
writers may use a list of batch ids if a paper's author rows span batches.

## Manual Writers

If you generate a bundle outside `scripts/convert_to_arrow.py`:

1. Write Arrow IPC file-format files with the exact schemas above.
2. Store all ids used for Arrow joins and indexes as strings.
3. Keep `paper_id` values unique in `papers.arrow`.
4. Make `paper_authors.arrow.paper_id` refer to ids present in `papers.arrow`.
5. Make `block_index.json` and `block_counts.json` agree with `papers.arrow.block`.
6. Make batch index values point to valid zero-based record batches.
7. Set `manifest.schema` to `s2apler_arrow_bundle_v2`.
8. Run `validate_arrow_bundle(path, require_clusters=True)` when clusters are
   present, or `validate_arrow_bundle(path)` for inference-only bundles.

Example validation:

```python
from s2apler.arrow_io import validate_arrow_bundle

validate_arrow_bundle("data/arrow", require_clusters=True)
```
