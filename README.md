# match-recognize-datagen

Synthetic data generator for MATCH RECOGNIZE queries with pattern specifications and constraint definitions.

## Quick Start

```bash
pip install -e .
python scripts/main.py --output-dir ./output
```

## Usage

```python
from match_recognize_datagen import DataGenerator, GeneratorConfig

config = GeneratorConfig(
    initial_table_size=1000,
    total_rows=5000,
    num_columns=5,
    batch_sizes=[1000, 1000, 1000],
    rows_per_window=100,
)

generator = DataGenerator(config)
full_table, batches = generator.generate()
```

See `examples/examples.py` for more examples.

## Insertion-First API (pattern_insertions_approach)

This repository also includes an insertion-first generator API that models
per-variable/per-column distributions as weighted rules (`exact` and `range`).

Run the example:

```bash
python examples/pattern_insertions_example.py
```

The example demonstrates `config_a/config_b` style configs and dependent
distance constraints with retry-and-shortfall reporting.

## Features

- **PATTERN clause**: Variables, wildcards, Kleene+ operators
- **DEFINE clause**: Independent/pairwise/window conditions with selectivity
- **Flexible batching**: Configurable batch sizes and window constraints
- **Output**: Parquet files per batch
- **Distributions**: Uniform, Zipf, Normal for numerical attributes
