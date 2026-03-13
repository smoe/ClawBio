---
name: methylation-clock
description: Compute epigenetic age from DNA methylation arrays using PyAging clocks from GEO accessions or local files.
version: 0.1.0
tags: [epigenetics, methylation, aging, clock, pyaging, GEO, illlumina-450k, EPIC]
trigger_keywords: [epigenetic age, methylation clock, pyaging, horvath, grimage, dunedinpace, GEO, GSE]
metadata:
  openclaw:
    requires:
      bins:
        - python3
      env: []
      config: []
    always: false
    emoji: "🧪"
    homepage: https://github.com/ClawBio/ClawBio
    os: [macos, linux]
    install:
      - kind: uv
        package: pandas
        bins: []
      - kind: uv
        package: numpy
        bins: []
      - kind: uv
        package: matplotlib
        bins: []
      - kind: uv
        package: pyaging
        bins: []
---

# Methylation Clock

## Why This Exists

Epigenetic age analysis is often blocked by difficult preprocessing and model-specific input requirements.
This skill standardizes a reliable PyAging workflow, from input loading through clock prediction, with reproducible outputs.

## Core Capabilities

1. Accepts GEO accession input (`--geo-id`) and local methylation data files (`--input`).
2. Applies notebook-aligned preprocessing (female derivation and EPICv2 aggregation).
3. Converts tabular data to AnnData and runs one or multiple PyAging clocks.
4. Exports clock predictions, missing-feature diagnostics, and metadata.
5. Produces figures, markdown report, and reproducibility bundle.

## Input Contract

- Exactly one input source:
  - GEO accession with `--geo-id` (example: `GSE139307`)
  - Local file with `--input` (`.pkl`, `.pickle`, `.csv`, `.tsv`)
- Output directory via `--output`
- Optional clock list via `--clocks`

## Output Structure

```
methylation_clock_report/
├── report.md
├── figures/
│   ├── clock_distributions.png
│   └── clock_correlation.png
├── tables/
│   ├── predictions.csv
│   ├── prediction_summary.csv
│   ├── missing_features.csv
│   └── clock_metadata.json
└── reproducibility/
    ├── commands.sh
    ├── environment.yml
    └── checksums.sha256
```

## Demo

```bash
python skills/methylation-clock/methylation_clock.py \
  --input pyaging_data/GSE139307_small.pkl \
  --output /tmp/methylation_clock_demo
```

## Usage

```bash
python skills/methylation-clock/methylation_clock.py \
  --geo-id GSE139307 \
  --output /tmp/methylation_clock_geo

python skills/methylation-clock/methylation_clock.py \
  --input my_methylation.pkl \
  --clocks Horvath2013,AltumAge,PCGrimAge,GrimAge2,DunedinPACE \
  --output /tmp/methylation_clock_local
```

## Safety

- Local-first processing.
- Warns before writing into non-empty output directories.
- Always includes ClawBio disclaimer in report.

## Integration with Bio Orchestrator

Triggered by terms such as:
- epigenetic age
- methylation clock
- Horvath
- GrimAge
- DunedinPACE
- GEO accession / GSE

Can be chained with:
- `rnaseq-de` for combined transcriptomic-aging analyses.
- `equity-scorer` for demographic context across cohorts.
