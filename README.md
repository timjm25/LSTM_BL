# CDMO Program Margin Analyzer

A command-line tool for **Contract Development and Manufacturing Organizations (CDMOs)** to analyse, report, and model margins across a portfolio of customer programs — from preclinical through commercial manufacturing.

---

## What It Does

| Capability | Description |
|---|---|
| **Program margin reporting** | Revenue, COGS, gross profit, gross margin %, EBITDA, EBITDA margin per program |
| **Portfolio roll-up** | Aggregate metrics across all active programs |
| **Breakdowns** | By phase (Preclinical → Commercial), by customer, by cost category, by revenue type |
| **Batch economics** | Revenue, COGS, and gross profit per batch for manufacturing programs |
| **At-risk flagging** | Programs below a configurable gross margin threshold |
| **Scenario / what-if modelling** | Price sensitivity, volume sensitivity, cost reduction impact, portfolio scenarios |
| **CSV export** | Export full portfolio data for Excel or BI tools |

---

## Quick Start

```bash
# Full portfolio report (default FY2025 sample data)
python run_cdmo_margin.py

# Program detail
python run_cdmo_margin.py --program COM-001

# Scenario: price sensitivity for a specific program
python run_cdmo_margin.py --scenario price --program P3-001

# Scenario: volume sensitivity (batch count changes)
python run_cdmo_margin.py --scenario volume --program COM-002

# Scenario: cost reduction impact
python run_cdmo_margin.py --scenario cost --program P2-001

# Portfolio-level what-if (revenue %, cost %)
python run_cdmo_margin.py --scenario portfolio

# Export to CSV
python run_cdmo_margin.py --export results.csv

# Export report + CSV to a directory
python run_cdmo_margin.py --export-dir ./reports/

# Custom at-risk threshold (default is 20%)
python run_cdmo_margin.py --threshold 25

# Different fiscal year
python run_cdmo_margin.py --fiscal-year 2026
```

---

## Sample Output

```
========================================================================
  CDMO PROGRAM MARGIN REPORT  —  BIOMANUFACTURING PARTNERS  FY2025
========================================================================
  Active Programs  :  8 of 9
  Total Revenue    :  $16,290.0K
  Total COGS       :  $10,328.0K
  Gross Profit     :  $5,962.0K
  Gross Margin     :  36.6%
  EBITDA           :  $5,360.0K
  EBITDA Margin    :  32.9%
========================================================================

PROGRAM MARGIN SUMMARY
  Program               Customer          Phase         Revenue     COGS        Gross Profit    GM%     EBITDA      EBITDA%
  mAb Drug Substance    AntibodyCo Ltd.   Commercial    $5,850.0K   $3,036.0K   $2,814.0K       48.1%   $2,706.0K   46.3%
  Gene Therapy Vector   GeneTherapeutic   Phase III     $2,930.0K   $2,000.0K   $930.0K         31.7%   $790.0K     27.0%
  ...

BATCH ECONOMICS
  Program               Phase         Batches/yr    Rev/Batch   COGS/Batch  GP/Batch    GM%
  mAb Drug Substance    Commercial    18            $325.0K     $168.7K     $156.3K     48.1%
  ...
```

---

## Architecture

```
LSTM_BL/
├── cdmo_margin/
│   ├── __init__.py
│   ├── models.py        # Program, Portfolio, CostItem, RevenueItem dataclasses + enums
│   ├── calculator.py    # MarginCalculator — aggregations, breakdowns, rankings
│   ├── scenario.py      # ScenarioAnalyzer — price / volume / cost / portfolio what-if
│   ├── report.py        # ReportGenerator — formatted text output + CSV export
│   └── sample_data.py   # Realistic 9-program CDMO portfolio (8 active, 1 on-hold)
├── tests/
│   └── test_cdmo_margin.py    # 36 unit tests
└── run_cdmo_margin.py         # CLI entry point
```

### Data Model

```
Portfolio
  └── Program (many)
        ├── RevenueItem (many)   — Batch Fee, FTE Charge, Milestone, Tech Transfer, …
        └── CostItem (many)      — Direct Materials, Labor, Equipment, QC/QA, Overhead, …
```

**Margin hierarchy:**

| Metric | Formula |
|---|---|
| Gross Profit | Revenue − COGS |
| Gross Margin % | Gross Profit / Revenue |
| EBITDA | Revenue − COGS − SG&A − R&D |
| Contribution Margin | Revenue − Variable Costs |

**Cost classification:** each `CostItem` carries an `is_variable` flag (variable vs fixed), enabling contribution margin and breakeven analysis.

---

## Scenario Analysis

### Price Sensitivity (`--scenario price --program <ID>`)
Models margin impact of pricing changes (−20% to +20%) while holding costs flat.

### Volume Sensitivity (`--scenario volume --program <ID>`)
Models margin impact of batch count changes. Variable costs scale with volume; fixed costs are held constant (fixed cost absorption effect).

### Cost Reduction (`--scenario cost --program <ID>`)
Models margin improvement from 0–20% cost reduction initiatives applied to all cost items.

### Portfolio Scenario (`--scenario portfolio`)
Applies uniform revenue and cost deltas across all active programs. Includes standard scenarios:
- Revenue ±10%
- COGS +10% (inflation)
- COGS −10% (savings initiative)
- Combined revenue growth + cost reduction
- Bear case

---

## Sample Portfolio

The built-in sample portfolio (`cdmo_margin/sample_data.py`) contains 9 programs across the full development spectrum, modelled on typical CDMO program types:

| ID | Program | Phase | Customer | Molecule type |
|---|---|---|---|---|
| COM-001 | Recombinant Protein A | Commercial | BioPharm Co. | rHSA |
| COM-002 | mAb Drug Substance | Commercial | AntibodyCo Ltd. | Anti-IL6 mAb |
| P3-001 | Gene Therapy Vector | Phase III | GeneTherapeutics | AAV9 |
| P2-001 | mRNA Vaccine DS | Phase II | VaccineGen Inc. | mRNA-LNP |
| P2-002 | Cell Therapy Process | Phase II | CellTx Partners | CAR-T |
| P1-001 | ADC Conjugation | Phase I | Oncology Rx LLC | HER2-MMAE ADC |
| P1-002 | Plasmid DNA GMP | Phase I | GeneMed Corp. | pDNA |
| PRE-001 | Enzyme Replacement | Preclinical | RareDx Biotech | Lysosomal enzyme |
| P2-003 | Bispecific Antibody | On Hold | ImmunoCo | bsAb |

---

## Using Your Own Data

Replace or extend `create_sample_portfolio()` in `cdmo_margin/sample_data.py`, or build a `Portfolio` programmatically:

```python
from cdmo_margin.models import (
    Program, Portfolio, CostItem, RevenueItem,
    ProgramPhase, ProgramStatus, CostCategory, RevenueType,
)
from cdmo_margin.report import ReportGenerator

program = Program(
    program_id="MY-001",
    name="My Program",
    customer="Acme Pharma",
    molecule="mAb-XYZ",
    phase=ProgramPhase.PHASE_2,
    status=ProgramStatus.ACTIVE,
    annual_batches=6,
    fte_count=4.0,
    revenue_items=[
        RevenueItem(RevenueType.BATCH_FEE, "Batch manufacture", 1_200_000),
        RevenueItem(RevenueType.FTE_CHARGE, "Process dev FTE", 320_000),
    ],
    cost_items=[
        CostItem(CostCategory.DIRECT_MATERIALS, "Media, reagents", 300_000, is_variable=True),
        CostItem(CostCategory.DIRECT_LABOR,     "Manufacturing labor", 400_000, is_variable=False),
        CostItem(CostCategory.QC_QA,            "Release testing", 100_000, is_variable=True),
        CostItem(CostCategory.OVERHEAD,         "Facility overhead", 80_000, is_variable=False),
        CostItem(CostCategory.SGA,              "Account management", 40_000, is_variable=False),
    ],
)

portfolio = Portfolio(name="My CDMO", fiscal_year=2025, programs=[program])
print(ReportGenerator(portfolio).full_report_text())
```

---

## Tests

```bash
python tests/test_cdmo_margin.py
```

36 tests covering:
- Program margin math (gross profit, EBITDA, contribution margin, zero-revenue edge cases)
- Portfolio aggregation (active-only roll-up, gross margin arithmetic)
- Calculator breakdowns (phase, customer, cost category — all sum-to-total verified)
- Scenario monotonicity (price increase → margin increase, cost reduction → margin improvement)
- Edge cases (invalid program IDs, zero-batch programs, threshold boundary conditions)
- CSV export (row count, required column names)

---

## Requirements

No external dependencies — pure Python standard library (dataclasses, csv, copy, enum).

```bash
python --version   # Python 3.8+
python run_cdmo_margin.py
```

---

## LSTM Financial Models

This repository also contains two LSTM-based quantitative finance models:

- **`lstm_options/`** — LSTM Options Pricer (Black-Scholes approximation + extension)
- **`lstm_bl/`** — LSTM Black-Litterman Portfolio Optimiser

```bash
pip install torch numpy scipy pandas matplotlib

python run_options.py   # LSTM options pricing
python run_bl.py        # LSTM Black-Litterman portfolio
```

---

*CDMO margin logic follows standard pharma/biotech contract manufacturing P&L conventions. All figures in the sample portfolio are illustrative.*
