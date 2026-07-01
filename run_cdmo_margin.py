#!/usr/bin/env python3
"""CDMO Program Margin Analyzer — CLI entry point."""

import argparse
import os
import sys


def _col(cells, widths):
    return "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def _div(widths, char="─"):
    return "  " + "  ".join(char * w for w in widths)


def main():
    parser = argparse.ArgumentParser(
        description="CDMO Program Margin Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cdmo_margin.py                             # Full portfolio report
  python run_cdmo_margin.py --program COM-001           # Single program detail
  python run_cdmo_margin.py --scenario price --program P3-001
  python run_cdmo_margin.py --scenario volume --program COM-002
  python run_cdmo_margin.py --scenario cost --program P2-001
  python run_cdmo_margin.py --scenario portfolio        # Portfolio what-if
  python run_cdmo_margin.py --export results.csv        # Export to CSV
  python run_cdmo_margin.py --export-dir ./reports/     # Export all to dir
  python run_cdmo_margin.py --threshold 25              # Custom at-risk threshold
        """,
    )
    parser.add_argument("--fiscal-year", type=int, default=2025)
    parser.add_argument("--program", type=str, help="Program ID for detail or scenario analysis")
    parser.add_argument("--scenario", choices=["price", "volume", "cost", "portfolio"],
                        help="Scenario / what-if analysis mode")
    parser.add_argument("--threshold", type=float, default=20.0,
                        help="Gross margin %% below which programs are flagged at-risk (default: 20)")
    parser.add_argument("--export", type=str, metavar="FILE", help="Export portfolio CSV")
    parser.add_argument("--export-dir", type=str, metavar="DIR", help="Export report + CSV to directory")
    args = parser.parse_args()

    from cdmo_margin.sample_data import create_sample_portfolio
    from cdmo_margin.calculator import MarginCalculator, format_currency, format_pct
    from cdmo_margin.report import ReportGenerator
    from cdmo_margin.scenario import ScenarioAnalyzer

    portfolio = create_sample_portfolio(fiscal_year=args.fiscal_year)
    calc = MarginCalculator(portfolio)
    report = ReportGenerator(portfolio)

    # ── CSV / dir export ───────────────────────────────────────────────────
    if args.export:
        report.export_portfolio_csv(args.export)
        print(f"Exported → {args.export}")
        return

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        csv_path = os.path.join(args.export_dir, "portfolio.csv")
        txt_path = os.path.join(args.export_dir, "report.txt")
        report.export_portfolio_csv(csv_path)
        with open(txt_path, "w") as f:
            f.write(report.full_report_text(args.threshold))
        print(f"Exported → {csv_path}\nExported → {txt_path}")
        return

    # ── Single-program detail ──────────────────────────────────────────────
    if args.program and not args.scenario:
        prog = next((p for p in portfolio.programs if p.program_id == args.program), None)
        if not prog:
            ids = ", ".join(p.program_id for p in portfolio.programs)
            print(f"Program '{args.program}' not found.\nAvailable IDs: {ids}")
            sys.exit(1)

        s = calc.program_summary(prog)
        print(f"\n{'='*62}")
        print(f"  PROGRAM DETAIL — {prog.name}")
        print(f"{'='*62}")
        print(f"  ID          : {prog.program_id}")
        print(f"  Customer    : {prog.customer}")
        print(f"  Molecule    : {prog.molecule}")
        print(f"  Phase       : {prog.phase.value}")
        print(f"  Status      : {prog.status.value}")
        print(f"  Batches/yr  : {prog.annual_batches}")
        print(f"  FTEs        : {prog.fte_count:.1f}")
        print(f"{'─'*62}")
        print(f"  Revenue              : {format_currency(s['revenue'])}")
        print(f"  COGS                 : {format_currency(s['cogs'])}")
        print(f"  Gross Profit         : {format_currency(s['gross_profit'])}")
        print(f"  Gross Margin         : {format_pct(s['gross_margin_pct'])}")
        print(f"  SG&A                 : {format_currency(s['sga'])}")
        print(f"  EBITDA               : {format_currency(s['ebitda'])}")
        print(f"  EBITDA Margin        : {format_pct(s['ebitda_margin_pct'])}")
        print(f"{'─'*62}")
        print(f"  Variable Costs       : {format_currency(s['variable_costs'])}")
        print(f"  Fixed Costs          : {format_currency(s['fixed_costs'])}")
        print(f"  Contribution Margin  : {format_currency(s['contribution_margin'])}  "
              f"({format_pct(s['contribution_margin_pct'])})")
        print(f"\n  Revenue Breakdown:")
        for r in prog.revenue_items:
            pct = (r.annual_amount / prog.total_revenue * 100) if prog.total_revenue else 0.0
            print(f"    {r.revenue_type.value:<28} {format_currency(r.annual_amount):>10}  ({pct:.1f}%)")
        print(f"\n  Cost Breakdown:")
        for c in sorted(prog.cost_items, key=lambda x: -x.annual_amount):
            flag = "V" if c.is_variable else "F"
            print(f"    [{flag}] {c.category.value:<22} {format_currency(c.annual_amount):>10}   {c.description}")
        print()
        return

    # ── Scenario analysis ──────────────────────────────────────────────────
    if args.scenario:
        analyzer = ScenarioAnalyzer(portfolio)

        if args.scenario == "portfolio":
            print("\nPORTFOLIO SCENARIO ANALYSIS")
            print("=" * 62)
            scenarios = [
                ("Base Case",                          0,    0,    0),
                ("Revenue  −10%",                    -10,    0,    0),
                ("Revenue  +10%",                     10,    0,    0),
                ("COGS     +10%  (inflation)",          0,   10,   10),
                ("COGS     −10%  (savings initiative)", 0,  -10,  -10),
                ("Rev +5%, Cost −5%",                   5,   -5,   -5),
                ("Bear: Rev −15%, Cost +10%",         -15,   10,   10),
            ]
            headers = ["Scenario", "Revenue", "Gross Profit", "GM%", "EBITDA", "EBITDA%"]
            widths  = [38,          12,         14,             6,     10,        8]
            print(_col(headers, widths))
            print(_div(widths))
            for label, rd, vd, fd in scenarios:
                r = analyzer.portfolio_scenario(rd, vd, fd)["scenario"]
                print(_col([
                    label,
                    format_currency(r["total_revenue"]),
                    format_currency(r["total_gross_profit"]),
                    format_pct(r["gross_margin_pct"]),
                    format_currency(r["total_ebitda"]),
                    format_pct(r["ebitda_margin_pct"]),
                ], widths))
            print()
            return

        # Program-specific
        prog_id = args.program
        if not prog_id:
            active = portfolio.active_programs()
            if not active:
                print("No active programs.")
                sys.exit(1)
            prog_id = active[0].program_id
            print(f"No --program specified; defaulting to {prog_id}")

        prog = next((p for p in portfolio.programs if p.program_id == prog_id), None)
        if not prog:
            ids = ", ".join(p.program_id for p in portfolio.programs)
            print(f"Program '{prog_id}' not found.\nAvailable IDs: {ids}")
            sys.exit(1)

        print(f"\nSCENARIO ANALYSIS — {prog.name}  ({prog_id})")
        print("=" * 62)

        if args.scenario == "price":
            results = analyzer.price_sensitivity(prog_id)
            headers = ["Price Δ%", "Revenue", "Gross Profit", "GM%", "GM Δpp", "EBITDA"]
            widths  = [10,          12,         14,             6,     8,         10]
            print(_col(headers, widths))
            print(_div(widths))
            for r in results:
                marker = "  ◄ base" if r["price_change_pct"] == 0 else ""
                print(_col([
                    f"{r['price_change_pct']:+.0f}%",
                    format_currency(r["revenue"]),
                    format_currency(r["gross_profit"]),
                    format_pct(r["gross_margin_pct"]),
                    f"{r['gross_margin_delta_pp']:+.1f}pp",
                    format_currency(r["ebitda"]),
                ], widths) + marker)

        elif args.scenario == "volume":
            results = analyzer.volume_sensitivity(prog_id)
            if not results:
                print(f"  {prog_id} has no batch volume to model.")
            else:
                headers = ["Batch Δ", "Batches", "Revenue", "Gross Profit", "GM%", "GM Δpp"]
                widths  = [8,          8,          12,         14,             6,     8]
                print(_col(headers, widths))
                print(_div(widths))
                for r in results:
                    marker = "  ◄ base" if r["batch_delta"] == 0 else ""
                    print(_col([
                        f"{r['batch_delta']:+d}",
                        str(r["annual_batches"]),
                        format_currency(r["revenue"]),
                        format_currency(r["gross_profit"]),
                        format_pct(r["gross_margin_pct"]),
                        f"{r['gross_margin_delta_pp']:+.1f}pp",
                    ], widths) + marker)

        elif args.scenario == "cost":
            results = analyzer.cost_reduction_impact(prog_id)
            headers = ["Cost Cut%", "COGS", "Savings", "Gross Profit", "GM%", "GM Δpp"]
            widths  = [10,           12,      12,         14,             6,     8]
            print(_col(headers, widths))
            print(_div(widths))
            for r in results:
                marker = "  ◄ base" if r["cost_reduction_pct"] == 0 else ""
                print(_col([
                    f"{r['cost_reduction_pct']:.0f}%",
                    format_currency(r["cogs"]),
                    format_currency(r["cogs_saved"]),
                    format_currency(r["gross_profit"]),
                    format_pct(r["gross_margin_pct"]),
                    f"{r['gross_margin_delta_pp']:+.1f}pp",
                ], widths) + marker)

        print()
        return

    # ── Default: full portfolio report + batch economics ──────────────────
    print(report.full_report_text(args.threshold))

    batch_data = calc.cost_per_batch()
    if batch_data:
        print("BATCH ECONOMICS")
        headers = ["Program", "Phase", "Batches/yr", "Rev/Batch", "COGS/Batch", "GP/Batch", "GM%"]
        widths  = [22,         12,       12,            12,          12,            12,         6]
        print(_col(headers, widths))
        print(_div(widths))
        for b in batch_data:
            print(_col([
                b["name"][:21],
                b["phase"][:11],
                str(b["annual_batches"]),
                format_currency(b["revenue_per_batch"]),
                format_currency(b["cogs_per_batch"]),
                format_currency(b["gross_profit_per_batch"]),
                format_pct(b["gross_margin_pct"]),
            ], widths))
        print()


if __name__ == "__main__":
    main()
