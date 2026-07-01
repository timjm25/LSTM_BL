"""Unit tests for the CDMO Program Margin Analyzer."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cdmo_margin.models import (
    Program, Portfolio, CostItem, RevenueItem,
    ProgramPhase, ProgramStatus, CostCategory, RevenueType,
)
from cdmo_margin.calculator import MarginCalculator
from cdmo_margin.report import ReportGenerator
from cdmo_margin.scenario import ScenarioAnalyzer
from cdmo_margin.sample_data import create_sample_portfolio


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _program(program_id="T-001", revenue=1_000_000, cogs=600_000, sga=50_000) -> Program:
    p = Program(
        program_id=program_id, name="Test Program", customer="TestCo",
        molecule="Drug A", phase=ProgramPhase.COMMERCIAL, status=ProgramStatus.ACTIVE,
        annual_batches=10, fte_count=5.0,
    )
    p.revenue_items = [RevenueItem(RevenueType.BATCH_FEE, "Batch fee", revenue)]
    p.cost_items = [
        CostItem(CostCategory.DIRECT_MATERIALS, "Materials", cogs * 0.4, is_variable=True),
        CostItem(CostCategory.DIRECT_LABOR,     "Labor",     cogs * 0.4, is_variable=False),
        CostItem(CostCategory.OVERHEAD,         "Overhead",  cogs * 0.2, is_variable=False),
        CostItem(CostCategory.SGA,              "SGA",       sga,        is_variable=False),
    ]
    return p


# ── Program model ─────────────────────────────────────────────────────────────

class TestProgramModel(unittest.TestCase):
    def setUp(self):
        self.p = _program()

    def test_total_revenue(self):
        self.assertEqual(self.p.total_revenue, 1_000_000)

    def test_total_cogs(self):
        self.assertAlmostEqual(self.p.total_cogs, 600_000)

    def test_gross_profit(self):
        self.assertAlmostEqual(self.p.gross_profit, 400_000)

    def test_gross_margin_pct(self):
        self.assertAlmostEqual(self.p.gross_margin_pct, 40.0)

    def test_ebitda(self):
        self.assertAlmostEqual(self.p.ebitda, 350_000)

    def test_ebitda_margin_pct(self):
        self.assertAlmostEqual(self.p.ebitda_margin_pct, 35.0)

    def test_variable_costs(self):
        self.assertAlmostEqual(self.p.variable_costs, 240_000)

    def test_fixed_costs(self):
        self.assertAlmostEqual(self.p.fixed_costs, 410_000)

    def test_contribution_margin(self):
        self.assertAlmostEqual(self.p.contribution_margin, 760_000)

    def test_zero_revenue_returns_zero_margins(self):
        p = Program(
            program_id="Z", name="Z", customer="Z", molecule="Z",
            phase=ProgramPhase.PRECLINICAL, status=ProgramStatus.ACTIVE,
        )
        self.assertEqual(p.gross_margin_pct, 0.0)
        self.assertEqual(p.ebitda_margin_pct, 0.0)
        self.assertEqual(p.contribution_margin_pct, 0.0)


# ── Portfolio model ───────────────────────────────────────────────────────────

class TestPortfolioModel(unittest.TestCase):
    def setUp(self):
        p1 = _program("P1", 1_000_000, 600_000)
        p2 = _program("P2", 2_000_000, 1_200_000)
        p3 = _program("P3", 500_000, 300_000)
        p3.status = ProgramStatus.ON_HOLD
        self.portfolio = Portfolio("Test", 2025, [p1, p2, p3])

    def test_total_revenue_excludes_on_hold(self):
        self.assertAlmostEqual(self.portfolio.total_revenue, 3_000_000)

    def test_gross_margin(self):
        self.assertAlmostEqual(self.portfolio.portfolio_gross_margin_pct, 40.0)

    def test_active_count(self):
        self.assertEqual(len(self.portfolio.active_programs()), 2)


# ── MarginCalculator ──────────────────────────────────────────────────────────

class TestMarginCalculator(unittest.TestCase):
    def setUp(self):
        self.portfolio = create_sample_portfolio(2025)
        self.calc = MarginCalculator(self.portfolio)

    def test_portfolio_summary_has_required_keys(self):
        s = self.calc.portfolio_summary()
        for k in ["total_revenue", "total_cogs", "gross_margin_pct", "ebitda_margin_pct", "active_programs"]:
            self.assertIn(k, s)

    def test_portfolio_revenue_positive(self):
        self.assertGreater(self.calc.portfolio_summary()["total_revenue"], 0)

    def test_portfolio_gross_margin_bounded(self):
        gm = self.calc.portfolio_summary()["gross_margin_pct"]
        self.assertGreater(gm, 0)
        self.assertLess(gm, 100)

    def test_phase_breakdown_sums_to_portfolio_revenue(self):
        bd = self.calc.phase_breakdown()
        self.assertAlmostEqual(
            sum(b["total_revenue"] for b in bd),
            self.portfolio.total_revenue,
            places=0,
        )

    def test_customer_breakdown_non_empty(self):
        self.assertGreater(len(self.calc.customer_breakdown()), 0)

    def test_at_risk_threshold_100_returns_all_active(self):
        n = len(self.portfolio.active_programs())
        self.assertEqual(len(self.calc.at_risk_programs(100.0)), n)

    def test_at_risk_threshold_0_returns_none(self):
        self.assertEqual(len(self.calc.at_risk_programs(0.0)), 0)

    def test_top_programs_by_margin_sorted(self):
        top = self.calc.top_programs_by_margin(3)
        margins = [p["gross_margin_pct"] for p in top]
        self.assertEqual(margins, sorted(margins, reverse=True))

    def test_cost_per_batch_has_positive_batches(self):
        for b in self.calc.cost_per_batch():
            self.assertGreater(b["annual_batches"], 0)
            self.assertGreater(b["revenue_per_batch"], 0)

    def test_program_summary_fields_match_model(self):
        prog = self.portfolio.programs[0]
        s = self.calc.program_summary(prog)
        self.assertEqual(s["program_id"], prog.program_id)
        self.assertAlmostEqual(s["revenue"], prog.total_revenue)
        self.assertAlmostEqual(s["gross_margin_pct"], prog.gross_margin_pct)


# ── ScenarioAnalyzer ──────────────────────────────────────────────────────────

class TestScenarioAnalyzer(unittest.TestCase):
    def setUp(self):
        self.portfolio = create_sample_portfolio(2025)
        self.analyzer = ScenarioAnalyzer(self.portfolio)
        self.prog_id = self.portfolio.programs[0].program_id

    def test_price_sensitivity_returns_results(self):
        self.assertGreater(len(self.analyzer.price_sensitivity(self.prog_id)), 0)

    def test_price_zero_delta_matches_base(self):
        base = next(p for p in self.portfolio.programs if p.program_id == self.prog_id)
        result = self.analyzer.price_sensitivity(self.prog_id, [0])
        self.assertAlmostEqual(result[0]["gross_margin_pct"], base.gross_margin_pct, places=5)

    def test_price_increase_improves_margin(self):
        results = self.analyzer.price_sensitivity(self.prog_id, [-10, 0, 10])
        margins = [r["gross_margin_pct"] for r in results]
        self.assertGreater(margins[2], margins[1])
        self.assertLess(margins[0], margins[1])

    def test_volume_sensitivity_for_batch_program(self):
        batch_prog = next(p for p in self.portfolio.programs if p.annual_batches > 0)
        results = self.analyzer.volume_sensitivity(batch_prog.program_id)
        self.assertGreater(len(results), 0)

    def test_volume_sensitivity_no_batches_returns_empty(self):
        no_batch = next(p for p in self.portfolio.programs if p.annual_batches == 0)
        self.assertEqual(self.analyzer.volume_sensitivity(no_batch.program_id), [])

    def test_cost_reduction_improves_margin(self):
        results = self.analyzer.cost_reduction_impact(self.prog_id, [0, 10])
        self.assertGreaterEqual(results[1]["gross_margin_pct"], results[0]["gross_margin_pct"])

    def test_portfolio_base_scenario_unchanged(self):
        r = self.analyzer.portfolio_scenario(0, 0, 0)
        self.assertAlmostEqual(r["base"]["total_revenue"], r["scenario"]["total_revenue"], places=0)

    def test_portfolio_revenue_increase(self):
        r = self.analyzer.portfolio_scenario(revenue_delta_pct=10)
        self.assertGreater(r["scenario"]["total_revenue"], r["base"]["total_revenue"])

    def test_invalid_program_id_returns_empty(self):
        self.assertEqual(self.analyzer.price_sensitivity("DOES-NOT-EXIST"), [])


# ── ReportGenerator ───────────────────────────────────────────────────────────

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.portfolio = create_sample_portfolio(2025)
        self.report = ReportGenerator(self.portfolio)

    def test_full_report_non_empty(self):
        text = self.report.full_report_text()
        self.assertGreater(len(text), 100)
        self.assertIn("CDMO PROGRAM MARGIN REPORT", text)

    def test_program_table_lists_all_active(self):
        text = self.report.program_table_text()
        for p in self.portfolio.active_programs():
            self.assertIn(p.name[:10], text)

    def test_at_risk_empty_when_threshold_zero(self):
        text = self.report.at_risk_text(0.0)
        self.assertIn("No active programs", text)

    def test_csv_export_has_all_programs(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.report.export_portfolio_csv(path)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len(self.portfolio.programs))
            self.assertIn("gross_margin_pct", rows[0])
            self.assertIn("program_id", rows[0])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
