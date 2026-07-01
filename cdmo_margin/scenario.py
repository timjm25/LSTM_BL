"""Scenario / what-if modeling for CDMO program margins."""

from copy import deepcopy
from typing import List, Dict, Optional
from .models import Portfolio, ProgramStatus
from .calculator import MarginCalculator


class ScenarioAnalyzer:
    def __init__(self, base_portfolio: Portfolio):
        self.base = base_portfolio
        self._base_calc = MarginCalculator(base_portfolio)

    def _mutate_program(
        self,
        portfolio: Portfolio,
        program_id: str,
        revenue_delta_pct: float = 0.0,
        variable_cost_delta_pct: float = 0.0,
        fixed_cost_delta_pct: float = 0.0,
    ) -> None:
        for p in portfolio.programs:
            if p.program_id != program_id:
                continue
            for r in p.revenue_items:
                r.annual_amount *= 1 + revenue_delta_pct / 100
            for c in p.cost_items:
                if c.is_variable:
                    c.annual_amount *= 1 + variable_cost_delta_pct / 100
                else:
                    c.annual_amount *= 1 + fixed_cost_delta_pct / 100

    # --- Program-level scenarios ---

    def price_sensitivity(
        self,
        program_id: str,
        price_changes_pct: Optional[List[float]] = None,
    ) -> List[Dict]:
        if price_changes_pct is None:
            price_changes_pct = [-20, -10, -5, 0, 5, 10, 20]

        base_prog = next((p for p in self.base.programs if p.program_id == program_id), None)
        if not base_prog:
            return []

        base_rev = base_prog.total_revenue
        base_gm = base_prog.gross_margin_pct
        results = []
        for delta in price_changes_pct:
            s = deepcopy(self.base)
            self._mutate_program(s, program_id, revenue_delta_pct=delta)
            prog = next(p for p in s.programs if p.program_id == program_id)
            results.append({
                "price_change_pct": delta,
                "revenue": prog.total_revenue,
                "revenue_delta": prog.total_revenue - base_rev,
                "gross_profit": prog.gross_profit,
                "gross_margin_pct": prog.gross_margin_pct,
                "gross_margin_delta_pp": prog.gross_margin_pct - base_gm,
                "ebitda": prog.ebitda,
                "ebitda_margin_pct": prog.ebitda_margin_pct,
            })
        return results

    def volume_sensitivity(
        self,
        program_id: str,
        batch_changes: Optional[List[int]] = None,
    ) -> List[Dict]:
        if batch_changes is None:
            batch_changes = [-4, -2, -1, 0, 1, 2, 4]

        base_prog = next((p for p in self.base.programs if p.program_id == program_id), None)
        if not base_prog or base_prog.annual_batches == 0:
            return []

        base_batches = base_prog.annual_batches
        rev_per_batch = base_prog.total_revenue / base_batches
        var_cost_per_batch = base_prog.variable_costs / base_batches
        fixed = base_prog.fixed_costs
        base_gm = base_prog.gross_margin_pct

        results = []
        for delta in batch_changes:
            new_batches = max(0, base_batches + delta)
            new_rev = rev_per_batch * new_batches
            new_var = var_cost_per_batch * new_batches
            new_cogs = new_var + fixed
            new_gp = new_rev - new_cogs
            new_gm = (new_gp / new_rev * 100) if new_rev else 0.0
            results.append({
                "batch_delta": delta,
                "annual_batches": new_batches,
                "revenue": new_rev,
                "cogs": new_cogs,
                "gross_profit": new_gp,
                "gross_margin_pct": new_gm,
                "gross_margin_delta_pp": new_gm - base_gm,
            })
        return results

    def cost_reduction_impact(
        self,
        program_id: str,
        reduction_pcts: Optional[List[float]] = None,
    ) -> List[Dict]:
        if reduction_pcts is None:
            reduction_pcts = [0, 5, 10, 15, 20]

        base_prog = next((p for p in self.base.programs if p.program_id == program_id), None)
        if not base_prog:
            return []

        base_cogs = base_prog.total_cogs
        base_gm = base_prog.gross_margin_pct
        results = []
        for reduction in reduction_pcts:
            s = deepcopy(self.base)
            self._mutate_program(s, program_id,
                                  variable_cost_delta_pct=-reduction,
                                  fixed_cost_delta_pct=-reduction)
            prog = next(p for p in s.programs if p.program_id == program_id)
            results.append({
                "cost_reduction_pct": reduction,
                "cogs": prog.total_cogs,
                "cogs_saved": base_cogs - prog.total_cogs,
                "gross_profit": prog.gross_profit,
                "gross_margin_pct": prog.gross_margin_pct,
                "gross_margin_delta_pp": prog.gross_margin_pct - base_gm,
                "ebitda": prog.ebitda,
                "ebitda_margin_pct": prog.ebitda_margin_pct,
            })
        return results

    # --- Portfolio-level scenario ---

    def portfolio_scenario(
        self,
        revenue_delta_pct: float = 0.0,
        variable_cost_delta_pct: float = 0.0,
        fixed_cost_delta_pct: float = 0.0,
    ) -> Dict:
        s = deepcopy(self.base)
        for p in s.programs:
            if p.status == ProgramStatus.ACTIVE:
                self._mutate_program(s, p.program_id,
                                      revenue_delta_pct=revenue_delta_pct,
                                      variable_cost_delta_pct=variable_cost_delta_pct,
                                      fixed_cost_delta_pct=fixed_cost_delta_pct)
        base_sum = self._base_calc.portfolio_summary()
        scen_sum = MarginCalculator(s).portfolio_summary()
        return {
            "base": base_sum,
            "scenario": scen_sum,
            "revenue_change": scen_sum["total_revenue"] - base_sum["total_revenue"],
            "gross_profit_change": scen_sum["total_gross_profit"] - base_sum["total_gross_profit"],
            "ebitda_change": scen_sum["total_ebitda"] - base_sum["total_ebitda"],
            "gross_margin_delta_pp": scen_sum["gross_margin_pct"] - base_sum["gross_margin_pct"],
            "ebitda_margin_delta_pp": scen_sum["ebitda_margin_pct"] - base_sum["ebitda_margin_pct"],
        }
