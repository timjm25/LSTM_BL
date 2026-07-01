"""Margin calculator — aggregations and breakdowns across a CDMO portfolio."""

from typing import List, Dict
from .models import Program, Portfolio, ProgramPhase, ProgramStatus


def format_currency(amount: float) -> str:
    return f"${amount / 1_000:,.1f}K"


def format_pct(pct: float) -> str:
    return f"{pct:.1f}%"


class MarginCalculator:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    # --- Per-program ---

    def program_summary(self, program: Program) -> Dict:
        return {
            "program_id": program.program_id,
            "name": program.name,
            "customer": program.customer,
            "molecule": program.molecule,
            "phase": program.phase.value,
            "status": program.status.value,
            "annual_batches": program.annual_batches,
            "fte_count": program.fte_count,
            "revenue": program.total_revenue,
            "cogs": program.total_cogs,
            "gross_profit": program.gross_profit,
            "gross_margin_pct": program.gross_margin_pct,
            "sga": program.total_sga,
            "rd": program.total_rd,
            "ebitda": program.ebitda,
            "ebitda_margin_pct": program.ebitda_margin_pct,
            "variable_costs": program.variable_costs,
            "fixed_costs": program.fixed_costs,
            "contribution_margin": program.contribution_margin,
            "contribution_margin_pct": program.contribution_margin_pct,
        }

    # --- Portfolio roll-up ---

    def portfolio_summary(self) -> Dict:
        active = self.portfolio.active_programs()
        return {
            "portfolio_name": self.portfolio.name,
            "fiscal_year": self.portfolio.fiscal_year,
            "active_programs": len(active),
            "total_programs": len(self.portfolio.programs),
            "total_revenue": self.portfolio.total_revenue,
            "total_cogs": self.portfolio.total_cogs,
            "total_gross_profit": self.portfolio.total_gross_profit,
            "gross_margin_pct": self.portfolio.portfolio_gross_margin_pct,
            "total_ebitda": self.portfolio.total_ebitda,
            "ebitda_margin_pct": self.portfolio.portfolio_ebitda_margin_pct,
        }

    # --- Breakdowns ---

    def phase_breakdown(self) -> List[Dict]:
        breakdown = []
        total_rev = self.portfolio.total_revenue
        for phase in ProgramPhase:
            programs = [
                p for p in self.portfolio.programs_by_phase().get(phase, [])
                if p.status == ProgramStatus.ACTIVE
            ]
            if not programs:
                continue
            rev = sum(p.total_revenue for p in programs)
            gp = sum(p.gross_profit for p in programs)
            eb = sum(p.ebitda for p in programs)
            breakdown.append({
                "phase": phase.value,
                "program_count": len(programs),
                "total_revenue": rev,
                "total_gross_profit": gp,
                "gross_margin_pct": (gp / rev * 100) if rev else 0.0,
                "total_ebitda": eb,
                "ebitda_margin_pct": (eb / rev * 100) if rev else 0.0,
                "revenue_share_pct": (rev / total_rev * 100) if total_rev else 0.0,
            })
        return breakdown

    def customer_breakdown(self) -> List[Dict]:
        buckets: Dict[str, Dict] = {}
        for p in self.portfolio.active_programs():
            b = buckets.setdefault(p.customer, {"revenue": 0.0, "gross_profit": 0.0, "ebitda": 0.0, "programs": 0})
            b["revenue"] += p.total_revenue
            b["gross_profit"] += p.gross_profit
            b["ebitda"] += p.ebitda
            b["programs"] += 1

        total_rev = self.portfolio.total_revenue
        result = []
        for customer, b in sorted(buckets.items(), key=lambda x: -x[1]["revenue"]):
            rev, gp, eb = b["revenue"], b["gross_profit"], b["ebitda"]
            result.append({
                "customer": customer,
                "programs": b["programs"],
                "revenue": rev,
                "gross_profit": gp,
                "gross_margin_pct": (gp / rev * 100) if rev else 0.0,
                "ebitda": eb,
                "ebitda_margin_pct": (eb / rev * 100) if rev else 0.0,
                "revenue_share_pct": (rev / total_rev * 100) if total_rev else 0.0,
            })
        return result

    def cost_category_breakdown(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for p in self.portfolio.active_programs():
            for c in p.cost_items:
                totals[c.category.value] = totals.get(c.category.value, 0.0) + c.annual_amount
        return dict(sorted(totals.items(), key=lambda x: -x[1]))

    def revenue_type_breakdown(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for p in self.portfolio.active_programs():
            for r in p.revenue_items:
                totals[r.revenue_type.value] = totals.get(r.revenue_type.value, 0.0) + r.annual_amount
        return dict(sorted(totals.items(), key=lambda x: -x[1]))

    # --- Rankings / flags ---

    def top_programs_by_margin(self, n: int = 5) -> List[Dict]:
        active = sorted(self.portfolio.active_programs(), key=lambda p: -p.gross_margin_pct)
        return [self.program_summary(p) for p in active[:n]]

    def top_programs_by_revenue(self, n: int = 5) -> List[Dict]:
        active = sorted(self.portfolio.active_programs(), key=lambda p: -p.total_revenue)
        return [self.program_summary(p) for p in active[:n]]

    def at_risk_programs(self, margin_threshold: float = 20.0) -> List[Dict]:
        at_risk = [p for p in self.portfolio.active_programs() if p.gross_margin_pct < margin_threshold]
        return [self.program_summary(p) for p in sorted(at_risk, key=lambda p: p.gross_margin_pct)]

    def cost_per_batch(self) -> List[Dict]:
        result = []
        for p in self.portfolio.active_programs():
            if p.annual_batches > 0:
                result.append({
                    "program_id": p.program_id,
                    "name": p.name,
                    "phase": p.phase.value,
                    "annual_batches": p.annual_batches,
                    "revenue_per_batch": p.total_revenue / p.annual_batches,
                    "cogs_per_batch": p.total_cogs / p.annual_batches,
                    "gross_profit_per_batch": p.gross_profit / p.annual_batches,
                    "gross_margin_pct": p.gross_margin_pct,
                })
        return sorted(result, key=lambda x: -x["gross_profit_per_batch"])
