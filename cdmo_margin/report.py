"""Text report and CSV export for CDMO portfolio margin analysis."""

import csv
import os
from typing import List
from .models import Portfolio
from .calculator import MarginCalculator, format_currency, format_pct


class ReportGenerator:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.calc = MarginCalculator(portfolio)

    # --- Formatting helpers ---

    def _row(self, cells: List[str], widths: List[int]) -> str:
        return "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    def _divider(self, widths: List[int], char: str = "─") -> str:
        return "  " + "  ".join(char * w for w in widths)

    # --- Sections ---

    def portfolio_summary_text(self) -> str:
        s = self.calc.portfolio_summary()
        lines = [
            f"\n{'='*72}",
            f"  CDMO PROGRAM MARGIN REPORT  —  {s['portfolio_name'].upper()}  FY{s['fiscal_year']}",
            f"{'='*72}",
            f"  Active Programs  :  {s['active_programs']} of {s['total_programs']}",
            f"  Total Revenue    :  {format_currency(s['total_revenue'])}",
            f"  Total COGS       :  {format_currency(s['total_cogs'])}",
            f"  Gross Profit     :  {format_currency(s['total_gross_profit'])}",
            f"  Gross Margin     :  {format_pct(s['gross_margin_pct'])}",
            f"  EBITDA           :  {format_currency(s['total_ebitda'])}",
            f"  EBITDA Margin    :  {format_pct(s['ebitda_margin_pct'])}",
            f"{'='*72}",
        ]
        return "\n".join(lines)

    def program_table_text(self) -> str:
        programs = sorted(
            [self.calc.program_summary(p) for p in self.portfolio.active_programs()],
            key=lambda x: -x["revenue"],
        )
        headers = ["Program", "Customer", "Phase", "Revenue", "COGS", "Gross Profit", "GM%", "EBITDA", "EBITDA%"]
        widths  = [20,        16,          12,       10,        10,     14,             6,     10,        8]

        lines = [
            "\nPROGRAM MARGIN SUMMARY",
            self._divider(widths),
            self._row(headers, widths),
            self._divider(widths),
        ]
        for p in programs:
            lines.append(self._row([
                p["name"][:19],
                p["customer"][:15],
                p["phase"][:11],
                format_currency(p["revenue"]),
                format_currency(p["cogs"]),
                format_currency(p["gross_profit"]),
                format_pct(p["gross_margin_pct"]),
                format_currency(p["ebitda"]),
                format_pct(p["ebitda_margin_pct"]),
            ], widths))

        lines.append(self._divider(widths))
        total_rev = sum(p["revenue"] for p in programs)
        total_cogs = sum(p["cogs"] for p in programs)
        total_gp = sum(p["gross_profit"] for p in programs)
        total_eb = sum(p["ebitda"] for p in programs)
        gm = (total_gp / total_rev * 100) if total_rev else 0.0
        em = (total_eb / total_rev * 100) if total_rev else 0.0
        lines.append(self._row(
            ["TOTAL", "", "", format_currency(total_rev), format_currency(total_cogs),
             format_currency(total_gp), format_pct(gm), format_currency(total_eb), format_pct(em)],
            widths,
        ))
        lines.append(self._divider(widths))
        return "\n".join(lines)

    def phase_breakdown_text(self) -> str:
        bd = self.calc.phase_breakdown()
        headers = ["Phase", "Programs", "Revenue", "Gross Profit", "GM%", "EBITDA", "EBITDA%", "Rev Share"]
        widths  = [14,       10,          12,        14,             6,     10,        8,          10]
        lines = ["\nBREAKDOWN BY PHASE", self._divider(widths), self._row(headers, widths), self._divider(widths)]
        for b in bd:
            lines.append(self._row([
                b["phase"], str(b["program_count"]),
                format_currency(b["total_revenue"]), format_currency(b["total_gross_profit"]),
                format_pct(b["gross_margin_pct"]), format_currency(b["total_ebitda"]),
                format_pct(b["ebitda_margin_pct"]), format_pct(b["revenue_share_pct"]),
            ], widths))
        lines.append(self._divider(widths))
        return "\n".join(lines)

    def customer_breakdown_text(self) -> str:
        customers = self.calc.customer_breakdown()
        headers = ["Customer", "Programs", "Revenue", "Gross Profit", "GM%", "EBITDA", "Rev Share"]
        widths  = [22,          10,          12,        14,             6,     10,        10]
        lines = ["\nBREAKDOWN BY CUSTOMER", self._divider(widths), self._row(headers, widths), self._divider(widths)]
        for c in customers:
            lines.append(self._row([
                c["customer"][:21], str(c["programs"]),
                format_currency(c["revenue"]), format_currency(c["gross_profit"]),
                format_pct(c["gross_margin_pct"]), format_currency(c["ebitda"]),
                format_pct(c["revenue_share_pct"]),
            ], widths))
        lines.append(self._divider(widths))
        return "\n".join(lines)

    def cost_and_revenue_mix_text(self) -> str:
        cost_cats = self.calc.cost_category_breakdown()
        rev_types = self.calc.revenue_type_breakdown()

        lines = ["\nCOST CATEGORY BREAKDOWN"]
        total_cost = sum(cost_cats.values())
        for cat, amount in cost_cats.items():
            pct = (amount / total_cost * 100) if total_cost else 0.0
            bar = "█" * int(pct / 2)
            lines.append(f"  {cat:<22} {format_currency(amount):>10}   {pct:5.1f}%  {bar}")

        lines.append("\nREVENUE TYPE BREAKDOWN")
        total_rev = sum(rev_types.values())
        for rt, amount in rev_types.items():
            pct = (amount / total_rev * 100) if total_rev else 0.0
            bar = "█" * int(pct / 2)
            lines.append(f"  {rt:<22} {format_currency(amount):>10}   {pct:5.1f}%  {bar}")

        return "\n".join(lines)

    def at_risk_text(self, threshold: float = 20.0) -> str:
        at_risk = self.calc.at_risk_programs(threshold)
        if not at_risk:
            return f"\n  No active programs below {threshold:.0f}% gross margin threshold."
        lines = [f"\nAT-RISK PROGRAMS  (GM < {threshold:.0f}%)"]
        for p in at_risk:
            lines.append(
                f"  • {p['name']} ({p['customer']}) "
                f"  GM: {format_pct(p['gross_margin_pct'])}  "
                f"Revenue: {format_currency(p['revenue'])}"
            )
        return "\n".join(lines)

    def full_report_text(self, margin_threshold: float = 20.0) -> str:
        return "\n".join([
            self.portfolio_summary_text(),
            self.program_table_text(),
            self.phase_breakdown_text(),
            self.customer_breakdown_text(),
            self.cost_and_revenue_mix_text(),
            self.at_risk_text(margin_threshold),
            "",
        ])

    # --- CSV export ---

    def export_portfolio_csv(self, filepath: str) -> None:
        rows = [self.calc.program_summary(p) for p in self.portfolio.programs]
        if not rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
