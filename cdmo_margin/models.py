"""Data models for CDMO program margin analysis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict


class ProgramPhase(Enum):
    PRECLINICAL = "Preclinical"
    PHASE_1 = "Phase I"
    PHASE_2 = "Phase II"
    PHASE_3 = "Phase III"
    COMMERCIAL = "Commercial"
    DISCONTINUED = "Discontinued"


class ProgramStatus(Enum):
    ACTIVE = "Active"
    ON_HOLD = "On Hold"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"


class CostCategory(Enum):
    DIRECT_MATERIALS = "Direct Materials"
    DIRECT_LABOR = "Direct Labor"
    EQUIPMENT = "Equipment"
    CONSUMABLES = "Consumables"
    QC_QA = "QC/QA"
    OVERHEAD = "Overhead"
    SGA = "SG&A"
    RD = "R&D"


class RevenueType(Enum):
    BATCH_FEE = "Batch Fee"
    FTE_CHARGE = "FTE Charge"
    MILESTONE = "Milestone"
    TECHNOLOGY_TRANSFER = "Technology Transfer"
    ROYALTY = "Royalty"
    ANALYTICAL_SERVICES = "Analytical Services"
    STABILITY_STORAGE = "Stability/Storage"


@dataclass
class CostItem:
    category: CostCategory
    description: str
    annual_amount: float
    is_variable: bool = True


@dataclass
class RevenueItem:
    revenue_type: RevenueType
    description: str
    annual_amount: float


@dataclass
class Program:
    program_id: str
    name: str
    customer: str
    molecule: str
    phase: ProgramPhase
    status: ProgramStatus
    revenue_items: List[RevenueItem] = field(default_factory=list)
    cost_items: List[CostItem] = field(default_factory=list)
    annual_batches: int = 0
    fte_count: float = 0.0
    start_year: int = 2024

    # --- Revenue ---

    @property
    def total_revenue(self) -> float:
        return sum(r.annual_amount for r in self.revenue_items)

    # --- Cost buckets ---

    _COGS_CATEGORIES = frozenset({
        CostCategory.DIRECT_MATERIALS,
        CostCategory.DIRECT_LABOR,
        CostCategory.EQUIPMENT,
        CostCategory.CONSUMABLES,
        CostCategory.QC_QA,
        CostCategory.OVERHEAD,
    })

    @property
    def total_cogs(self) -> float:
        return sum(c.annual_amount for c in self.cost_items if c.category in self._COGS_CATEGORIES)

    @property
    def total_sga(self) -> float:
        return sum(c.annual_amount for c in self.cost_items if c.category == CostCategory.SGA)

    @property
    def total_rd(self) -> float:
        return sum(c.annual_amount for c in self.cost_items if c.category == CostCategory.RD)

    @property
    def variable_costs(self) -> float:
        return sum(c.annual_amount for c in self.cost_items if c.is_variable)

    @property
    def fixed_costs(self) -> float:
        return sum(c.annual_amount for c in self.cost_items if not c.is_variable)

    # --- Margin metrics ---

    @property
    def gross_profit(self) -> float:
        return self.total_revenue - self.total_cogs

    @property
    def gross_margin_pct(self) -> float:
        return (self.gross_profit / self.total_revenue * 100) if self.total_revenue else 0.0

    @property
    def ebitda(self) -> float:
        return self.total_revenue - self.total_cogs - self.total_sga - self.total_rd

    @property
    def ebitda_margin_pct(self) -> float:
        return (self.ebitda / self.total_revenue * 100) if self.total_revenue else 0.0

    @property
    def contribution_margin(self) -> float:
        return self.total_revenue - self.variable_costs

    @property
    def contribution_margin_pct(self) -> float:
        return (self.contribution_margin / self.total_revenue * 100) if self.total_revenue else 0.0


@dataclass
class Portfolio:
    name: str
    fiscal_year: int
    programs: List[Program] = field(default_factory=list)

    def active_programs(self) -> List[Program]:
        return [p for p in self.programs if p.status == ProgramStatus.ACTIVE]

    def programs_by_phase(self) -> Dict[ProgramPhase, List[Program]]:
        result: Dict[ProgramPhase, List[Program]] = {}
        for p in self.programs:
            result.setdefault(p.phase, []).append(p)
        return result

    @property
    def total_revenue(self) -> float:
        return sum(p.total_revenue for p in self.active_programs())

    @property
    def total_cogs(self) -> float:
        return sum(p.total_cogs for p in self.active_programs())

    @property
    def total_gross_profit(self) -> float:
        return self.total_revenue - self.total_cogs

    @property
    def portfolio_gross_margin_pct(self) -> float:
        return (self.total_gross_profit / self.total_revenue * 100) if self.total_revenue else 0.0

    @property
    def total_ebitda(self) -> float:
        return sum(p.ebitda for p in self.active_programs())

    @property
    def portfolio_ebitda_margin_pct(self) -> float:
        return (self.total_ebitda / self.total_revenue * 100) if self.total_revenue else 0.0
