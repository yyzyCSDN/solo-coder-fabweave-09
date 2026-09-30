"""Furnace tube batch formation.

A furnace tube processes a batch of lots that share one recipe family.  A
valid batch must also respect the tube load range, keep contamination
compatible across its lots, fall inside every lot's delivery window at the
decision instant, and match the tool's equipment qualifications.

Formation is deterministic: the same candidate set always yields the same
batches regardless of input order, because lots are processed in stable
``lot_id`` order and batch identifiers are derived from their contents.  Every
lot is used at most once, and lots that cannot be placed are returned with
explicit rejection reasons so their disposition is fully accounted for.
"""

from __future__ import annotations

from dataclasses import dataclass

from fabweave.contamination.segregation import compatible, incompatible_pairs
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.ids import deterministic_id
from fabweave.core.timebox import TimeBox


@dataclass(frozen=True, slots=True)
class FurnaceLot:
    """A candidate WIP lot that may be loaded into a furnace batch."""

    lot_id: str
    recipe_family: str
    contamination_class: str
    quantity: int
    window: TimeBox | None = None
    active_holds: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Furnace:
    """A furnace tube with its load range and equipment capabilities."""

    tool_id: str
    min_load: int
    max_load: int
    qualified_families: frozenset[str] = frozenset()
    allowed_contamination: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.min_load <= 0 or self.max_load < self.min_load:
            raise ValueError("furnace load range is invalid")


@dataclass(frozen=True, slots=True)
class Batch:
    """A formed furnace batch: one tool, one recipe family, one load."""

    batch_id: str
    tool_id: str
    recipe_family: str
    lot_ids: tuple[str, ...]
    load: int


@dataclass(frozen=True, slots=True)
class Rejection:
    """A lot that could not be placed, with the reasons for its disposition."""

    lot_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchingResult:
    """The outcome of forming batches: placed batches and rejected lots."""

    tool_id: str
    batches: tuple[Batch, ...]
    rejected: tuple[Rejection, ...]

    @property
    def total_load(self) -> int:
        return sum(batch.load for batch in self.batches)

    @property
    def placed_lot_count(self) -> int:
        return sum(len(batch.lot_ids) for batch in self.batches)

    @property
    def rejected_lot_count(self) -> int:
        return len(self.rejected)

    def accounts_for(self, lot_ids: set[str]) -> bool:
        """True exactly when every lot is placed once or rejected once."""

        placed = {lot_id for batch in self.batches for lot_id in batch.lot_ids}
        rejected = {rejection.lot_id for rejection in self.rejected}
        return placed | rejected == set(lot_ids) and not placed & rejected

    def disposition(self) -> dict[str, str]:
        """Map every lot id to its batch id or its rejection reasons."""

        outcome = {
            lot_id: f"batch:{batch.batch_id}"
            for batch in self.batches
            for lot_id in batch.lot_ids
        }
        for rejection in self.rejected:
            outcome[rejection.lot_id] = "rejected:" + ",".join(rejection.reasons)
        return outcome


def _validate_lots(lots: list[FurnaceLot]) -> None:
    if any(not isinstance(lot.quantity, int) for lot in lots):
        raise ValueError("lot quantities must be integers")
    lot_ids = [lot.lot_id for lot in lots]
    if len(set(lot_ids)) != len(lot_ids):
        raise ValueError("the same lot cannot be batched twice")


def _hard_gate_reasons(lot: FurnaceLot, furnace: Furnace, at: int) -> tuple[str, ...]:
    reasons: list[str] = []
    if lot.active_holds:
        reasons.append("lot_held")
    if lot.window is not None and not lot.window.contains(at):
        reasons.append("outside_due_window")
    if lot.recipe_family not in furnace.qualified_families:
        reasons.append("recipe_not_qualified")
    if lot.contamination_class not in furnace.allowed_contamination:
        reasons.append("contamination_not_allowed")
    if lot.quantity <= 0:
        reasons.append("invalid_load")
    elif lot.quantity > furnace.max_load:
        reasons.append("load_exceeds_max")
    return tuple(reasons)


def batch_decision(
    furnace: Furnace,
    lots: list[FurnaceLot],
    at: int,
    contamination_matrix: dict[str, set[str]] | None = None,
) -> Decision:
    """Validate that a set of lots forms one legal furnace batch.

    Checks load range, recipe family unity, contamination compatibility, the
    lots' delivery windows and the tool's equipment capabilities together.
    """

    _validate_lots(lots)
    if not lots:
        return deny("empty_batch")
    matrix = contamination_matrix or {}
    reasons: list[str] = []
    if len({lot.recipe_family for lot in lots}) > 1:
        reasons.append("mixed_recipe_family")
    classes = {lot.contamination_class for lot in lots}
    for left, right in incompatible_pairs(classes, matrix):
        reasons.append(f"incompatible_contamination:{left}:{right}")
    load = sum(lot.quantity for lot in lots)
    if load < furnace.min_load:
        reasons.append("load_below_min")
    if load > furnace.max_load:
        reasons.append("load_exceeds_max")
    for lot in lots:
        reasons.extend(_hard_gate_reasons(lot, furnace, at))
    return deny(*reasons, batch_load=load, lot_count=len(lots)) if reasons else allow(
        batch_load=load, lot_count=len(lots)
    )


class _OpenBatch:
    __slots__ = ("lot_ids", "load", "classes")

    def __init__(self) -> None:
        self.lot_ids: list[str] = []
        self.load = 0
        self.classes: set[str] = set()

    def add(self, lot: FurnaceLot) -> None:
        self.lot_ids.append(lot.lot_id)
        self.load += lot.quantity
        self.classes.add(lot.contamination_class)

    def recompute_classes(self, lot_by_id: dict[str, FurnaceLot]) -> None:
        self.classes = {lot_by_id[lot_id].contamination_class for lot_id in self.lot_ids}


def _can_accept(open_batch: _OpenBatch, lot: FurnaceLot, furnace: Furnace, matrix: dict[str, set[str]]) -> bool:
    if open_batch.load + lot.quantity > furnace.max_load:
        return False
    return compatible(open_batch.classes | {lot.contamination_class}, matrix)


def _repack(
    open_batches: list[_OpenBatch],
    furnace: Furnace,
    lot_by_id: dict[str, FurnaceLot],
    matrix: dict[str, set[str]],
) -> list[_OpenBatch]:
    """Move lots out of underfull batches into any batch with room.

    Repeats until no underfull batch can be merged or absorbed, so two
    underfull batches that are compatible and fit together are combined rather
    than rejected separately.
    """

    active = [open_batch for open_batch in open_batches if open_batch.load > 0]
    progressed = True
    while progressed:
        progressed = False
        for source in list(active):
            if source.load >= furnace.min_load:
                continue
            for lot_id in list(source.lot_ids):
                lot = lot_by_id[lot_id]
                target = next(
                    (
                        other
                        for other in active
                        if other is not source and _can_accept(other, lot, furnace, matrix)
                    ),
                    None,
                )
                if target is None:
                    continue
                source.lot_ids.remove(lot_id)
                source.load -= lot.quantity
                source.recompute_classes(lot_by_id)
                target.add(lot)
                progressed = True
        active = [open_batch for open_batch in active if open_batch.load > 0]
    return active


def _finalize(furnace: Furnace, family: str, open_batch: _OpenBatch) -> Batch:
    lot_ids = tuple(sorted(open_batch.lot_ids))
    batch_id = deterministic_id("BATCH", furnace.tool_id, family, *lot_ids)
    return Batch(batch_id, furnace.tool_id, family, lot_ids, open_batch.load)


def form_batches(
    furnace: Furnace,
    lots: list[FurnaceLot],
    at: int,
    contamination_matrix: dict[str, set[str]] | None = None,
) -> BatchingResult:
    """Form deterministic, non-reusable furnace batches from candidate WIP.

    Lots are evaluated in stable ``lot_id`` order regardless of the input
    order.  Hard-gate failures are rejected outright; the remaining lots are
    grouped by recipe family and packed with contamination-aware first-fit.
    Underfull tail batches are then merged or absorbed where possible, and any
    lot that still cannot reach the minimum load is rejected with a reason.
    """

    _validate_lots(lots)
    matrix = contamination_matrix or {}
    ordered = sorted(lots, key=lambda lot: lot.lot_id)

    rejected: dict[str, tuple[str, ...]] = {}
    eligible: list[FurnaceLot] = []
    for lot in ordered:
        reasons = _hard_gate_reasons(lot, furnace, at)
        if reasons:
            rejected[lot.lot_id] = reasons
        else:
            eligible.append(lot)

    by_family: dict[str, list[FurnaceLot]] = {}
    for lot in eligible:
        by_family.setdefault(lot.recipe_family, []).append(lot)

    finalized: list[Batch] = []
    for family in sorted(by_family):
        family_lots = by_family[family]
        open_batches: list[_OpenBatch] = []
        for lot in family_lots:
            target = next(
                (open_batch for open_batch in open_batches if _can_accept(open_batch, lot, furnace, matrix)),
                None,
            )
            if target is None:
                target = _OpenBatch()
                open_batches.append(target)
            target.add(lot)

        lot_by_id = {lot.lot_id: lot for lot in family_lots}
        open_batches = _repack(open_batches, furnace, lot_by_id, matrix)
        for open_batch in open_batches:
            if open_batch.load >= furnace.min_load:
                finalized.append(_finalize(furnace, family, open_batch))
            else:
                for lot_id in open_batch.lot_ids:
                    rejected[lot_id] = ("below_min_load",)

    finalized.sort(key=lambda batch: (batch.recipe_family, batch.lot_ids[0]))
    rejected_tuple = tuple(
        Rejection(lot_id, reasons) for lot_id, reasons in sorted(rejected.items())
    )
    return BatchingResult(furnace.tool_id, tuple(finalized), rejected_tuple)
