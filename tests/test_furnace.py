from __future__ import annotations

import pytest

from fabweave.contamination.segregation import incompatible_pairs
from fabweave.core.timebox import TimeBox
from fabweave.furnace.batching import (
    Batch,
    BatchingResult,
    Furnace,
    FurnaceLot,
    batch_decision,
    form_batches,
)


FURNACE = Furnace(
    tool_id="FUR-1",
    min_load=4,
    max_load=10,
    qualified_families=frozenset({"OXIDE", "NITRIDE"}),
    allowed_contamination=frozenset({"CLEAN", "CU", "FE"}),
)
MATRIX = {"CU": {"FE"}}


def lot(lot_id, family="OXIDE", contamination="CLEAN", quantity=4, **kwargs):
    return FurnaceLot(lot_id, family, contamination, quantity, **kwargs)


def test_basic_batches_meet_load_range_and_are_content_derived():
    result = form_batches(FURNACE, [lot("LOT-A", quantity=4), lot("LOT-B", quantity=4)], 0, MATRIX)
    assert result.placed_lot_count == 2
    assert result.rejected_lot_count == 0
    assert all(FURNACE.min_load <= batch.load <= FURNACE.max_load for batch in result.batches)
    batch = result.batches[0]
    assert batch.lot_ids == ("LOT-A", "LOT-B")
    assert batch.load == 8
    assert batch.batch_id.startswith("BATCH-")
    # same content -> same id, regardless of input order
    reversed_result = form_batches(FURNACE, [lot("LOT-B", quantity=4), lot("LOT-A", quantity=4)], 0, MATRIX)
    assert reversed_result.batches[0].batch_id == batch.batch_id


def test_result_is_order_independent():
    lots = [
        lot("LOT-C", quantity=4),
        lot("LOT-A", quantity=3),
        lot("LOT-B", quantity=3),
        lot("LOT-D", family="NITRIDE", quantity=5),
        lot("LOT-E", family="NITRIDE", quantity=5),
    ]
    forward = form_batches(FURNACE, lots, 0, MATRIX)
    backward = form_batches(FURNACE, list(reversed(lots)), 0, MATRIX)
    assert [(b.recipe_family, b.lot_ids, b.load) for b in forward.batches] == [
        (b.recipe_family, b.lot_ids, b.load) for b in backward.batches
    ]


def test_lots_are_never_reused_and_disposition_accounts_for_every_lot():
    lots = [lot("LOT-A", quantity=4), lot("LOT-B", quantity=4), lot("LOT-C", quantity=4)]
    result = form_batches(FURNACE, lots, 0, MATRIX)
    placed = [lot_id for batch in result.batches for lot_id in batch.lot_ids]
    assert len(placed) == len(set(placed))
    assert result.accounts_for({"LOT-A", "LOT-B", "LOT-C"})
    disposition = result.disposition()
    assert set(disposition) == {"LOT-A", "LOT-B", "LOT-C"}
    assert all(value.startswith("batch:") for value in disposition.values())


def test_hard_gate_failures_are_rejected_with_reasons():
    lots = [
        lot("LOT-OK", quantity=4),
        lot("LOT-HELD", quantity=4, active_holds=frozenset({"QA"})),
        lot("LOT-EARLY", quantity=4, window=TimeBox(10, 20)),
        lot("LOT-DUE", quantity=4, window=TimeBox(-10, 0)),
        lot("LOT-POLY", family="POLY", quantity=4),
        lot("LOT-NA", contamination="NA", quantity=4),
        lot("LOT-BIG", quantity=11),
    ]
    result = form_batches(FURNACE, lots, 0, MATRIX)
    reasons = {rejection.lot_id: rejection.reasons for rejection in result.rejected}
    assert "LOT-OK" not in reasons
    assert reasons["LOT-HELD"] == ("lot_held",)
    assert reasons["LOT-EARLY"] == ("outside_due_window",)
    assert reasons["LOT-DUE"] == ("outside_due_window",)
    assert reasons["LOT-POLY"] == ("recipe_not_qualified",)
    assert reasons["LOT-NA"] == ("contamination_not_allowed",)
    assert reasons["LOT-BIG"] == ("load_exceeds_max",)
    assert result.accounts_for({l.lot_id for l in lots})


def test_incompatible_contamination_is_split_into_separate_batches():
    lots = [
        lot("LOT-CU", contamination="CU", quantity=4),
        lot("LOT-FE", contamination="FE", quantity=4),
    ]
    result = form_batches(FURNACE, lots, 0, MATRIX)
    assert len(result.batches) == 2
    by_class = {batch.lot_ids[0]: batch for batch in result.batches}
    assert set(by_class) == {"LOT-CU", "LOT-FE"}
    for batch in result.batches:
        classes = {lot_contamination(l.lot_id) for l in lots if l.lot_id in batch.lot_ids}
        assert incompatible_pairs(classes, MATRIX) == ()


def lot_contamination(lot_id):
    return {"LOT-CU": "CU", "LOT-FE": "FE"}[lot_id]


def test_lots_that_cannot_reach_min_load_are_rejected():
    lots = [lot("LOT-A", quantity=3)]
    result = form_batches(FURNACE, lots, 0, MATRIX)
    assert result.placed_lot_count == 0
    assert result.rejected_lot_count == 1
    assert result.rejected[0].reasons == ("below_min_load",)
    assert result.disposition()["LOT-A"] == "rejected:below_min_load"


def test_underfull_tail_is_absorbed_when_there_is_room():
    # max=8: 3+3=6 fits, then 3 more would be 9 > 8 -> tail of 3 below min.
    furnace = Furnace(
        tool_id="FUR-2",
        min_load=4,
        max_load=8,
        qualified_families=frozenset({"OXIDE"}),
        allowed_contamination=frozenset({"CLEAN"}),
    )
    lots = [lot("LOT-A", quantity=3), lot("LOT-B", quantity=3), lot("LOT-C", quantity=3)]
    result = form_batches(furnace, lots, 0, {})
    # A+B fill 6; C cannot join (9 > 8) and cannot form its own min batch.
    assert result.placed_lot_count == 2
    assert result.rejected[0].lot_id == "LOT-C"
    assert result.rejected[0].reasons == ("below_min_load",)


def test_batch_decision_validates_one_batch():
    good = batch_decision(FURNACE, [lot("LOT-A", quantity=4), lot("LOT-B", quantity=4)], 0, MATRIX)
    assert good.allowed
    bad_family = batch_decision(
        FURNACE, [lot("LOT-A", quantity=4), lot("LOT-B", family="NITRIDE", quantity=4)], 0, MATRIX
    )
    assert bad_family.reasons == ("mixed_recipe_family",)
    bad_load = batch_decision(FURNACE, [lot("LOT-A", quantity=2)], 0, MATRIX)
    assert "load_below_min" in bad_load.reasons
    bad_contamination = batch_decision(
        FURNACE, [lot("LOT-A", contamination="CU", quantity=4), lot("LOT-B", contamination="FE", quantity=4)],
        0,
        MATRIX,
    )
    assert "incompatible_contamination:CU:FE" in bad_contamination.reasons


def test_duplicate_lot_is_rejected():
    with pytest.raises(ValueError):
        form_batches(FURNACE, [lot("LOT-A", quantity=4), lot("LOT-A", quantity=4)], 0, MATRIX)


def test_invalid_furnace_load_range():
    with pytest.raises(ValueError):
        Furnace("FUR-X", min_load=0, max_load=10)
    with pytest.raises(ValueError):
        Furnace("FUR-X", min_load=10, max_load=4)


def test_empty_input():
    result = form_batches(FURNACE, [], 0, MATRIX)
    assert result.batches == () and result.rejected == ()
    assert result.accounts_for(set())
