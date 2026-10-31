import numpy as np
import pytest

from pipeline.modeling import TargetSpec
from pipeline.modeling.splits import outer_selections, resolve_selections, select_split


def test_five_outer_selections_preserve_three_on_one_inner_fits(nested_dataset):
    mapping = outer_selections(nested_dataset)
    assert mapping == {f"holdout_{chr(65 + i)}": i for i in range(5)}
    for name, outer_fold in mapping.items():
        split = select_split(nested_dataset, TargetSpec("y", "y"), name)
        assert split.holdout.empty
        assert len(split.inner_cv) == 4
        assert outer_fold not in set(split.train.spatial_cv_fold)
        assert set(split.train.spatial_split_role) == {"training"}
        for fitting, validation in split.inner_cv:
            assert split.train.iloc[fitting].spatial_cv_fold.nunique() == 3
            assert split.train.iloc[validation].spatial_cv_fold.nunique() == 1
            assert not set(split.train.iloc[fitting].NUTS3) & set(split.train.iloc[validation].NUTS3)


@pytest.mark.parametrize("stage,selection", [("develop", "final_test"),
    ("outer_evaluate", "final_test"), ("final_evaluate", "holdout_A"),
    ("typo", "holdout_A"), ("develop", "holdout_F")])
def test_invalid_stage_selection_combinations_fail(nested_dataset, stage, selection):
    with pytest.raises(ValueError):
        resolve_selections(nested_dataset, selection, stage=stage)


def test_missing_target_does_not_reassign_folds_or_read_test_outcomes(nested_dataset):
    nested_dataset.frame.loc[nested_dataset.frame.spatial_split_role.eq("fixed_test"), "other"] = np.nan
    split = select_split(nested_dataset, TargetSpec("other", "other"), "holdout_A")
    assert set(split.train.spatial_cv_fold) == {1, 2, 3, 4}
    assert split.train.other.notna().all()
    with pytest.raises(ValueError, match="no eligible holdout"):
        select_split(nested_dataset, TargetSpec("other", "other"), "final_test", stage="final_evaluate")
