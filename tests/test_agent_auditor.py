from cartograph.agents.auditor import assign_evals_to_regions, flag_redundancies


def test_auditor_assigns_eval_to_nearest_region(tiny_eval_suite, tiny_regions):
    suspicious = assign_evals_to_regions(tiny_eval_suite, tiny_regions)
    assert suspicious == 0
    assert tiny_eval_suite[0].region_id == "region_00"
    assert tiny_eval_suite[0].assignment_distance == 0.0


def test_auditor_flags_redundancies(tiny_eval_suite):
    duplicate = tiny_eval_suite[0].model_copy(update={"id": "eval_2"})
    redundant = flag_redundancies([tiny_eval_suite[0], duplicate])
    assert redundant[0].case_id == "eval_2"
