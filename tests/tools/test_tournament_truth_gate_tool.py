import json
from tools import tournament_truth_gate_tool as tool
def test_validator_requires_evidence_but_never_a_turn_contract():
    result=json.loads(tool.run_tournament_truth_gate({"candidate":"draft","request":{},"artifact_metadata":{}},task_id="none",session_id="none"))
    assert result["code"] in {"trusted_runtime_roots_unavailable", "trusted_source_snapshot_required"}
