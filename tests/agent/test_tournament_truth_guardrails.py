from agent.tool_guardrails import ToolCallGuardrailController


class Contract:
    def __init__(self, valid=False): self.valid = valid
    def has_valid_receipt(self): return self.valid


def test_tournament_turn_blocks_mutations_delivery_and_broad_discovery_until_receipt():
    controller = ToolCallGuardrailController()
    controller.set_tournament_contract(Contract())
    for name in ("write_file", "patch", "terminal", "execute_code", "image_generation", "delegate_task", "web_search", "send_message"):
        assert controller.preflight_request_contract(name, {}).code == "tournament_receipt_required"
    assert controller.preflight_request_contract("read_file", {}).action == "allow"
    assert controller.preflight_request_contract("tournament_truth_gate", {}).action == "allow"


def test_valid_receipt_releases_existing_tool_policy_and_clear_restores_normal_turns():
    controller = ToolCallGuardrailController()
    contract = Contract(valid=True)
    controller.set_tournament_contract(contract)
    assert controller.before_call("terminal", {}).action == "allow"
    controller.set_tournament_contract(None)
    assert controller.before_call("web_search", {}).action == "allow"
