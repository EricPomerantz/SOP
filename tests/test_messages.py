from sopvision.messages import AlertMsg, DecisionMsg, RunSummaryMsg, SceneMsg, StepUpdateMsg, to_ws


def test_messages_carry_their_type():
    assert to_ws(SceneMsg(t=1.0, scene={"version": 1}))["type"] == "scene"
    assert to_ws(StepUpdateMsg(t=1.0, step_id="fan_1", status="done", source="jev", duration=6.4)) == {
        "type": "step_update", "t": 1.0, "step_id": "fan_1", "status": "done", "source": "jev", "duration": 6.4}
    assert to_ws(AlertMsg(t=2.0, level="violation", kind="order_violation", message="m", step_id="fan_2"))["type"] == "alert"
    assert to_ws(DecisionMsg(t=1.0, current_step="fan_1", confidence=0.9, active_step_id="fan_1", active_step_done=0.1,
                             anomaly=0.0, latency_ms=150.0))["type"] == "decision"
    assert to_ws(RunSummaryMsg(t=3.0, report={}))["type"] == "run_summary"
