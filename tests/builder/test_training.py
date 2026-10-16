from sopvision.builder import training


def test_versions_are_numbered_per_sop_and_listed_with_their_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MODELS", tmp_path)
    assert training.next_version("wheel") == 1
    for name in ("wheel-v1", "wheel-v2", "wheel_big-v7"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.safetensors").write_bytes(b"")
    (tmp_path / "wheel-v2.eval").mkdir()
    (tmp_path / "wheel-v2.eval" / "report.json").write_text('{"bench": {"current_step_accuracy": 0.9}}')
    assert training.next_version("wheel") == 3
    assert [(v["checkpoint"], v["report"]) for v in training.versions("wheel")] == [
        ("wheel-v2", {"bench": {"current_step_accuracy": 0.9}}), ("wheel-v1", None)]
    p = training.paths("wheel", 3)
    assert (p.model, str(p.benchmark), str(p.training_set)) == (tmp_path / "wheel-v3", "benchmarks/wheel-v3-test",
                                                               "data/training/wheel-v3")
