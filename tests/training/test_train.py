from training.train import summarize_metrics


def test_summarize_metrics_reads_ultralytics_keys():
    d = {"metrics/precision(B)": 0.9, "metrics/mAP50(B)": 0.91234, "metrics/mAP50-95(B)": 0.71234, "fitness": 0.7}
    assert summarize_metrics(d) == {"map50": 0.9123, "map50_95": 0.7123}
