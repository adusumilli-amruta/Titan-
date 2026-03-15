import torch
import unittest
import os
import tempfile
from titan.models.modeling_titan import TitanForCausalLM, TitanConfig
from titan.serving.inference_engine import InferenceEngine, GenerationRequest
from titan.monitoring.db_tracker import ExperimentDatabase, MetricsCollector


class TestInferenceEngine(unittest.TestCase):
    """Tests the serving layer's inference engine on CPU with a tiny model."""

    def setUp(self):
        self.config = TitanConfig(
            vocab_size=1000, hidden_size=128, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=512, sliding_window_size=256,
        )
        self.engine = InferenceEngine(model_path="mock", device="cpu", dtype="float32")
        self.engine.load_model(config_overrides={
            "vocab_size": 1000, "hidden_size": 128, "intermediate_size": 512,
            "num_hidden_layers": 2, "num_attention_heads": 4,
        })

    def test_engine_loads(self):
        self.assertTrue(self.engine.is_loaded)

    def test_greedy_generation(self):
        request = GenerationRequest(
            request_id="test-001",
            input_ids=torch.randint(0, 1000, (16,)),
            max_new_tokens=5,
            temperature=0.0,
            do_sample=False,
        )
        response = self.engine.generate(request)
        self.assertEqual(response.request_id, "test-001")
        self.assertGreater(response.generated_tokens, 0)
        self.assertIn(response.finish_reason, ["stop", "length"])
        self.assertGreater(response.latency_ms, 0)

    def test_sampled_generation(self):
        request = GenerationRequest(
            request_id="test-002",
            input_ids=torch.randint(0, 1000, (8,)),
            max_new_tokens=3,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            do_sample=True,
        )
        response = self.engine.generate(request)
        self.assertEqual(response.request_id, "test-002")
        self.assertGreater(response.generated_tokens, 0)

    def test_stats(self):
        stats = self.engine.get_stats()
        self.assertTrue(stats["model_loaded"])
        self.assertEqual(stats["total_requests_served"], 0)


class TestExperimentDatabase(unittest.TestCase):
    """Tests the SQLite experiment tracking database."""

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".db")
        self.db = ExperimentDatabase(db_path=self.tmp)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_create_experiment(self):
        exp_id = self.db.create_experiment("exp-001", "test_run", "pretrain", {"lr": 3e-4})
        self.assertEqual(exp_id, "exp-001")
        exp = self.db.get_experiment("exp-001")
        self.assertIsNotNone(exp)
        self.assertEqual(exp["name"], "test_run")

    def test_log_and_retrieve_metrics(self):
        self.db.create_experiment("exp-002", "metric_test", "rlhf")
        self.db.log_metrics("exp-002", step=10, metrics={"loss": 2.5, "accuracy": 0.6})
        self.db.log_metrics("exp-002", step=20, metrics={"loss": 1.8, "accuracy": 0.75})

        history = self.db.get_metric_history("exp-002", "loss")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["step"], 10)
        self.assertAlmostEqual(history[0]["value"], 2.5)

    def test_log_checkpoint(self):
        self.db.create_experiment("exp-003", "ckpt_test", "pretrain")
        self.db.log_checkpoint("exp-003", step=5000, path="/ckpts/step5000", size_mb=13200.0)
        ckpts = self.db.get_checkpoints("exp-003")
        self.assertEqual(len(ckpts), 1)
        self.assertEqual(ckpts[0]["step"], 5000)

    def test_log_evaluation(self):
        self.db.create_experiment("exp-004", "eval_test", "rlhf")
        self.db.log_evaluation("exp-004", checkpoint_step=5000, benchmark="gsm8k", score=0.63)
        evals = self.db.get_evaluations("exp-004")
        self.assertEqual(len(evals), 1)
        self.assertAlmostEqual(evals[0]["score"], 0.63)

    def test_list_experiments(self):
        self.db.create_experiment("e1", "run1", "pretrain")
        self.db.create_experiment("e2", "run2", "rlhf")
        all_exps = self.db.list_experiments()
        self.assertEqual(len(all_exps), 2)
        pretrain_only = self.db.list_experiments(stage="pretrain")
        self.assertEqual(len(pretrain_only), 1)


class TestMetricsCollector(unittest.TestCase):
    """Tests the Prometheus-style metrics collector."""

    def test_record_and_export(self):
        collector = MetricsCollector()
        collector.record_request(latency_ms=150.0, tokens_generated=50, success=True)
        collector.record_request(latency_ms=300.0, tokens_generated=100, success=True)
        collector.record_request(latency_ms=5000.0, tokens_generated=0, success=False)

        metrics = collector.get_metrics()
        self.assertEqual(metrics["total_requests"], 3)
        self.assertEqual(metrics["total_errors"], 1)
        self.assertEqual(metrics["total_tokens_generated"], 150)

    def test_prometheus_format(self):
        collector = MetricsCollector()
        collector.record_request(latency_ms=100.0, tokens_generated=50)
        output = collector.to_prometheus_format()
        self.assertIn("titan_requests_total 1", output)
        self.assertIn("titan_tokens_per_second", output)


if __name__ == "__main__":
    unittest.main()
