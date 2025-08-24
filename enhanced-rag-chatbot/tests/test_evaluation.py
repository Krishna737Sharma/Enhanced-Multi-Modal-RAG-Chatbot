import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import RAGEvaluator, RAGMetrics

class TestRAGEvaluator:
    """Test suite for RAGEvaluator"""
    
    def setup_method(self):
        self.evaluator = RAGEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        assert self.evaluator is not None
        assert self.evaluator.evaluation_history == []
    
    def test_evaluate_response(self):
        """Test response evaluation"""
        question = "What is the capital of France?"
        answer = "Paris is the capital of France"
        contexts = ["Paris is the capital and most populous city of France."]
        
        metrics = self.evaluator.evaluate_response(question, answer, contexts)
        
        assert isinstance(metrics, RAGMetrics)
        assert 0 <= metrics.faithfulness <= 1
        assert 0 <= metrics.answer_relevance <= 1
        assert 0 <= metrics.context_precision <= 1
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary"""
        metrics = RAGMetrics(
            faithfulness=0.8,
            answer_relevance=0.9,
            context_precision=0.7,
            response_time=1.5
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert "faithfulness" in metrics_dict
        assert "answer_relevance" in metrics_dict
        assert "context_precision" in metrics_dict
        assert "response_time" in metrics_dict

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
