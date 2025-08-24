from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass

@dataclass
class RAGMetrics:
    """Data class for RAG evaluation metrics"""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    response_time: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "response_time": self.response_time
        }

class RAGEvaluator:
    """Simple RAG evaluation system"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_response(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        response_time: float = 0.0
    ) -> RAGMetrics:
        """Basic evaluation of RAG response"""
        # Simple heuristic-based evaluation
        faithfulness = self._calculate_simple_faithfulness(answer, retrieved_contexts)
        answer_relevance = self._calculate_simple_relevance(question, answer)
        context_precision = self._calculate_simple_precision(question, retrieved_contexts)
        
        metrics = RAGMetrics(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            response_time=response_time
        )
        
        # Store in history
        self.evaluation_history.append({
            "question": question,
            "answer": answer,
            "metrics": metrics,
            "timestamp": time.time()
        })
        
        return metrics
    
    def _calculate_simple_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Simple faithfulness calculation based on word overlap"""
        if not contexts or not answer:
            return 0.0
        
        answer_words = set(answer.lower().split())
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        return min(overlap / len(answer_words), 1.0)
    
    def _calculate_simple_relevance(self, question: str, answer: str) -> float:
        """Simple relevance calculation based on word overlap"""
        if not question or not answer:
            return 0.0
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(answer_words))
        return min(overlap / len(question_words), 1.0)
    
    def _calculate_simple_precision(self, question: str, contexts: List[str]) -> float:
        """Simple precision calculation"""
        if not contexts or not question:
            return 0.0
        
        question_words = set(question.lower().split())
        relevant_contexts = 0
        
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(question_words.intersection(context_words))
            if overlap > 0:
                relevant_contexts += 1
        
        return relevant_contexts / len(contexts) if contexts else 0.0
    
    def get_average_metrics(self, recent_n: Optional[int] = None) -> Dict[str, float]:
        """Get average metrics from evaluation history"""
        if not self.evaluation_history:
            return {}
        
        history = self.evaluation_history[-recent_n:] if recent_n else self.evaluation_history
        
        metrics_sum = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "response_time": 0.0
        }
        
        for entry in history:
            metrics_dict = entry["metrics"].to_dict()
            for key in metrics_sum:
                metrics_sum[key] += metrics_dict[key]
        
        count = len(history)
        return {key: value / count for key, value in metrics_sum.items()}
