from typing import List, Dict, Any, Optional
import time
from .metrics import RAGEvaluator, RAGMetrics

class BatchEvaluator:
    """Batch evaluation system for RAG performance"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.evaluator = RAGEvaluator()
        self.test_cases = []
    
    def add_test_case(self, question: str, expected_answer: Optional[str] = None):
        """Add a test case for evaluation"""
        self.test_cases.append({
            "question": question,
            "expected_answer": expected_answer
        })
    
    def run_evaluation(self, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Run batch evaluation"""
        cases_to_run = test_cases or self.test_cases
        
        if not cases_to_run:
            return {"error": "No test cases available"}
        
        results = []
        
        for i, case in enumerate(cases_to_run):
            print(f"Evaluating case {i+1}/{len(cases_to_run)}: {case['question'][:50]}...")
            
            # Measure response time
            start_time = time.time()
            
            try:
                # Get response from RAG system
                response = self.rag_system.query(case["question"])
                response_time = time.time() - start_time
                
                # Get retrieved contexts
                retrieved_docs = self.rag_system.get_last_retrieved_docs()
                contexts = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
                
                # Evaluate
                metrics = self.evaluator.evaluate_response(
                    question=case["question"],
                    answer=response,
                    retrieved_contexts=contexts,
                    response_time=response_time
                )
                
                results.append({
                    "question": case["question"],
                    "answer": response,
                    "expected_answer": case.get("expected_answer"),
                    "metrics": metrics.to_dict(),
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "question": case["question"],
                    "error": str(e),
                    "success": False
                })
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            avg_metrics = self._calculate_aggregate_metrics(successful_results)
        else:
            avg_metrics = {}
        
        return {
            "results": results,
            "aggregate_metrics": avg_metrics,
            "total_cases": len(cases_to_run),
            "successful_cases": len(successful_results),
            "failure_rate": (len(cases_to_run) - len(successful_results)) / len(cases_to_run)
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics from results"""
        if not results:
            return {}
        
        metrics_keys = ["faithfulness", "answer_relevance", "context_precision", "response_time"]
        aggregates = {}
        
        for key in metrics_keys:
            values = [r["metrics"][key] for r in results if key in r["metrics"]]
            if values:
                aggregates[f"avg_{key}"] = sum(values) / len(values)
                aggregates[f"min_{key}"] = min(values)
                aggregates[f"max_{key}"] = max(values)
        
        return aggregates
