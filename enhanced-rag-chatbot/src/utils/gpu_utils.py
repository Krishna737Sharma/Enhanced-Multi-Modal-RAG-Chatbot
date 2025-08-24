import torch
import logging
from typing import Dict, Any, Optional

class GPUManager:
    """GPU management utilities"""
    
    def __init__(self):
        self.device = self._get_best_device()
        self.gpu_info = self._get_gpu_info()
    
    def _get_best_device(self) -> str:
        """Get the best available device"""
        if torch.cuda.is_available():
            # Select GPU with most free memory
            best_gpu = 0
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                if gpu_memory > max_memory:
                    max_memory = gpu_memory
                    best_gpu = i
            
            device = f"cuda:{best_gpu}"
            logging.info(f"Using GPU device: {device}")
            return device
        else:
            logging.info("CUDA not available, using CPU")
            return "cpu"
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        info = {
            "available": True,
            "count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "name": device_props.name,
                "total_memory": device_props.total_memory / (1024**3),  # GB
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            }
            info["devices"].append(device_info)
        
        return info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {}
        
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        cached = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "free_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) - cached
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU cache cleared")
    
    def optimize_for_inference(self):
        """Optimize PyTorch for inference"""
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        logging.info("Optimized for inference")
