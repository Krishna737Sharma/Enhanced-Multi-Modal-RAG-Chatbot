import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# Make OpenCV import optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class ImageProcessor:
    """Advanced image processing for multi-modal RAG"""
    
    def __init__(self):
        self.device = "cpu"  # Default to CPU for compatibility
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            import pytesseract
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            print("pytesseract not available. OCR functionality disabled.")
            return ""
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
    
    def generate_image_description(self, image_path: str) -> str:
        """Generate basic description of image content"""
        try:
            image = Image.open(image_path)
            
            # Basic image analysis
            width, height = image.size
            mode = image.mode
            format_name = image.format or "Unknown"
            
            description = f"Image with dimensions {width}x{height}, color mode: {mode}, format: {format_name}"
            
            # Add basic color analysis
            if mode == "RGB":
                description += ", color image"
            elif mode == "L":
                description += ", grayscale image"
            
            return description
            
        except Exception as e:
            print(f"Image description failed: {e}")
            return "Could not analyze image"
    
    def analyze_image_structure(self, image_path: str) -> Dict[str, Any]:
        """Basic image structure analysis"""
        try:
            image = Image.open(image_path)
            
            if CV2_AVAILABLE:
                # Use OpenCV for advanced analysis
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Detect edges
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                return {
                    "total_shapes": len(contours),
                    "image_complexity": "high" if len(contours) > 10 else "low",
                    "has_edges": True
                }
            else:
                # Basic PIL analysis
                return {
                    "total_shapes": 0,
                    "image_complexity": "unknown",
                    "has_edges": False
                }
                
        except Exception as e:
            print(f"Image structure analysis failed: {e}")
            return {"error": str(e)}
    
    def process_image_document(self, image_path: str, filename: str) -> Document:
        """Process image into a comprehensive document"""
        # Extract text via OCR
        ocr_text = self.extract_text_from_image(image_path)
        
        # Generate description
        description = self.generate_image_description(image_path)
        
        # Analyze structure
        structure = self.analyze_image_structure(image_path)
        
        # Combine all information
        content_parts = [f"Image: {filename}"]
        
        if description:
            content_parts.append(f"Description: {description}")
        
        if ocr_text:
            content_parts.append(f"Extracted Text: {ocr_text}")
        
        if structure and "error" not in structure:
            content_parts.append(f"Structure Analysis: {structure.get('total_shapes', 0)} shapes detected")
            content_parts.append(f"Complexity: {structure.get('image_complexity', 'unknown')}")
        
        content = "\n".join(content_parts)
        
        return Document(
            page_content=content,
            metadata={
                "source": filename,
                "file_type": "image",
                "has_ocr_text": bool(ocr_text),
                "has_description": bool(description),
                "structure_info": structure
            }
        )
