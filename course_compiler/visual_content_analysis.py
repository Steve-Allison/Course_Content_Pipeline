"""
Visual Content Analysis Module

Extracts comprehensive knowledge from images, charts, diagrams, and visual elements
using OCR, computer vision, and AI models for downstream AI reuse.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import base64
import io

# Computer vision and OCR
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# AI vision models
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
    VISION_AI_AVAILABLE = True
except ImportError:
    VISION_AI_AVAILABLE = False

from .logging_utils import get_logger

logger = get_logger(__name__)

class VisualContentAnalyzer:
    """Comprehensive visual content analysis for educational materials."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = self.config.get('output_dir', 'output/visual_analysis')

        # Initialize AI models
        self.image_captioning_model = None
        self.chart_analysis_pipeline = None
        self._load_ai_models()

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_ai_models(self):
        """Load AI models for visual analysis."""
        if VISION_AI_AVAILABLE:
            try:
                # Image captioning model
                self.image_captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.image_captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

                # Object detection pipeline
                self.object_detection = pipeline("object-detection", model="facebook/detr-resnet-50")

                logger.info("AI vision models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load AI vision models: {e}")

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single image.

        Returns:
            Dictionary containing all extracted visual knowledge
        """
        image_path = Path(image_path)
        logger.info(f"Analyzing image: {image_path.name}")

        results = {
            'file_path': str(image_path),
            'file_name': image_path.name,
            'available_analyzers': {
                'ocr': OCR_AVAILABLE,
                'cv2': CV2_AVAILABLE,
                'vision_ai': VISION_AI_AVAILABLE,
                'plotting': PLOTTING_AVAILABLE
            }
        }

        try:
            # 1. Basic image properties
            results['basic_properties'] = self._extract_basic_properties(image_path)

            # 2. OCR text extraction
            if OCR_AVAILABLE:
                results['ocr_analysis'] = self._extract_text_from_image(image_path)

            # 3. AI-powered image captioning and description
            if VISION_AI_AVAILABLE and self.image_captioning_model:
                results['ai_description'] = self._generate_ai_description(image_path)

            # 4. Object and element detection
            if VISION_AI_AVAILABLE and self.object_detection:
                results['object_detection'] = self._detect_objects(image_path)

            # 5. Chart and diagram analysis
            results['chart_analysis'] = self._analyze_charts_diagrams(image_path)

            # 6. Visual structure analysis
            if CV2_AVAILABLE:
                results['structure_analysis'] = self._analyze_visual_structure(image_path)

            # 7. Educational content classification
            results['educational_classification'] = self._classify_educational_content(image_path, results)

            # 8. Extract actionable knowledge
            results['knowledge_extraction'] = self._extract_actionable_knowledge(results)

        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}", exc_info=True)
            results['error'] = str(e)

        return results

    def _extract_basic_properties(self, image_path: Path) -> Dict[str, Any]:
        """Extract basic image properties."""
        try:
            with Image.open(image_path) as img:
                return {
                    'dimensions': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'file_size_bytes': image_path.stat().st_size,
                    'aspect_ratio': img.size[0] / img.size[1] if img.size[1] > 0 else 0
                }
        except Exception as e:
            logger.warning(f"Failed to extract basic properties: {e}")
            return {'error': str(e)}

    def _extract_text_from_image(self, image_path: Path) -> Dict[str, Any]:
        """Extract text using OCR with preprocessing."""
        try:
            # Load and preprocess image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Enhance image for better OCR
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)

                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(2.0)

                # Extract text with confidence scores
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

                # Filter out low-confidence text
                text_elements = []
                full_text = []

                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    confidence = int(ocr_data['conf'][i])

                    if text and confidence > 30:  # Filter low-confidence text
                        text_elements.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': (
                                ocr_data['left'][i],
                                ocr_data['top'][i],
                                ocr_data['width'][i],
                                ocr_data['height'][i]
                            ),
                            'level': ocr_data['level'][i]
                        })
                        full_text.append(text)

                return {
                    'full_text': ' '.join(full_text),
                    'text_elements': text_elements,
                    'total_elements': len(text_elements),
                    'avg_confidence': np.mean([elem['confidence'] for elem in text_elements]) if text_elements else 0,
                    'languages_detected': pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
                }

        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return {'error': str(e)}

    def _generate_ai_description(self, image_path: Path) -> Dict[str, Any]:
        """Generate AI-powered image description and analysis."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Generate caption
                inputs = self.image_captioning_processor(img, return_tensors="pt")
                out = self.image_captioning_model.generate(**inputs, max_length=100, num_beams=5)
                caption = self.image_captioning_processor.decode(out[0], skip_special_tokens=True)

                # Generate conditional descriptions for educational content
                educational_prompts = [
                    "This image shows a diagram of",
                    "This image contains a chart showing",
                    "This educational image illustrates",
                    "This figure demonstrates"
                ]

                conditional_descriptions = {}
                for prompt in educational_prompts:
                    inputs = self.image_captioning_processor(img, prompt, return_tensors="pt")
                    out = self.image_captioning_model.generate(**inputs, max_length=100)
                    desc = self.image_captioning_processor.decode(out[0], skip_special_tokens=True)
                    conditional_descriptions[prompt] = desc

                return {
                    'general_caption': caption,
                    'educational_descriptions': conditional_descriptions,
                    'content_type_hints': self._classify_image_type(caption, conditional_descriptions)
                }

        except Exception as e:
            logger.warning(f"AI description generation failed: {e}")
            return {'error': str(e)}

    def _detect_objects(self, image_path: Path) -> Dict[str, Any]:
        """Detect objects and elements in the image."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Object detection
                objects = self.object_detection(img)

                # Process and filter results
                detected_objects = []
                for obj in objects:
                    if obj['score'] > 0.3:  # Filter low-confidence detections
                        detected_objects.append({
                            'label': obj['label'],
                            'confidence': obj['score'],
                            'bbox': obj['box']
                        })

                # Categorize objects for educational content
                educational_objects = self._categorize_educational_objects(detected_objects)

                return {
                    'detected_objects': detected_objects,
                    'object_count': len(detected_objects),
                    'educational_categories': educational_objects,
                    'high_confidence_objects': [obj for obj in detected_objects if obj['confidence'] > 0.7]
                }

        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return {'error': str(e)}

    def _analyze_charts_diagrams(self, image_path: Path) -> Dict[str, Any]:
        """Analyze charts, graphs, and diagrams."""
        try:
            analysis = {
                'chart_type': 'unknown',
                'has_axes': False,
                'has_legend': False,
                'has_title': False,
                'data_points_estimated': 0,
                'color_scheme': [],
                'educational_value': {}
            }

            if CV2_AVAILABLE:
                # Load image with OpenCV
                img = cv2.imread(str(image_path))
                if img is None:
                    return {'error': 'Could not load image with OpenCV'}

                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Detect chart elements
                analysis['chart_elements'] = self._detect_chart_elements(img, gray)

                # Analyze color distribution (for charts/graphs)
                analysis['color_analysis'] = self._analyze_color_distribution(img, hsv)

                # Detect text regions (titles, labels, legends)
                analysis['text_regions'] = self._detect_text_regions(gray)

                # Estimate chart type based on visual features
                analysis['chart_type'] = self._estimate_chart_type(analysis)

            return analysis

        except Exception as e:
            logger.warning(f"Chart analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_visual_structure(self, image_path: Path) -> Dict[str, Any]:
        """Analyze the visual structure and layout."""
        try:
            img = cv2.imread(str(image_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

            # Contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze layout
            layout_analysis = {
                'edge_density': np.sum(edges > 0) / edges.size,
                'line_count': len(lines) if lines is not None else 0,
                'contour_count': len(contours),
                'has_grid_structure': self._detect_grid_structure(lines),
                'layout_type': self._classify_layout_type(edges, lines, contours),
                'visual_complexity': self._calculate_visual_complexity(edges, contours)
            }

            return layout_analysis

        except Exception as e:
            logger.warning(f"Visual structure analysis failed: {e}")
            return {'error': str(e)}

    def _classify_educational_content(self, image_path: Path, analysis_results: Dict) -> Dict[str, Any]:
        """Classify the educational content type and value."""
        classification = {
            'content_category': 'unknown',
            'educational_value': 'medium',
            'instructional_type': [],
            'complexity_level': 'intermediate',
            'learning_objectives': []
        }

        try:
            # Analyze text content for educational keywords
            text_content = analysis_results.get('ocr_analysis', {}).get('full_text', '')
            ai_description = analysis_results.get('ai_description', {}).get('general_caption', '')

            combined_text = f"{text_content} {ai_description}".lower()

            # Educational content patterns
            if any(word in combined_text for word in ['diagram', 'flowchart', 'process', 'workflow']):
                classification['content_category'] = 'process_diagram'
                classification['instructional_type'].append('conceptual_framework')

            if any(word in combined_text for word in ['chart', 'graph', 'data', 'statistics']):
                classification['content_category'] = 'data_visualization'
                classification['instructional_type'].append('data_analysis')

            if any(word in combined_text for word in ['screenshot', 'interface', 'menu', 'button']):
                classification['content_category'] = 'interface_guide'
                classification['instructional_type'].append('procedural_knowledge')

            if any(word in combined_text for word in ['map', 'location', 'geography']):
                classification['content_category'] = 'spatial_information'
                classification['instructional_type'].append('spatial_knowledge')

            # Complexity assessment
            word_count = len(text_content.split())
            if word_count > 100:
                classification['complexity_level'] = 'advanced'
            elif word_count < 20:
                classification['complexity_level'] = 'basic'

            return classification

        except Exception as e:
            logger.warning(f"Educational classification failed: {e}")
            return classification

    def _extract_actionable_knowledge(self, analysis_results: Dict) -> Dict[str, Any]:
        """Extract actionable knowledge for AI downstream processing."""
        knowledge = {
            'key_concepts': [],
            'learning_points': [],
            'actionable_items': [],
            'ai_prompt_suggestions': [],
            'knowledge_graph_nodes': []
        }

        try:
            # Extract from OCR text
            ocr_text = analysis_results.get('ocr_analysis', {}).get('full_text', '')
            if ocr_text:
                # Extract key concepts using simple NLP
                import re
                sentences = re.split(r'[.!?]+', ocr_text)
                for sentence in sentences[:5]:  # Top 5 sentences
                    if len(sentence.strip()) > 10:
                        knowledge['learning_points'].append(sentence.strip())

                # Extract potential concepts (capitalized words/phrases)
                concepts = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', ocr_text)
                knowledge['key_concepts'] = list(set(concepts[:10]))  # Top 10 unique concepts

            # Extract from AI description
            ai_desc = analysis_results.get('ai_description', {}).get('general_caption', '')
            if ai_desc:
                knowledge['ai_prompt_suggestions'].append(f"Analyze this {ai_desc} for educational insights")

            # Generate knowledge graph nodes
            classification = analysis_results.get('educational_classification', {})
            content_category = classification.get('content_category', 'unknown')

            if content_category != 'unknown':
                knowledge['knowledge_graph_nodes'].append({
                    'type': 'visual_content',
                    'category': content_category,
                    'concepts': knowledge['key_concepts'],
                    'source': analysis_results.get('file_name', ''),
                    'confidence': 0.8
                })

            # Generate actionable items
            if knowledge['key_concepts']:
                knowledge['actionable_items'].append(f"Define and explain: {', '.join(knowledge['key_concepts'][:3])}")

            if classification.get('instructional_type'):
                inst_type = classification['instructional_type'][0]
                knowledge['actionable_items'].append(f"Create {inst_type} learning activity based on this visual")

            return knowledge

        except Exception as e:
            logger.warning(f"Knowledge extraction failed: {e}")
            return knowledge

    # Helper methods
    def _classify_image_type(self, caption: str, descriptions: Dict) -> List[str]:
        """Classify image type based on AI descriptions."""
        types = []
        combined_text = f"{caption} {' '.join(descriptions.values())}".lower()

        if any(word in combined_text for word in ['chart', 'graph', 'plot']):
            types.append('chart')
        if any(word in combined_text for word in ['diagram', 'flowchart', 'schema']):
            types.append('diagram')
        if any(word in combined_text for word in ['screenshot', 'interface', 'screen']):
            types.append('screenshot')
        if any(word in combined_text for word in ['photo', 'picture', 'image']):
            types.append('photograph')

        return types if types else ['unknown']

    def _categorize_educational_objects(self, objects: List[Dict]) -> Dict[str, List]:
        """Categorize detected objects by educational relevance."""
        categories = {
            'text_elements': [],
            'visual_aids': [],
            'interactive_elements': [],
            'data_representations': []
        }

        for obj in objects:
            label = obj['label'].lower()
            if any(word in label for word in ['text', 'book', 'paper', 'document']):
                categories['text_elements'].append(obj)
            elif any(word in label for word in ['chart', 'graph', 'table']):
                categories['data_representations'].append(obj)
            elif any(word in label for word in ['button', 'menu', 'icon']):
                categories['interactive_elements'].append(obj)
            else:
                categories['visual_aids'].append(obj)

        return categories

    def _detect_chart_elements(self, img, gray):
        """Detect common chart elements."""
        # This is a simplified version - could be enhanced with more sophisticated detection
        elements = {
            'has_axes': False,
            'has_grid': False,
            'has_bars': False,
            'has_lines': False,
            'has_circles': False
        }

        # Detect lines (potential axes or grid)
        lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=50)
        if lines is not None and len(lines) > 10:
            elements['has_axes'] = True
            if len(lines) > 20:
                elements['has_grid'] = True

        # Detect rectangles (potential bars)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_count = sum(1 for contour in contours if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4)
        if rect_count > 5:
            elements['has_bars'] = True

        # Detect circles (potential pie charts or data points)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            elements['has_circles'] = True

        return elements

    def _analyze_color_distribution(self, img, hsv):
        """Analyze color distribution in the image."""
        # Calculate color histogram
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        return {
            'dominant_hues': np.argsort(hist_h.flatten())[-5:].tolist(),
            'color_diversity': float(np.std(hist_h)),
            'brightness_avg': float(np.mean(hist_v)),
            'saturation_avg': float(np.mean(hist_s))
        }

    def _detect_text_regions(self, gray):
        """Detect potential text regions."""
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        connected = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:  # Minimum size for text regions
                text_regions.append({'x': x, 'y': y, 'width': w, 'height': h})

        return text_regions

    def _estimate_chart_type(self, analysis):
        """Estimate chart type based on visual analysis."""
        elements = analysis.get('chart_elements', {})

        if elements.get('has_circles') and not elements.get('has_axes'):
            return 'pie_chart'
        elif elements.get('has_bars') and elements.get('has_axes'):
            return 'bar_chart'
        elif elements.get('has_lines') and elements.get('has_axes'):
            return 'line_chart'
        elif elements.get('has_grid') and elements.get('has_axes'):
            return 'scatter_plot'
        else:
            return 'diagram'

    def _detect_grid_structure(self, lines):
        """Detect if the image has a grid structure."""
        if lines is None or len(lines) < 4:
            return False

        # Simplified grid detection - count horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 10 or abs(angle) > 170:  # Horizontal
                horizontal_lines += 1
            elif 80 < abs(angle) < 100:  # Vertical
                vertical_lines += 1

        return horizontal_lines >= 2 and vertical_lines >= 2

    def _classify_layout_type(self, edges, lines, contours):
        """Classify the overall layout type."""
        edge_density = np.sum(edges > 0) / edges.size
        line_count = len(lines) if lines is not None else 0
        contour_count = len(contours)

        if edge_density > 0.1 and line_count > 20:
            return 'structured'
        elif contour_count > 10:
            return 'complex'
        elif edge_density < 0.05:
            return 'simple'
        else:
            return 'moderate'

    def _calculate_visual_complexity(self, edges, contours):
        """Calculate a visual complexity score."""
        edge_density = np.sum(edges > 0) / edges.size
        contour_complexity = sum(len(contour) for contour in contours) / len(contours) if contours else 0

        complexity_score = (edge_density * 0.7) + (min(contour_complexity / 100, 1.0) * 0.3)

        if complexity_score < 0.3:
            return 'low'
        elif complexity_score < 0.7:
            return 'medium'
        else:
            return 'high'

# Batch processing function
def analyze_visual_content_directory(input_dir: str, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Analyze all images in a directory."""
    analyzer = VisualContentAnalyzer(config)
    results = {}

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}

    for image_file in Path(input_dir).rglob('*'):
        if image_file.suffix.lower() in image_extensions:
            try:
                analysis = analyzer.analyze_image(str(image_file))
                results[str(image_file)] = analysis

                # Save individual analysis
                output_file = Path(output_dir) / f"{image_file.stem}_visual_analysis.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Failed to analyze {image_file}: {e}")
                results[str(image_file)] = {'error': str(e)}

    return results
