# -*- coding: utf-8 -*-
"""
Image Processing Module
Provides template matching and image filtering functionality
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import cv2
import numpy as np

from modules.constants import (
    DEFAULT_FILTER_TYPE,
    DEFAULT_MATCH_METHOD,
    DEFAULT_THRESHOLD,
    DEFAULT_CANNY_T1,
    DEFAULT_CANNY_T2,
    CV2_MATCH_METHODS
)

logger = logging.getLogger(__name__)

# LRU Template Cache for improved performance
class TemplateCache:
    """Simple LRU Cache for processed template images"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize template cache with maximum size"""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
        logger.info(f"Template cache initialized with max size: {max_size}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get template from cache if it exists"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, template_data: Dict[str, Any]) -> None:
        """Add template to cache, possibly evicting oldest entries"""
        # Check if cache is full and needs eviction
        if len(self.cache) >= self.max_size:
            # Find least recently used entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            # Remove it
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        # Add new entry
        self.cache[key] = template_data
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear the template cache"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Template cache cleared")

# Image Processor Class
class ImageProcessor:
    """
    Handles image processing operations including template loading,
    preprocessing with filters, and template matching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image processor
        
        Args:
            config: Configuration dictionary with default settings
        """
        self.config = config
        self.template_cache = TemplateCache(config.get('template_cache_size', 1000))
        logger.info("Image processor initialized")
    
    def _load_template(self, template_path: str) -> Optional[np.ndarray]:
        """
        Load template image from disk
        
        Args:
            template_path: Path to template image
            
        Returns:
            Template image as numpy array or None if loading failed
        """
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return None
        
        try:
            template = cv2.imread(template_path)
            if template is None:
                logger.error(f"Failed to load template: {template_path}")
                return None
                
            # Convert to grayscale for matching
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
            return template
            
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")
            return None
    
    def _preprocess_image(
        self, 
        image: np.ndarray, 
        filter_type: str,
        canny_t1: int = DEFAULT_CANNY_T1,
        canny_t2: int = DEFAULT_CANNY_T2
    ) -> np.ndarray:
        """
        Apply preprocessing filters to an image
        
        Args:
            image: Input image (grayscale)
            filter_type: Filter type to apply ('none' or 'canny')
            canny_t1: Low threshold for Canny edge detection
            canny_t2: High threshold for Canny edge detection
            
        Returns:
            Processed image
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply filter
        if filter_type.lower() == 'canny':
            try:
                return cv2.Canny(gray, canny_t1, canny_t2)
            except Exception as e:
                logger.error(f"Error applying Canny filter: {e}")
                return gray
        else:
            # Default is 'none' - just return grayscale
            return gray
    
    def _get_match_method_cv2(self, method_str: str) -> int:
        """
        Get the OpenCV constant for match method
        
        Args:
            method_str: Name of match method ('ccoeff_normed', 'sqdiff_normed', or 'ccorr_normed')
            
        Returns:
            OpenCV constant for match method
        """
        method_map = {
            'ccoeff_normed': cv2.TM_CCOEFF_NORMED,
            'sqdiff_normed': cv2.TM_SQDIFF_NORMED,
            'ccorr_normed': cv2.TM_CCORR_NORMED
        }
        return method_map.get(method_str.lower(), cv2.TM_CCOEFF_NORMED)
    
    def _calculate_score(self, match_val: float, method_str: str) -> float:
        """
        Calculate normalized score (0.0-1.0) for match value
        
        Args:
            match_val: Raw match value from cv2.matchTemplate
            method_str: Match method used
            
        Returns:
            Normalized score (higher is better)
        """
        # For TM_SQDIFF_NORMED, 0 is perfect match and 1 is worst match
        # For other methods, 1 is perfect match and -1 is worst match
        if method_str.lower() == 'sqdiff_normed':
            return 1.0 - match_val  # Invert so higher is better
        return max(0.0, match_val)  # Ensure non-negative
    
    def _generate_cache_key(
        self, 
        template_path: str, 
        filter_type: str, 
        canny_t1: int = None, 
        canny_t2: int = None
    ) -> str:
        """
        Generate cache key for template
        
        Args:
            template_path: Path to template image
            filter_type: Filter type applied
            canny_t1: Canny low threshold (if applicable)
            canny_t2: Canny high threshold (if applicable)
            
        Returns:
            Cache key string
        """
        key = f"{template_path}|{filter_type}"
        if filter_type.lower() == 'canny' and canny_t1 is not None and canny_t2 is not None:
            key += f"|{canny_t1}|{canny_t2}"
        return key
    
    def _get_cached_template(
        self, 
        template_path: str, 
        filter_type: str,
        canny_t1: int = None,
        canny_t2: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get processed template from cache or create and cache it
        
        Args:
            template_path: Path to template image
            filter_type: Filter type to apply
            canny_t1: Canny low threshold (if applicable)
            canny_t2: Canny high threshold (if applicable)
            
        Returns:
            Dictionary with processed template and metadata or None if failed
        """
        cache_key = self._generate_cache_key(template_path, filter_type, canny_t1, canny_t2)
        
        # Check cache first
        cached = self.template_cache.get(cache_key)
        if cached:
            return cached
        
        # Load and process template
        template = self._load_template(template_path)
        if template is None:
            return None
            
        # Process template with filter
        processed_template = self._preprocess_image(
            template, 
            filter_type, 
            canny_t1 or DEFAULT_CANNY_T1,
            canny_t2 or DEFAULT_CANNY_T2
        )
        
        # Create template data
        template_data = {
            'original': template,
            'processed': processed_template,
            'width': processed_template.shape[1],
            'height': processed_template.shape[0],
            'path': template_path,
            'name': os.path.basename(template_path),
            'filter_type': filter_type
        }
        
        # Cache it
        self.template_cache.put(cache_key, template_data)
        return template_data
    
    def match_template(
        self,
        frame: np.ndarray,
        template_path: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Match template in frame using parameters
        
        Args:
            frame: Source frame to search in
            template_path: Path to template image
            params: Dictionary of matching parameters
            
        Returns:
            Dictionary with match results
        """
        # Extract parameters with defaults from config
        filter_type = params.get('filter_type', self.config.get('default_filter_type', DEFAULT_FILTER_TYPE))
        match_method = params.get('match_method', self.config.get('default_match_method', DEFAULT_MATCH_METHOD))
        threshold = float(params.get('threshold', self.config.get('default_threshold', DEFAULT_THRESHOLD)))
        
        # Extract Canny parameters if needed
        canny_t1 = None
        canny_t2 = None
        if filter_type.lower() == 'canny':
            canny_t1 = int(params.get('canny_t1', self.config.get('default_canny_t1', DEFAULT_CANNY_T1)))
            canny_t2 = int(params.get('canny_t2', self.config.get('default_canny_t2', DEFAULT_CANNY_T2)))
        
        # Load and prepare template
        template_data = self._get_cached_template(template_path, filter_type, canny_t1, canny_t2)
        if not template_data:
            return {
                'found': False,
                'error': f"Failed to load template: {template_path}",
                'filter_type_used': filter_type,
                'match_method_used': match_method,
                'threshold': threshold
            }
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Extract search region parameters
        x1 = params.get('match_range_x1')
        y1 = params.get('match_range_y1')
        x2 = params.get('match_range_x2')
        y2 = params.get('match_range_y2')
        
        # Initialize search region to full frame
        search_region = {'x1': 0, 'y1': 0, 'x2': frame_width, 'y2': frame_height}
        full_search = True
        
        # Apply search region if all coordinates are provided
        if all(coord is not None for coord in [x1, y1, x2, y2]):
            # Convert to int and validate
            try:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame_width, int(x2)), min(frame_height, int(y2))
                
                # Ensure x1 < x2 and y1 < y2
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Invalid search region: ({x1},{y1})-({x2},{y2}). Using full frame.")
                else:
                    search_region = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    full_search = False
                    logger.debug(f"Using search region: ({x1},{y1})-({x2},{y2})")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing search region coordinates: {e}. Using full frame.")
        
        # Extract region from frame
        if full_search:
            region = frame
        else:
            region = frame[search_region['y1']:search_region['y2'], search_region['x1']:search_region['x2']]
        
        # Apply the same preprocessing to the frame
        processed_region = self._preprocess_image(
            region, 
            filter_type, 
            canny_t1 or DEFAULT_CANNY_T1,
            canny_t2 or DEFAULT_CANNY_T2
        )
        
        # Get template
        template = template_data['processed']
        
        # Check if template is larger than search region
        if template.shape[0] > processed_region.shape[0] or template.shape[1] > processed_region.shape[1]:
            return {
                'found': False,
                'error': "Template is larger than search region",
                'filter_type_used': filter_type,
                'match_method_used': match_method,
                'threshold': threshold,
                'search_region_x1': search_region['x1'],
                'search_region_y1': search_region['y1'],
                'search_region_x2': search_region['x2'],
                'search_region_y2': search_region['y2'],
                'search_region_full_search': full_search,
                'highest_score': 0.0,
                'frame_width': frame_width,
                'frame_height': frame_height
            }
        
        # Perform template matching
        try:
            cv2_method = self._get_match_method_cv2(match_method)
            result = cv2.matchTemplate(processed_region, template, cv2_method)
            
            # Find the best match location
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Calculate score based on match method
            if match_method.lower() == 'sqdiff_normed':
                match_loc = min_loc
                score = self._calculate_score(min_val, match_method)
                highest_score = self._calculate_score(min_val, match_method)
            else:
                match_loc = max_loc
                score = self._calculate_score(max_val, match_method)
                highest_score = self._calculate_score(max_val, match_method)
            
            # Add search region offset to coordinates
            if not full_search:
                match_loc = (match_loc[0] + search_region['x1'], match_loc[1] + search_region['y1'])
            
            # Convert match location to result coordinates
            top_left_x, top_left_y = match_loc
            width, height = template_data['width'], template_data['height']
            center_x = top_left_x + width // 2
            center_y = top_left_y + height // 2
            
            # Apply offsets if provided
            offset_x = int(params.get('offset_x', 0))
            offset_y = int(params.get('offset_y', 0))
            
            # Check if the match exceeds the threshold
            found = score >= threshold
            
            # Create result dictionary
            result = {
                'found': found,
                'center_x': center_x + offset_x,
                'center_y': center_y + offset_y,
                'template_name': template_data['name'],
                'template_path': template_data['path'],
                'score': score,
                'top_left_x': top_left_x,
                'top_left_y': top_left_y,
                'width': width,
                'height': height,
                'top_left_x_with_offset': top_left_x + offset_x,
                'top_left_y_with_offset': top_left_y + offset_y,
                'offset_applied_x': offset_x,
                'offset_applied_y': offset_y,
                'verify_wait': 0.0,
                'verify_confirmed': False,
                'verify_score': None,
                'recheck_status': "Not performed",
                'recheck_frame_timestamp': None,
                'search_region_x1': search_region['x1'],
                'search_region_y1': search_region['y1'],
                'search_region_x2': search_region['x2'],
                'search_region_y2': search_region['y2'],
                'search_region_full_search': full_search,
                'filter_type_used': filter_type,
                'match_method_used': match_method,
                'frame_timestamp': time.time(),
                'frame_width': frame_width,
                'frame_height': frame_height,
                'threshold': threshold,
                'highest_score': highest_score,
                'error': None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during template matching: {e}")
            return {
                'found': False,
                'error': f"Template matching error: {str(e)}",
                'filter_type_used': filter_type,
                'match_method_used': match_method,
                'threshold': threshold,
                'search_region_x1': search_region['x1'],
                'search_region_y1': search_region['y1'],
                'search_region_x2': search_region['x2'],
                'search_region_y2': search_region['y2'],
                'search_region_full_search': full_search,
                'highest_score': 0.0,
                'frame_width': frame_width,
                'frame_height': frame_height
            } 