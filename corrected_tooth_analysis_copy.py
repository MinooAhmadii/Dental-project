# corrected_tooth_analysis.py
# Tooth parallelism analysis with corrected anatomical numbering

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import math
import pandas as pd 
import yaml 
import time  
import json  

class CorrectedToothAnalysis:
    def __init__(self):
        self.model = None
        self.teeth_data = []
        self.individual_teeth = []
        self.tooth_axes = []
        self.anatomical_mapping = {}

    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def remove_overlapping_teeth(self, iou_threshold=0.3):
        """Remove overlapping tooth detections using IoU"""
        if not self.teeth_data:
            return
        
        # Sort by confidence (highest first)
        self.teeth_data.sort(key=lambda x: x['confidence'], reverse=True)
        
        kept_teeth = []
        
        for i, tooth in enumerate(self.teeth_data):
            keep = True
            
            # Check IoU with already kept teeth
            for kept_tooth in kept_teeth:
                iou = self.calculate_iou(tooth['bbox'], kept_tooth['bbox'])
                
                if iou > iou_threshold:
                    # If overlapping, keep the one with higher confidence (already kept)
                    keep = False
                    break
            
            if keep:
                kept_teeth.append(tooth)
        
        removed_count = len(self.teeth_data) - len(kept_teeth)
        if removed_count > 0:
            print(f"  ❌ Removed {removed_count} overlapping detections")
        
        self.teeth_data = kept_teeth
     
    def load_model(self, model_path):
        """Load YOLO model"""
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Model loaded!")
            return True
        return False
    
    def process_image(self, image_path):
        """Segment teeth and extract data"""
        print(f"\n🔍 Processing: {image_path}")
        
        # Load image
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation (back to original - no confidence filtering)
        results = self.model(image_path)
        
        # Clear previous data
        self.teeth_data = []
        self.individual_teeth = []
        self.tooth_axes = []
        
        # Extract tooth data (original logic)
        for r in results:
            if r.masks is not None:
                masks = r.masks.data
                boxes = r.boxes.data
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    polygon = r.masks.xy[i]
                    x1, y1, x2, y2, conf, cls = box
                    
                    # ADD THIS: Get the class name
                    class_name = self.model.names[int(cls)]
                    
                    tooth = {
                        'yolo_id': i,  # Original YOLO ID
                        'polygon': polygon,
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': conf.item(),
                        'center_x': (x1.item() + x2.item()) / 2,
                        'center_y': (y1.item() + y2.item()) / 2,
                        'cavity_class': class_name  # ADD THIS
                    }
                    self.teeth_data.append(tooth)
        
        print(f"✅ Found {len(self.teeth_data)} initial detections")
        
        # ADD THIS: Remove overlapping detections
        self.remove_overlapping_teeth(iou_threshold=0.3)  # Adjust threshold as needed
        
        print(f"✅ After overlap removal: {len(self.teeth_data)} teeth")
        
        # Create anatomical mapping
        self.create_anatomical_mapping()
        
        return len(self.teeth_data)
    
    def create_anatomical_mapping(self):
        """Create mapping from YOLO IDs to anatomical positions"""
        # Separate upper and lower teeth
        teeth_sorted_y = sorted(self.teeth_data, key=lambda x: x['center_y'])
        
        # Find the gap between upper and lower teeth
        y_diffs = []
        for i in range(len(teeth_sorted_y) - 1):
            y_diff = teeth_sorted_y[i+1]['center_y'] - teeth_sorted_y[i]['center_y']
            y_diffs.append((y_diff, i))
        
        # The largest gap should be between upper and lower teeth
        max_gap_idx = max(y_diffs, key=lambda x: x[0])[1]
        
        # Split into upper and lower
        upper_teeth = teeth_sorted_y[:max_gap_idx + 1]
        lower_teeth = teeth_sorted_y[max_gap_idx + 1:]
        
        # Sort each row by x-coordinate (left to right in image)
        upper_teeth.sort(key=lambda x: x['center_x'])
        lower_teeth.sort(key=lambda x: x['center_x'])
        
        print(f"\n📊 Teeth Distribution:")
        print(f"  Upper teeth: {len(upper_teeth)}")
        print(f"  Lower teeth: {len(lower_teeth)}")
        
        # Create mapping - Upper teeth numbered 1 to N, Lower teeth numbered N+1 to total
        self.anatomical_mapping = {}
        
        # Map upper teeth (1 to number of upper teeth)
        for i, tooth in enumerate(upper_teeth):
            anatomical_num = i + 1
            self.anatomical_mapping[tooth['yolo_id']] = anatomical_num
            tooth['anatomical_id'] = anatomical_num
        
        # Map lower teeth (starting from number of upper teeth + 1)
        upper_count = len(upper_teeth)
        for i, tooth in enumerate(lower_teeth):
            anatomical_num = upper_count + i + 1
            self.anatomical_mapping[tooth['yolo_id']] = anatomical_num
            tooth['anatomical_id'] = anatomical_num
        
        print(f"\n📋 Anatomical Mapping Created:")
        print(f"  Upper teeth: T1 - T{len(upper_teeth)}")
        print(f"  Lower teeth: T{upper_count + 1} - T{upper_count + len(lower_teeth)}")
    
    def extract_individual_teeth(self):
        """Cut out each tooth from the original image with improved segmentation"""
        self.individual_teeth = []
        
        for tooth in self.teeth_data:
            # Get bounding box
            x1, y1, x2, y2 = [int(v) for v in tooth['bbox']]
            
            # Add padding
            padding = 15  # Increased padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(self.image.shape[1], x2 + padding)
            y2 = min(self.image.shape[0], y2 + padding)
            
            # Create mask for this tooth
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            if len(tooth['polygon']) > 0:
                pts = np.array(tooth['polygon'], np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
                # Only keep mask refinement (gentle improvement)
                kernel = np.ones((2,2), np.uint8)  # Smaller kernel
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill tiny gaps only
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
                
                # Smooth the mask edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                mask = (mask > 127).astype(np.uint8) * 255  # Re-threshold
            
            # Extract tooth region
            tooth_roi = self.image[y1:y2, x1:x2].copy()
            mask_roi = mask[y1:y2, x1:x2]
            
            # Apply mask to get only the tooth
            tooth_only = cv2.bitwise_and(tooth_roi, tooth_roi, mask=mask_roi)
            
            # Store individual tooth data with correct anatomical ID
            individual_tooth = {
                'yolo_id': tooth['yolo_id'],
                'anatomical_id': tooth['anatomical_id'],
                'image': tooth_only,
                'mask': mask_roi,
                'bbox_in_original': [x1, y1, x2, y2],
                'original_bbox': tooth['bbox'],
                'confidence': tooth['confidence']
            }
            
            self.individual_teeth.append(individual_tooth)
        
        # Sort by anatomical ID for proper ordering
        self.individual_teeth.sort(key=lambda x: x['anatomical_id'])
        
        print(f"✂️ Extracted {len(self.individual_teeth)} individual teeth with improved masks")
        return self.individual_teeth
    
    def calculate_straight_axis(self, tooth_data):
        """Calculate straight axis line for a tooth using better center finding"""
        mask = tooth_data['mask']
        h, w = mask.shape
        
        # Get better centers for each row
        better_centers = []
        
        for y in range(h):
            row = mask[y, :]
            if np.sum(row) > 0:
                # Find all white pixels in this row
                x_coords = np.where(row > 0)[0]
                
                if len(x_coords) > 2:
                    # SIMPLE FIX: Find the middle between tooth edges instead of pixel average
                    leftmost = x_coords[0]
                    rightmost = x_coords[-1]
                    center_x = (leftmost + rightmost) / 2  # True center between edges
                    
                    better_centers.append([center_x, y])  # Use this one only
        
        if len(better_centers) < 3:
            return None
        
        better_centers = np.array(better_centers)
        
        # Same linear regression as before, but with better centers
        X = better_centers[:, 1].reshape(-1, 1)  # y-coordinates
        y = better_centers[:, 0]  # x-coordinates
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get top and bottom y-coordinates
        y_top = better_centers[0, 1]
        y_bottom = better_centers[-1, 1]
        
        # Predict x-coordinates for straight line
        x_top = model.predict([[y_top]])[0]
        x_bottom = model.predict([[y_bottom]])[0]
        
        # Calculate angle from vertical
        dx = x_bottom - x_top
        dy = y_bottom - y_top
        angle = math.degrees(math.atan2(dx, dy))
        
        axis_data = {
            'top_point': [x_top, y_top],
            'bottom_point': [x_bottom, y_bottom],
            'angle': angle,
            'centers': better_centers  # Use better_centers, not centers
        }
        
        return axis_data
    
    def calculate_all_axes(self):
        """Calculate axes for all teeth"""
        self.tooth_axes = []
        
        for tooth_data in self.individual_teeth:
            axis_data = self.calculate_straight_axis(tooth_data)
            
            if axis_data is not None:
                self.tooth_axes.append({
                    'anatomical_id': tooth_data['anatomical_id'],
                    'yolo_id': tooth_data['yolo_id'],
                    'axis': axis_data,
                    'bbox_in_original': tooth_data['bbox_in_original']
                })
        
        print(f"📏 Calculated {len(self.tooth_axes)} tooth axes")

    def refine_with_sam(self, sam_checkpoint_path):
        """Refine YOLO detections using SAM for better segmentation"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch
            
            print("🎯 Loading SAM model for refinement...")
            
            # Load SAM
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            predictor = SamPredictor(sam)
            predictor.set_image(self.image_rgb)
            
            print("🔧 Refining tooth segmentation with SAM...")
            
            # Refine each tooth mask using SAM
            refined_teeth = []
            
            for i, tooth in enumerate(self.teeth_data):
                x1, y1, x2, y2 = tooth['bbox']
                
                # Use multiple prompts for better results
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Method 1: Center point + bounding box
                masks_combined, scores_combined, _ = predictor.predict(
                    point_coords=np.array([[center_x, center_y]]),
                    point_labels=np.array([1]),
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True
                )
                
                # Method 2: Multiple points across the tooth
                # Add corner points and center for more robust segmentation
                points = np.array([
                    [center_x, center_y],                    # Center
                    [int(x1 + (x2-x1)*0.3), center_y],      # Left-center
                    [int(x1 + (x2-x1)*0.7), center_y],      # Right-center
                    [center_x, int(y1 + (y2-y1)*0.3)],      # Top-center
                    [center_x, int(y1 + (y2-y1)*0.7)]       # Bottom-center
                ])
                point_labels = np.array([1, 1, 1, 1, 1])  # All positive points
                
                masks_multi, scores_multi, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True
                )
                
                # Choose the best mask from both methods
                all_masks = np.concatenate([masks_combined, masks_multi])
                all_scores = np.concatenate([scores_combined, scores_multi])
                
                best_idx = np.argmax(all_scores)
                best_mask = all_masks[best_idx]
                best_score = all_scores[best_idx]
                
                # Calculate IoU with original YOLO mask for quality check
                original_mask = np.zeros(self.image.shape[:2], dtype=bool)
                if len(tooth['polygon']) > 0:
                    pts = np.array(tooth['polygon'], np.int32)
                    cv2.fillPoly(original_mask.astype(np.uint8), [pts], 1)
                    original_mask = original_mask.astype(bool)
                    
                    # Calculate IoU
                    intersection = np.logical_and(best_mask, original_mask)
                    union = np.logical_or(best_mask, original_mask)
                    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
                else:
                    iou = 0
                
                # Update polygon from SAM mask
                contours, _ = cv2.findContours(
                    best_mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Simplify the contour
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # More precise
                    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # Update tooth data with SAM results
                    refined_tooth = tooth.copy()
                    refined_tooth['polygon'] = simplified_contour.squeeze()
                    refined_tooth['sam_score'] = best_score
                    refined_tooth['iou_with_yolo'] = iou
                    refined_tooth['refinement_method'] = 'SAM'
                    
                    refined_teeth.append(refined_tooth)
                    
                    print(f"  Tooth {i+1}: SAM score={best_score:.3f}, IoU with YOLO={iou:.3f}")
                else:
                    # If SAM fails, keep original
                    refined_teeth.append(tooth)
                    print(f"  Tooth {i+1}: SAM failed, keeping YOLO mask")
            
            # Replace original data with refined data
            self.teeth_data = refined_teeth
            
            # Recreate anatomical mapping with refined data
            self.create_anatomical_mapping()
            
            print(f"✅ SAM refinement complete! Refined {len(refined_teeth)} teeth")
            
            # Print quality statistics
            sam_scores = [t.get('sam_score', 0) for t in self.teeth_data if 'sam_score' in t]
            ious = [t.get('iou_with_yolo', 0) for t in self.teeth_data if 'iou_with_yolo' in t]
            
            if sam_scores:
                print(f"📊 SAM Quality Metrics:")
                print(f"  Average SAM confidence: {np.mean(sam_scores):.3f}")
                print(f"  Average IoU with YOLO: {np.mean(ious):.3f}")
                print(f"  Min IoU: {np.min(ious):.3f}, Max IoU: {np.max(ious):.3f}")
            
            return True
            
        except ImportError:
            print("❌ SAM not available. Install with:")
            print("   pip install git+https://github.com/facebookresearch/segment-anything.git")
            print("   Download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return False
        except Exception as e:
            print(f"❌ SAM refinement failed: {e}")
            return False

    def test_confidence_levels(self, image_path):
        """Test different confidence thresholds to find optimal segmentation"""
        print("\n🧪 Testing different confidence levels:")
        confidence_levels = [0.3, 0.5, 0.6, 0.7, 0.8]
        
        for conf in confidence_levels:
            results = self.model(image_path, conf=conf)
            tooth_count = 0
            if results[0].masks is not None:
                tooth_count = len(results[0].masks.data)
            print(f"  Confidence {conf}: {tooth_count} teeth detected")
        
        print("💡 Try different confidence values in process_image() method")

    def draw_straight_axes_individual(self, save_path=None):
        """Draw straight axis lines on individual teeth without tooth labels"""
        n_teeth = len(self.individual_teeth)
        if n_teeth == 0:
            print("No teeth to display")
            return
            
        cols = min(7, n_teeth)
        rows = (n_teeth + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Sort individual teeth by anatomical ID for proper display order
        sorted_individual_teeth = sorted(self.individual_teeth, key=lambda x: x['anatomical_id'])

        for idx, tooth_data in enumerate(sorted_individual_teeth):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Convert to RGB
            tooth_rgb = cv2.cvtColor(tooth_data['image'], cv2.COLOR_BGR2RGB)
            ax.imshow(tooth_rgb)
            
            # Calculate straight axis
            axis_data = self.calculate_straight_axis(tooth_data)
            
            if axis_data is not None:
                # Draw the straight line
                ax.plot([axis_data['top_point'][0], axis_data['bottom_point'][0]],
                        [axis_data['top_point'][1], axis_data['bottom_point'][1]],
                        'r-', linewidth=1, label='Axis')
                
                # Draw center points (faded)
                ax.plot(axis_data['centers'][:, 0], axis_data['centers'][:, 1],
                        'g.', markersize=1, alpha=0.3, label='Centers')
                
                # Mark endpoints
                ax.plot(axis_data['top_point'][0], axis_data['top_point'][1],
                        'bo', markersize=2, label='Crown')
                ax.plot(axis_data['bottom_point'][0], axis_data['bottom_point'][1],
                        'ro', markersize=2, label='Root')
                
                # Add angle text
                ax.text(5, 15, f"{axis_data['angle']:.1f}°",
                        color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
            
            # REMOVED: Tooth title/label - no anatomical numbering displayed
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_teeth, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.suptitle('Individual Teeth with Straight Axes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Individual teeth axes saved to: {save_path}")
        
        plt.show()


        
    def analyze_corrected_parallelism(self):
        """Analyze parallelism between anatomically adjacent teeth"""
        # Sort by anatomical ID
        sorted_axes = sorted(self.tooth_axes, key=lambda x: x['anatomical_id'])
        
        # Find the boundary between upper and lower teeth
        # Upper teeth have lower anatomical IDs, lower teeth have higher IDs
        upper_count = len([t for t in self.teeth_data if 'anatomical_id' in t and t['anatomical_id'] <= len([t for t in self.teeth_data if t['center_y'] < sorted(self.teeth_data, key=lambda x: x['center_y'])[len(self.teeth_data)//2]['center_y']])])
        
        # Separate upper and lower for proper adjacency
        upper_teeth = [t for t in sorted_axes if t['anatomical_id'] <= upper_count]
        lower_teeth = [t for t in sorted_axes if t['anatomical_id'] > upper_count]
        
        parallelism_results = []
        
        # Analyze upper teeth adjacency
        for i in range(len(upper_teeth) - 1):
            tooth1 = upper_teeth[i]
            tooth2 = upper_teeth[i + 1]
            
            angle1 = tooth1['axis']['angle']
            angle2 = tooth2['axis']['angle']
            angle_diff = abs(angle1 - angle2)
            
            result = self.rate_parallelism(tooth1, tooth2, angle1, angle2, angle_diff)
            parallelism_results.append(result)
        
        # Analyze lower teeth adjacency
        for i in range(len(lower_teeth) - 1):
            tooth1 = lower_teeth[i]
            tooth2 = lower_teeth[i + 1]
            
            angle1 = tooth1['axis']['angle']
            angle2 = tooth2['axis']['angle']
            angle_diff = abs(angle1 - angle2)
            
            result = self.rate_parallelism(tooth1, tooth2, angle1, angle2, angle_diff)
            parallelism_results.append(result)
        
        return parallelism_results
    
    def rate_parallelism(self, tooth1, tooth2, angle1, angle2, angle_diff):
        """Rate parallelism between two teeth"""
        if angle_diff < 5:
            rating = "Excellent"
            color = 'green'
        elif angle_diff < 10:
            rating = "Good"
            color = 'yellow'
        elif angle_diff < 15:
            rating = "Fair"
            color = 'orange'
        else:
            rating = "Poor"
            color = 'red'
        
        return {
            'tooth1_anatomical': tooth1['anatomical_id'],
            'tooth2_anatomical': tooth2['anatomical_id'],
            'tooth1_yolo': tooth1['yolo_id'],
            'tooth2_yolo': tooth2['yolo_id'],
            'angle1': angle1,
            'angle2': angle2,
            'angle_diff': angle_diff,
            'rating': rating,
            'color': color
        }
    

    
    def visualize_corrected_analysis(self, save_path_prefix=None):
        """Visualize the corrected analysis as three separate images"""
        
        # Get parallelism results
        parallelism_results = self.analyze_corrected_parallelism()
        
        # Create color map for each tooth
        tooth_colors = {}
        for result in parallelism_results:
            # Apply worst color to each tooth
            for tooth_id in [result['tooth1_anatomical'], result['tooth2_anatomical']]:
                if tooth_id not in tooth_colors:
                    tooth_colors[tooth_id] = result['color']
                else:
                    # Keep worst color
                    color_priority = {'red': 4, 'orange': 3, 'yellow': 2, 'green': 1}
                    if color_priority.get(result['color'], 0) > color_priority.get(tooth_colors[tooth_id], 0):
                        tooth_colors[tooth_id] = result['color']
        
        # Colors for visualization
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.teeth_data)))
        
        # IMAGE 1: Anatomical Numbering
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.imshow(self.image_rgb)
        ax1.set_title('Anatomical Numbering', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # IMAGE 2: Segmented Teeth with Axis Lines  
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        ax2.imshow(self.image_rgb)
        ax2.set_title('Segmented Teeth with Axis Lines', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # IMAGE 3: Parallelism Analysis
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
        ax3.imshow(self.image_rgb)
        ax3.set_title('Parallelism Analysis', fontsize=16, fontweight='bold')
        ax3.axis('off')
        
        # Draw teeth data on all three plots
        for idx, tooth in enumerate(self.teeth_data):
            anatomical_id = self.anatomical_mapping[tooth['yolo_id']]
            
            # Find axis data
            axis_data = None
            for ta in self.tooth_axes:
                if ta['anatomical_id'] == anatomical_id:
                    axis_data = ta
                    break
            
            if axis_data:
                # Transform coordinates
                x_offset = axis_data['bbox_in_original'][0]
                y_offset = axis_data['bbox_in_original'][1]
                
                top_x = axis_data['axis']['top_point'][0] + x_offset
                top_y = axis_data['axis']['top_point'][1] + y_offset
                bottom_x = axis_data['axis']['bottom_point'][0] + x_offset
                bottom_y = axis_data['axis']['bottom_point'][1] + y_offset
                
                # PLOT 1: Show anatomical numbers (if you want them)
                ax1.text(tooth['center_x'], tooth['center_y'], 
                        f"T{anatomical_id}", 
                        color='white', 
                        fontsize=12,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.4", 
                                facecolor='darkblue', 
                                alpha=0.8))
                
                # PLOT 2: Segmented teeth with axis lines (NO LABELS)
                # Draw polygon
                if len(tooth['polygon']) > 0:
                    polygon = plt.Polygon(tooth['polygon'], 
                                        fill=False, 
                                        edgecolor=colors[idx], 
                                        linewidth=2,
                                        alpha=0.8)
                    ax2.add_patch(polygon)
                
                # Draw axis line
                ax2.plot([top_x, bottom_x], [top_y, bottom_y], 
                        color=colors[idx], linewidth=2, alpha=0.9)
                
                # PLOT 3: Parallelism with color coding
                color = tooth_colors.get(anatomical_id, 'gray')
                ax3.plot([top_x, bottom_x], [top_y, bottom_y], 
                        color=color, linewidth=2, alpha=0.8)
                
                # Add angle measurements
                ax3.text(top_x, top_y - 20, 
                        f"{axis_data['axis']['angle']:.1f}°", 
                        color='white', 
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=color, 
                                alpha=0.8))
        
        # Add legend to parallelism plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Excellent (<5°)'),
            Patch(facecolor='yellow', label='Good (5-10°)'),
            Patch(facecolor='orange', label='Fair (10-15°)'),
            Patch(facecolor='red', label='Poor (>15°)')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        # Save individual images
        if save_path_prefix:
            fig1.savefig(f"{save_path_prefix}_anatomical_numbering.jpg", dpi=150, bbox_inches='tight')
            fig2.savefig(f"{save_path_prefix}_segmented_teeth.jpg", dpi=150, bbox_inches='tight')
            fig3.savefig(f"{save_path_prefix}_parallelism_analysis.jpg", dpi=150, bbox_inches='tight')
            
            print(f"💾 Saved three separate images:")
            print(f"  - {save_path_prefix}_anatomical_numbering.jpg")
            print(f"  - {save_path_prefix}_segmented_teeth.jpg") 
            print(f"  - {save_path_prefix}_parallelism_analysis.jpg")
        
        # Show plots
        plt.figure(fig1.number)
        plt.show()
        plt.figure(fig2.number) 
        plt.show()
        plt.figure(fig3.number)
        plt.show()
        
        # Print corrected report
        self.print_corrected_report(parallelism_results)

    
    def print_corrected_report(self, results):
        """Print the corrected parallelism report"""
        print("\n" + "="*80)
        print("📊 ORTHODONTIC PARALLELISM REPORT")
        print("="*80)
        
        print(f"\n{'Comparison':<20} {'Angle 1':<10} {'Angle 2':<10} {'Difference':<12} {'Rating':<10}")
        print("-"*70)
        
        # Separate upper and lower results
        upper_results = [r for r in results if r['tooth1_anatomical'] <= 14]
        lower_results = [r for r in results if r['tooth1_anatomical'] >= 15]
        
        if upper_results:
            print("\nUPPER TEETH:")
            for r in upper_results:
                comparison = f"T{r['tooth1_anatomical']} vs T{r['tooth2_anatomical']}"
                angle1 = f"{r['angle1']:.1f}°"
                angle2 = f"{r['angle2']:.1f}°"
                diff = f"{r['angle_diff']:.1f}°"
                rating = r['rating']
                symbol = {'green': '✅', 'yellow': '⚡', 'orange': '⚠️', 'red': '❌'}[r['color']]
                
                print(f"{comparison:<20} {angle1:<10} {angle2:<10} {diff:<12} {rating:<10} {symbol}")
        
        if lower_results:
            print("\nLOWER TEETH:")
            for r in lower_results:
                comparison = f"T{r['tooth1_anatomical']} vs T{r['tooth2_anatomical']}"
                angle1 = f"{r['angle1']:.1f}°"
                angle2 = f"{r['angle2']:.1f}°"
                diff = f"{r['angle_diff']:.1f}°"
                rating = r['rating']
                symbol = {'green': '✅', 'yellow': '⚡', 'orange': '⚠️', 'red': '❌'}[r['color']]
                
                print(f"{comparison:<20} {angle1:<10} {angle2:<10} {diff:<12} {rating:<10} {symbol}")
        
        print("-"*70)
        
        # Statistics
        angle_diffs = [r['angle_diff'] for r in results]
        print(f"\nOverall Statistics:")
        print(f"  Total comparisons: {len(results)}")
        print(f"  Average angle difference: {np.mean(angle_diffs):.2f}°")
        print(f"  Maximum angle difference: {np.max(angle_diffs):.2f}°")
        
        # Problem areas
        poor_results = [r for r in results if r['rating'] == 'Poor']
        if poor_results:
            print(f"\n⚠️ TEETH REQUIRING IMMEDIATE ATTENTION:")
            for r in poor_results:
                print(f"  - T{r['tooth1_anatomical']} and T{r['tooth2_anatomical']}: "
                      f"{r['angle_diff']:.1f}° difference")
                

    def visualize_tooth_detection_only(self, save_path=None):
        """Visualize only tooth detection and segmentation without axis lines and without labels"""
        if not self.teeth_data:
            print("No teeth data available. Please run process_image() first.")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Display the original image
        ax.imshow(self.image_rgb)
        ax.set_title('Automated Tooth Detection and Segmentation', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Colors for different teeth
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.teeth_data)))
        
        # Draw tooth polygons WITHOUT numbering
        for idx, tooth in enumerate(self.teeth_data):
            # Draw polygon outline only
            if len(tooth['polygon']) > 0:
                polygon = plt.Polygon(tooth['polygon'], 
                                    fill=False, 
                                    edgecolor=colors[idx], 
                                    linewidth=2.5,
                                    alpha=0.9)
                ax.add_patch(polygon)
            
            # REMOVED: tooth number labels - no text annotations
        
        # Add detection statistics
        total_teeth = len(self.teeth_data)
        upper_teeth = len([t for t in self.teeth_data if t.get('anatomical_id', 0) <= len(self.teeth_data)//2])
        lower_teeth = total_teeth - upper_teeth
        
        # Add text box with statistics
        stats_text = f"Detection Results:\nTotal Teeth: {total_teeth}\nUpper: {upper_teeth} | Lower: {lower_teeth}"
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", 
                        facecolor='black', 
                        alpha=0.7),
                color='white',
                fontsize=11,
                fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"💾 Tooth detection visualization saved to: {save_path}")
        
        plt.show()

    def visualize_tooth_detection_with_confidence(self, save_path=None):
        """Alternative version showing confidence scores without tooth labels"""
        if not self.teeth_data:
            print("No teeth data available. Please run process_image() first.")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Display the original image
        ax.imshow(self.image_rgb)
        ax.set_title('Tooth Detection with Confidence Scores', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Sort teeth by confidence for consistent coloring
        sorted_teeth = sorted(self.teeth_data, key=lambda x: x['confidence'], reverse=True)
        
        # Create color map based on confidence
        confidences = [t['confidence'] for t in sorted_teeth]
        norm = plt.Normalize(vmin=min(confidences), vmax=max(confidences))
        cmap = plt.cm.viridis
        
        # Draw tooth polygons
        for tooth in sorted_teeth:
            confidence = tooth['confidence']
            color = cmap(norm(confidence))
            
            # Draw polygon outline
            if len(tooth['polygon']) > 0:
                polygon = plt.Polygon(tooth['polygon'], 
                                    fill=False, 
                                    edgecolor=color, 
                                    linewidth=2.5,
                                    alpha=0.9)
                ax.add_patch(polygon)
            
            # Add only confidence score (no tooth number)
            label_text = f"{confidence:.2f}"
            ax.text(tooth['center_x'], tooth['center_y'], 
                    label_text, 
                    color='white', 
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=color, 
                            alpha=0.8,
                            edgecolor='white',
                            linewidth=1))
        
        # Add colorbar for confidence
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Detection Confidence', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"💾 Confidence visualization saved to: {save_path}")
        
        plt.show()


    def print_cavity_summary(self):
        """Print summary of teeth health status"""
        cavity_counts = {}
        for tooth in self.teeth_data:
            cavity_class = tooth.get('cavity_class', 'unknown')
            cavity_counts[cavity_class] = cavity_counts.get(cavity_class, 0) + 1
        
        print("\n🦷 Teeth Health Summary:")
        print("-" * 30)
        for cavity_class, count in sorted(cavity_counts.items()):
            percentage = (count / len(self.teeth_data)) * 100
            print(f"  {cavity_class.capitalize()}: {count} teeth ({percentage:.1f}%)")

    def check_training_metrics(self):
        """Check training history and model performance"""
        import yaml
        import os
        import pandas as pd
        
        # Path to your training run
        run_path = '/Users/minoo/Desktop/runs/segment/teeth_chunk_3'
        
        # Check for results.csv (contains training metrics)
        results_path = os.path.join(run_path, 'results.csv')
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            print("\n📊 TRAINING METRICS:")
            print("-" * 50)
            
            # Get final epoch metrics
            final_metrics = df.iloc[-1]
            
            # Key metrics for segmentation
            metrics_to_show = [
                ('train/box_loss', 'Box Loss'),
                ('train/seg_loss', 'Segmentation Loss'), 
                ('train/cls_loss', 'Classification Loss'),
                ('metrics/precision(B)', 'Box Precision'),
                ('metrics/recall(B)', 'Box Recall'),
                ('metrics/mAP50(B)', 'Box mAP@0.5'),
                ('metrics/mAP50-95(B)', 'Box mAP@0.5:0.95'),
                ('metrics/precision(M)', 'Mask Precision'),
                ('metrics/recall(M)', 'Mask Recall'),
                ('metrics/mAP50(M)', 'Mask mAP@0.5'),
                ('metrics/mAP50-95(M)', 'Mask mAP@0.5:0.95'),
            ]
            
            for col, name in metrics_to_show:
                if col in df.columns:
                    value = final_metrics[col]
                    print(f"{name:.<30} {value:.4f}")
        else:
            print(f"❌ No results.csv found at {results_path}")
        
        # Check args.yaml for dataset split
        args_path = os.path.join(run_path, 'args.yaml')
        if os.path.exists(args_path):
            with open(args_path, 'r') as f:
                args = yaml.safe_load(f)
                print(f"\n📁 DATASET INFO:")
                print(f"Data path: {args.get('data', 'Unknown')}")
                print(f"Image size: {args.get('imgsz', 'Unknown')}")
                print(f"Batch size: {args.get('batch', 'Unknown')}")
                print(f"Epochs: {args.get('epochs', 'Unknown')}")
    
    def evaluate_on_test_set(self, data_yaml_path):
        """Evaluate model on the original test set"""
        print("\n🧪 EVALUATING ON TEST SET...")
        
        # First, update the data.yaml to use absolute paths
        import yaml
        
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the base directory
        base_dir = os.path.dirname(data_yaml_path)
        
        # Update paths to absolute
        data['train'] = os.path.join(base_dir, 'train', 'images')
        data['val'] = os.path.join(base_dir, 'valid', 'images')
        
        # Check if test set exists
        test_path = os.path.join(base_dir, 'test', 'images')
        if os.path.exists(test_path):
            data['test'] = test_path
            print(f"✅ Found test set with {len(os.listdir(test_path))} images")
        else:
            print("⚠️ No test set found, using validation set")
            data['test'] = data['val']
        
        # Save updated yaml temporarily
        temp_yaml = 'temp_data.yaml'
        with open(temp_yaml, 'w') as f:
            yaml.dump(data, f)
        
        try:
            # Run validation on test set
            results = self.model.val(
                data=temp_yaml,
                split='test',
                batch=16,
                conf=0.001,
                iou=0.5,
                device='cpu'
            )
            
            print("\n📊 TEST SET RESULTS:")
            print("-" * 50)
            print(f"Box mAP@0.5: {results.box.map50:.4f}")
            print(f"Box mAP@0.5:0.95: {results.box.map:.4f}")
            print(f"Mask mAP@0.5: {results.seg.map50:.4f}")
            print(f"Mask mAP@0.5:0.95: {results.seg.map:.4f}")
            
            # Per-class results
            print("\n📊 PER-CLASS PERFORMANCE:")
            for i, class_name in enumerate(self.model.names.values()):
                print(f"\n{class_name}:")
                print(f"  Box Precision: {results.box.p[i]:.4f}")
                print(f"  Box Recall: {results.box.r[i]:.4f}")
                print(f"  Box mAP@0.5: {results.box.ap50[i]:.4f}")
                print(f"  Mask mAP@0.5: {results.seg.ap50[i]:.4f}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
        
        return results
    
    def quick_performance_test(self, test_images=None):
        """Quick performance test on a few images"""
        import time
        
        if test_images is None:
            test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg']
        
        detection_counts = []
        processing_times = []
        
        print("\n🚀 QUICK PERFORMANCE TEST:")
        print("-" * 50)
        
        for img in test_images:
            if os.path.exists(img):
                start = time.time()
                count = self.process_image(img)
                processing_time = time.time() - start
                processing_times.append(processing_time)
                detection_counts.append(count)
                print(f"{img}: {count} teeth detected in {processing_time:.2f}s")
        
        if detection_counts:
            print(f"\n📊 SUMMARY:")
            print(f"Average teeth detected: {np.mean(detection_counts):.1f} ± {np.std(detection_counts):.1f}")
            print(f"Average processing time: {np.mean(processing_times):.2f}s ± {np.std(processing_times):.2f}s")
            print(f"FPS: {1/np.mean(processing_times):.2f}")

    def find_roboflow_dataset(self):
            """Help find the Roboflow dataset on your machine"""
            import os
            
            print("\n🔍 Searching for Roboflow dataset...")
            
            # Common locations where Roboflow datasets might be
            possible_locations = [
                os.path.expanduser("~/Downloads/Teeth-8"),
                os.path.expanduser("~/Downloads/teeth-gzkv1-8"),
                os.path.expanduser("~/Desktop/Teeth-8"),
                os.path.expanduser("~/Desktop/teeth-gzkv1-8"),
                "./Teeth-8",
                "./teeth-gzkv1-8",
                "../Teeth-8",
                "../teeth-gzkv1-8",
                "/Users/minoo/Downloads/Teeth-8",
                "/Users/minoo/Downloads/teeth-gzkv1-8",
                "/Users/minoo/Desktop/Teeth-8",
                "/Users/minoo/Desktop/teeth-gzkv1-8",
            ]
            
            found_datasets = []
            
            for location in possible_locations:
                if os.path.exists(location):
                    data_yaml = os.path.join(location, "data.yaml")
                    if os.path.exists(data_yaml):
                        found_datasets.append(location)
                        print(f"✅ Found dataset at: {location}")
                        
                        # Check what's in it
                        for folder in ['train', 'valid', 'test']:
                            folder_path = os.path.join(location, folder, 'images')
                            if os.path.exists(folder_path):
                                count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
                                print(f"   - {folder}: {count} images")
            
            if not found_datasets:
                print("❌ Could not find Roboflow dataset automatically.")
                print("\nPlease download your dataset from Roboflow:")
                print("1. Go to your Roboflow project")
                print("2. Download in YOLOv8 format")
                print("3. Extract to your Desktop or Downloads folder")
            
            return found_datasets


# UPDATED MAIN FUNCTION (COMPLETE REPLACEMENT)
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = '/Users/minoo/Desktop/runs/segment/teeth_chunk_3/weights/best.pt'
    IMAGE_PATH = 'test2.jpg'
    SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
    
    print("🦷 CORRECTED TOOTH PARALLELISM ANALYSIS WITH SAM REFINEMENT")
    print("="*70)
    
    # Create analyzer
    analyzer = CorrectedToothAnalysis()
    
    # Load model
    if analyzer.load_model(MODEL_PATH):
        
        print("\n" + "="*70)
        print("📊 MODEL EVALUATION")
        print("="*70)
        
        # Check training metrics
        analyzer.check_training_metrics()
        
        # Find Roboflow dataset
        found_datasets = analyzer.find_roboflow_dataset()
        
        # If dataset found, evaluate on test set
        if found_datasets:
            dataset_path = found_datasets[0]  # Use first found dataset
            data_yaml_path = os.path.join(dataset_path, 'data.yaml')
            analyzer.evaluate_on_test_set(data_yaml_path)
        else:
            print("\n⚠️ Skipping test set evaluation - dataset not found locally")
        
        # Quick performance test on multiple images
        analyzer.quick_performance_test()
        
        print("\n" + "="*70)
        print("🦷 PROCESSING MAIN IMAGE")
        print("="*70)
        
        # Step 1: YOLO detection
        if analyzer.process_image(IMAGE_PATH) > 0:
            
            # Print cavity summary
            analyzer.print_cavity_summary()
            
            # Step 2: SAM refinement (optional but recommended)
            print("\n🎯 Refining segmentation with SAM...")
            sam_success = analyzer.refine_with_sam(SAM_CHECKPOINT)
            
            if sam_success:
                print("✅ Using SAM-refined segmentation")
            else:
                print("⚠️ Using YOLO segmentation (SAM unavailable)")
            
            # Step 3: Extract individual teeth
            analyzer.extract_individual_teeth()
            
            # *** NEW STEP 3.5: Visualize tooth detection only ***
            print("\n📊 Generating tooth detection visualization...")
            analyzer.visualize_tooth_detection_only(save_path='tooth_detection_only.jpg')
            
            # Optional: Also show confidence scores
            print("\n📊 Generating tooth detection with confidence scores...")
            analyzer.visualize_tooth_detection_with_confidence(save_path='tooth_detection_confidence.jpg')
            
            # Step 4: Calculate axes
            analyzer.calculate_all_axes()
            
            # Step 5: Visualizations
            print("\n📊 Drawing individual teeth with straight axes...")
            analyzer.draw_straight_axes_individual(save_path='individual_teeth_axes_sam_refined.jpg')
            
            print("\n📊 Running corrected parallelism analysis...")
            analyzer.visualize_corrected_analysis(save_path_prefix='orthodontic_analysis')
        else:
            print("❌ No teeth detected")
    else:
        print("❌ Failed to load model")
