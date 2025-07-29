# corrected_tooth_analysis.py
# Tooth parallelism analysis with corrected anatomical numbering

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import math

class CorrectedToothAnalysis:
    def __init__(self):
        self.model = None
        self.teeth_data = []
        self.individual_teeth = []
        self.tooth_axes = []
        self.anatomical_mapping = {}
        
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
        
        # Run segmentation
        results = self.model(image_path)
        
        # Clear previous data
        self.teeth_data = []
        self.individual_teeth = []
        self.tooth_axes = []
        
        # Extract tooth data
        for r in results:
            if r.masks is not None:
                masks = r.masks.data
                boxes = r.boxes.data
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    polygon = r.masks.xy[i]
                    x1, y1, x2, y2, conf, cls = box
                    
                    tooth = {
                        'yolo_id': i,  # Original YOLO ID
                        'polygon': polygon,
                        'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                        'confidence': conf.item(),
                        'center_x': (x1.item() + x2.item()) / 2,
                        'center_y': (y1.item() + y2.item()) / 2
                    }
                    self.teeth_data.append(tooth)
        
        print(f"✅ Found {len(self.teeth_data)} teeth")
        
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
        """Cut out each tooth from the original image"""
        self.individual_teeth = []
        
        for tooth in self.teeth_data:
            # Get bounding box
            x1, y1, x2, y2 = [int(v) for v in tooth['bbox']]
            
            # Add padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(self.image.shape[1], x2 + padding)
            y2 = min(self.image.shape[0], y2 + padding)
            
            # Create mask for this tooth
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            if len(tooth['polygon']) > 0:
                pts = np.array(tooth['polygon'], np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
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
        
        print(f"✂️ Extracted {len(self.individual_teeth)} individual teeth")
        return self.individual_teeth
    
    def calculate_straight_axis(self, tooth_data):
        """Calculate straight axis line for a tooth using linear regression"""
        mask = tooth_data['mask']
        h, w = mask.shape
        
        # Get center points (row-by-row)
        centers = []
        for y in range(h):
            row = mask[y, :]
            if np.sum(row) > 0:
                x_coords = np.where(row > 0)[0]
                center_x = np.mean(x_coords)
                centers.append([center_x, y])
        
        if len(centers) < 2:
            return None
        
        centers = np.array(centers)
        
        # Linear regression through all center points
        X = centers[:, 1].reshape(-1, 1)  # y-coordinates
        y = centers[:, 0]  # x-coordinates
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get top and bottom y-coordinates
        y_top = centers[0, 1]
        y_bottom = centers[-1, 1]
        
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
            'centers': centers
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

    def draw_straight_axes_individual(self, save_path=None):
        """Draw straight axis lines on individual teeth (like the original version)"""
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
            
            # Use anatomical numbering for title
            ax.set_title(f"T{tooth_data['anatomical_id']} (YOLO:{tooth_data['yolo_id']})", 
                        fontsize=12)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_teeth, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.suptitle('Individual Teeth with Straight Axes (Corrected Anatomical Order)', 
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
    

    
    def visualize_corrected_analysis(self, save_path=None):
        """Visualize the corrected analysis"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        
        # Left: Show corrected numbering
        ax1.imshow(self.image_rgb)
        ax1.set_title('Corrected Anatomical Numbering', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Middle: Segmented teeth with axis lines
        ax2.imshow(self.image_rgb)
        ax2.set_title('Segmented Teeth with Axis Lines', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # Right: Parallelism analysis with correct adjacency
        ax3.imshow(self.image_rgb)
        ax3.set_title('Corrected Parallelism Analysis', fontsize=16, fontweight='bold')
        ax3.axis('off')
        
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
        
        # Draw teeth with correct numbering
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
                
                # Left plot - show anatomical numbers
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
                
                # Show YOLO ID for reference
                ax1.text(tooth['center_x'], tooth['bbox'][1] - 15, 
                        f"ID:{tooth['yolo_id']}", 
                        color='black', 
                        fontsize=8,
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor='yellow', 
                                alpha=0.7))
                
                # Middle plot - segmented teeth with axis lines
                # Draw polygon
                if len(tooth['polygon']) > 0:
                    polygon = plt.Polygon(tooth['polygon'], 
                                        fill=False, 
                                        edgecolor=colors[idx], 
                                        linewidth=1,
                                        alpha=0.8)
                    ax2.add_patch(polygon)
                
                # Draw axis line
                ax2.plot([top_x, bottom_x], [top_y, bottom_y], 
                        color=colors[idx], linewidth=1, alpha=0.9)
                
                # Add tooth number
                ax2.text(tooth['center_x'], tooth['center_y'], 
                        f"T{anatomical_id}", 
                        color='white', 
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=colors[idx], 
                                alpha=0.8))
                
                # Right plot - parallelism with color
                color = tooth_colors.get(anatomical_id, 'gray')
                ax3.plot([top_x, bottom_x], [top_y, bottom_y], 
                        color=color, linewidth=1, alpha=0.8)
                
                # Add angle
                ax3.text(top_x, top_y - 20, 
                        f"{axis_data['axis']['angle']:.1f}°", 
                        color='white', 
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=color, 
                                alpha=0.8))
        
        # Add legend to right plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Excellent (<5°)'),
            Patch(facecolor='yellow', label='Good (5-10°)'),
            Patch(facecolor='orange', label='Fair (10-15°)'),
            Patch(facecolor='red', label='Poor (>15°)')
        ]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved to: {save_path}")
        
        plt.show()
        
        # Print corrected report
        self.print_corrected_report(parallelism_results)
    
    def print_corrected_report(self, results):
        """Print the corrected parallelism report"""
        print("\n" + "="*80)
        print("📊 CORRECTED ORTHODONTIC PARALLELISM REPORT")
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

# MAIN
# MAIN
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = '/Users/minoo/Desktop/runs/segment/teeth_chunk_3/weights/best.pt'  # Updated path
    IMAGE_PATH = 'test2.jpg'  # Your test image
    
    print("🦷 CORRECTED TOOTH PARALLELISM ANALYSIS")
    print("="*60)
    
    # Create analyzer
    analyzer = CorrectedToothAnalysis()
    
    # Load model
    if analyzer.load_model(MODEL_PATH):
        # Process image with corrected mapping
        if analyzer.process_image(IMAGE_PATH) > 0:
            # Extract teeth
            analyzer.extract_individual_teeth()
            
            # Calculate axes
            analyzer.calculate_all_axes()
            
            # NEW: Show individual teeth with axes (like the original version)
            print("\n📊 Drawing individual teeth with straight axes...")
            analyzer.draw_straight_axes_individual(save_path='individual_teeth_axes_corrected.jpg')
            
            # Show the corrected analysis visualization
            print("\n📊 Running corrected parallelism analysis...")
            analyzer.visualize_corrected_analysis(save_path='corrected_orthodontic_analysis.jpg')