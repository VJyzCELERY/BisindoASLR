import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
import glob
import json
import time
import random
from tqdm import tqdm
import gc
from collections import deque
import matplotlib.pyplot as plt
from modules.GraphUtil import create_norm_adjacency_matrix,create_adjacency_matrix
from modules.GCNModel import GCNBiLSTM
from modules.Constants import NUM_NODES,FEATURE_DIM
from sklearn.preprocessing import LabelEncoder

class RealtimeSignLanguageProcessor:
    def __init__(self, model_path=None, sequence_length=3):
        """
        Initialize the real-time sign language processor
        
        Args:
            model_path: Path to the trained model file
            sequence_length: Number of frames to process for prediction
        """
        self.sequence_length = sequence_length
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create the adjacency matrix
        self.normalized_adj_matrix = create_norm_adjacency_matrix()
        self.adj_tensor = torch.tensor(self.normalized_adj_matrix, dtype=torch.float32)

        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize pose indices for extraction
        self.pose_indices = [
            # Faces
            0,1,2,3,4,5,6,7,8,9,10,
            # Shoulders
            11, 12,
            # Arms
            13, 14, 15, 16,
            # Chest
            23, 24
        ]
        
        # For UI purposes
        self.current_prediction = None
        self.prediction_confidence = 0.0
        self.prediction_time = time.time()
        self.last_update_time = time.time()
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize the model with the correct num_classes from label_encoder
        num_classes = len(checkpoint['label_map'])  # Get the number of classes from the label encoder
        
        self.model = GCNBiLSTM(
            num_nodes=NUM_NODES,
            in_features=FEATURE_DIM,
            gcn_hidden=256,
            lstm_hidden=512,
            num_classes=num_classes,
            num_gcn_layers=2,
            dropout=0.3
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_map = checkpoint['label_map']
        self.model = self.model.to(device)
        self.model.eval()
        

        
    def enhance_image(self, image):
        """Apply image enhancement techniques to improve hand detection"""
        # Convert to LAB color space and CLAHE to improve contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels back
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def detect_skin(self, image):
        """Create a skin mask to help with hand segmentation"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Second range for skin detection
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def apply_skin_mask(self, image):
        """Apply skin mask to isolate hands in the image"""
        # Get skin mask
        skin_mask = self.detect_skin(image)
        
        # Dilate mask slightly to ensure hands are fully covered
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        # Create a dimmed version of the original image for background
        dimmed = cv2.addWeighted(image, 0.3, np.zeros_like(image), 0.7, 0)
        
        # Create output image
        result = dimmed.copy()
        
        # Copy the original image where mask is set
        result[dilated_mask > 0] = image[dilated_mask > 0]
        
        return result
    
    def has_enough_finger_landmarks(self, hand_landmarks, min_finger_landmarks=5):
        """Check if enough finger landmarks are visible in the hand"""
        if hand_landmarks is None:
            return False
        
        # Count visible landmarks
        visible_count = 0
        for landmark in hand_landmarks.landmark:
            # Consider a landmark valid if it has reasonable x,y coordinates
            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                visible_count += 1
        
        return visible_count >= min_finger_landmarks
    
    def process_with_multiple_approaches(self, image_rgb, min_finger_landmarks=5):
        """Process an image with multiple detection approaches and use the best result"""
        best_results = None
        visible_landmarks_count = 0
        
        # Try with holistic model first
        with self.mp_holistic.Holistic(
            static_image_mode=False,  # Set to False for video
            model_complexity=1,  # Use 1 for real-time
            min_detection_confidence=0.6) as holistic:
            
            holistic_results = holistic.process(image_rgb)
            
            # Count detected hand landmarks
            holistic_visible_count = 0
            
            if holistic_results.left_hand_landmarks:
                for landmark in holistic_results.left_hand_landmarks.landmark:
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        holistic_visible_count += 1
            
            if holistic_results.right_hand_landmarks:
                for landmark in holistic_results.right_hand_landmarks.landmark:
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        holistic_visible_count += 1
            
            if holistic_visible_count > visible_landmarks_count:
                visible_landmarks_count = holistic_visible_count
                best_results = holistic_results
        
        # Try the dedicated hands model if we don't have enough landmarks
        if visible_landmarks_count < min_finger_landmarks * 2:
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.2) as hands:
                
                hands_results = hands.process(image_rgb)
                
                # If hands were detected
                if hands_results.multi_hand_landmarks:
                    hands_visible_count = 0
                    
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                                hands_visible_count += 1
                    
                    # If we found more landmarks than before, incorporate these results
                    if hands_visible_count > visible_landmarks_count:
                        visible_landmarks_count = hands_visible_count
                        
                        # Create a holistic result-like structure with these hand landmarks
                        if best_results is None:
                            with self.mp_holistic.Holistic(static_image_mode=False) as dummy_holistic:
                                best_results = dummy_holistic.process(np.zeros_like(image_rgb))
                        
                        # Use existing best_results but add hand landmarks
                        if len(hands_results.multi_hand_landmarks) >= 1:
                            best_results.right_hand_landmarks = hands_results.multi_hand_landmarks[0]
                        
                        if len(hands_results.multi_hand_landmarks) >= 2:
                            best_results.left_hand_landmarks = hands_results.multi_hand_landmarks[1]
        
        # Try with enhanced image if we still don't have enough landmarks
        if visible_landmarks_count < min_finger_landmarks * 2:
            # Apply skin mask to isolate hands
            enhanced_image = self.apply_skin_mask(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            with self.mp_holistic.Holistic(
                static_image_mode=False, 
                model_complexity=1,
                min_detection_confidence=0.6) as holistic:
                
                enhanced_results = holistic.process(enhanced_image_rgb)
                
                # Count detected hand landmarks
                enhanced_visible_count = 0
                
                if enhanced_results.left_hand_landmarks:
                    for landmark in enhanced_results.left_hand_landmarks.landmark:
                        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                            enhanced_visible_count += 1
                
                if enhanced_results.right_hand_landmarks:
                    for landmark in enhanced_results.right_hand_landmarks.landmark:
                        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                            enhanced_visible_count += 1
                
                if enhanced_visible_count > visible_landmarks_count:
                    visible_landmarks_count = enhanced_visible_count
                    best_results = enhanced_results
        
        return best_results
    
    def extract_landmarks(self, results):
        """Extract key landmarks from MediaPipe results"""
        landmarks = {
            'pose': [],
            'left_hand': [],
            'right_hand': []
        }
        # Extract pose landmarks (arms, chest, shoulders)
        if results.pose_landmarks:
            for idx in self.pose_indices:
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    landmarks['pose'].append({
                        'indices': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': landmark.visibility
                    })
        else:
            # Set neutral state for pose
            landmarks['pose'] = [{'indices': idx, 'x': 0.0, 'y': 0.0, 'visibility': 0.0} 
                                for idx in self.pose_indices]
        
        # Extract left hand landmarks - even if only partial
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                # Check if this landmark is valid
                is_valid = (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)
                
                if is_valid:
                    landmarks['left_hand'].append({
                        'indices': i,
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': 1.0
                    })
                else:
                    # For invalid landmarks, set with reduced visibility
                    landmarks['left_hand'].append({
                        'indices': i,
                        'x': 0.0,
                        'y': 0.0,
                        'visibility': 0.0
                    })
        else:
            # Set neutral state for left hand
            landmarks['left_hand'] = [{'indices': i, 'x': 0.0, 'y': 0.0, 'visibility': 0.0} 
                                     for i in range(21)]
        
        # Extract right hand landmarks - even if only partial
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                # Check if this landmark is valid
                is_valid = (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)
                
                if is_valid:
                    landmarks['right_hand'].append({
                        'indices': i+21,
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': 1.0
                    })
                else:
                    # For invalid landmarks, set with reduced visibility
                    landmarks['right_hand'].append({
                        'indices': i+21,
                        'x': 0.0,
                        'y': 0.0,
                        'visibility': 0.0
                    })
        else:
            # Set neutral state for right hand
            landmarks['right_hand'] = [{'indices': i+21, 'x': 0.0, 'y': 0.0, 'visibility': 0.0} 
                                      for i in range(21)]
        
        return landmarks
    
    def parse_frame(self, frame):
        """Parse a single frame of landmarks into a flat list of features"""
        keypoints = []
        for part in ['pose', 'left_hand', 'right_hand']:
            for landmark in frame.get(part, []):
                keypoints.extend([landmark['x'], landmark['y'],landmark['visibility']])
        return keypoints
    
    def process_frame(self, frame):
        """Process a single webcam frame"""
        # Enhance the image
        enhanced_frame = self.enhance_image(frame)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.process_with_multiple_approaches(frame_rgb)
        
        # Extract landmarks
        if results is not None:
            landmarks = self.extract_landmarks(results)
            
            # Add to buffer
            keypoints = self.parse_frame(landmarks)
            self.frame_buffer.append(keypoints)
            
            # Draw landmarks on frame
            self.draw_landmarks_on_image(frame, results)
            
            # Make prediction if we have enough frames
            if len(self.frame_buffer) == self.sequence_length and self.model is not None:
                self.predict()
        
        # Draw prediction on frame
        self.draw_prediction(frame)
        
        return frame
    
    def predict(self):
        """Make a prediction based on the current frame buffer"""
        if len(self.frame_buffer) < self.sequence_length or self.model is None:
            return
        
        # Convert buffer to tensor
        sequence = np.array(list(self.frame_buffer))
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        model_device = next(self.model.parameters()).device
        sequence_tensor = sequence_tensor.to(model_device)
        adj_tensor = self.adj_tensor.to(model_device)
        # Make prediction
        with torch.no_grad():
            logits = self.model(sequence_tensor, adj_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
            # Update prediction info
            
            self.current_prediction = f"{pred_class.item()}"
                
            self.prediction_confidence = confidence.item()
            self.prediction_time = time.time()
    
    def draw_landmarks_on_image(self, image, results):
        """Draw landmarks on image for visualization"""
        if results is None:
            return
        
        # Define drawing styles
        pose_landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        pose_connection_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        
        hand_landmark_style = self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
        hand_connection_style = self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style
            )
    
    def draw_prediction(self, image):
        """Draw prediction text on image"""
        h, w, _ = image.shape
        
        # Display buffer status
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
        cv2.putText(image, buffer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display prediction if available
        if self.current_prediction:
            # Calculate time since last prediction
            elapsed = time.time() - self.prediction_time
            
            # Display prediction for 3 seconds with fade-out
            if elapsed < 3.0:
                alpha = 1.0 - (elapsed / 3.0)
                
                # Draw prediction with confidence
                prediction_text = f"Sign: {self.label_map[int(self.current_prediction)]}"
                confidence_text = f"Confidence: {self.prediction_confidence:.2f}"
                
                # Create overlay for prediction
                overlay = image.copy()
                cv2.rectangle(overlay, (10, h - 90), (320, h - 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5 * alpha, image, 1 - 0.5 * alpha, 0, image)
                
                # Add text
                cv2.putText(image, prediction_text, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(image, confidence_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

class SignLanguagePreprocessor:
    def __init__(self, raw_data_dir, output_dir, static_frames=15, 
                 min_hand_confidence=0.5,min_confidence=0.5, force_process=True, debug_mode=True,
                 enhanced_detection=True, min_finger_landmarks=5,target_width=648,target_height=648):
        """
        Initialize the Sign Language Preprocessor
        
        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory to save processed data
            static_frames: Number of frames to generate for static images
            min_hand_confidence: Minimum confidence for hand landmarks
            force_process: If True, process all data even without hand detection
            debug_mode: If True, save visualization images for debugging
            enhanced_detection: If True, apply additional preprocessing to improve detection
            min_finger_landmarks: Minimum number of finger landmarks required (out of 21 total)
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.static_frames = static_frames
        self.min_hand_confidence = min_hand_confidence
        self.min_confidence = min_confidence
        self.force_process = force_process
        self.debug_mode = debug_mode
        self.enhanced_detection = enhanced_detection
        self.min_finger_landmarks = min_finger_landmarks
        self.target_width=target_width
        self.target_height=target_height
        self.adj_mat=create_adjacency_matrix()
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands  # Add direct hands model for fallback
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create debug directory if needed
        self.debug_dir = os.path.join(self.output_dir, "debug")
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)

    def parse_frame(self, frame):
        """Parse a single frame of landmarks into a flat list of features"""
        keypoints = []
        for part in ['pose', 'left_hand', 'right_hand']:
            for landmark in frame.get(part, []):
                keypoints.append([landmark['x'], landmark['y'],landmark['visibility']])
        return keypoints
    
    def save_plot(self,landmarks,path):
        keypoints = self.parse_frame(landmarks)
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(121)
        ax1.set_title('2D View of MediaPipe Landmarks')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(1.0, 0) 
        ax1.grid(True, alpha=0.3)
        x = [coord[0] for coord in keypoints]
        y = [coord[1] for coord in keypoints]
        v = [coord[2] for coord in keypoints]
        num_nodes = len(keypoints)
        for i in range(num_nodes):
            if v[i] > 0:
                ax1.scatter(x[i],y[i], color='blue', s=30, alpha=0.7,label='Keypoints')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Only plot the upper triangle to avoid duplicates
                if self.adj_mat[i, j] > 0 and v[i] >0 and v[j] > 0:  # There is a connection between node i and node j
                    ax1.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.5)
        plt.savefig(path, dpi=300, bbox_inches='tight')

        plt.clf()
        plt.close()
        del ax1,x,y,num_nodes,fig,keypoints
        gc.collect()

    def resize_image(self,image):
        # Get the original image dimensions
        h, w = image.shape[:2]
        
        # Calculate the aspect ratio
        aspect_ratio = w / h
        
        # Resize the image while maintaining the aspect ratio
        if aspect_ratio > 1:
            new_w = self.target_width
            new_h = int(self.target_width / aspect_ratio)
        else:
            new_h = self.target_height
            new_w = int(self.target_height * aspect_ratio)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Add padding if necessary
        top_padding = (self.target_height - new_h) // 2
        bottom_padding = self.target_height - new_h - top_padding
        left_padding = (self.target_width - new_w) // 2
        right_padding = self.target_width - new_w - left_padding
        
        # Add padding to make the image the desired size
        padded_image = cv2.copyMakeBorder(
            resized_image, top_padding, bottom_padding, left_padding, right_padding, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        return padded_image


    def process_all_data(self):
        """Process all data in the raw data directory"""
        print("Starting preprocessing...")
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.raw_data_dir) 
                     if os.path.isdir(os.path.join(self.raw_data_dir, d))]
        
        # Initialize counters for reporting
        total_files = 0
        processed_files = 0
        skipped_files = 0
        
        for class_dir in class_dirs:
            print(f"Processing class: {class_dir}")
            class_path = os.path.join(self.raw_data_dir, class_dir)
            output_class_path = os.path.join(self.output_dir, class_dir)
            os.makedirs(output_class_path, exist_ok=True)
            
            # Create debug directory for this class if needed
            if self.debug_mode:
                class_debug_dir = os.path.join(self.debug_dir, class_dir)
                os.makedirs(class_debug_dir, exist_ok=True)
            
            # Get all files in the class directory
            files = glob.glob(os.path.join(class_path, '*.*'))
            total_files += len(files)
            
            for file_path in tqdm(files, desc=f"Processing {class_dir} files"):
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    success = self.process_image(file_path, output_class_path)
                elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    success = self.process_video(file_path, output_class_path)
                else:
                    success = False
                    
                if success:
                    processed_files += 1
                else:
                    skipped_files += 1
            gc.collect()
            
        
        print(f"\nPreprocessing Summary:")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {processed_files}")
        print(f"Skipped files: {skipped_files}")
    
    def enhance_image(self, image):
        """
        Apply image enhancement techniques to improve hand detection
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        if not self.enhanced_detection:
            return image

        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(image, (5, 5), 0)
        
        return enhanced
    
    def detect_skin(self, image):
        """
        Create a skin mask to help with hand segmentation
        
        Args:
            image: Input image
            
        Returns:
            Mask of skin regions
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Second range for skin detection
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def apply_skin_mask(self, image):
        """
        Apply skin mask to isolate hands in the image
        
        Args:
            image: Input image
            
        Returns:
            Image with background dimmed
        """
        if not self.enhanced_detection:
            return image
        
        # Get skin mask
        skin_mask = self.detect_skin(image)
        
        # Dilate mask slightly to ensure hands are fully covered
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        # Create a dimmed version of the original image for background
        dimmed = cv2.addWeighted(image, 0.3, np.zeros_like(image), 0.7, 0)
        
        # Create output image
        result = dimmed.copy()
        
        # Copy the original image where mask is set
        result[dilated_mask > 0] = image[dilated_mask > 0]
        
        return result
    
    def has_enough_finger_landmarks(self, hand_landmarks):
        """
        Check if enough finger landmarks are visible in the hand
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Boolean indicating if enough finger landmarks are detected
        """
        if hand_landmarks is None:
            return False
        
        # Count visible landmarks
        # Use x,y positions to determine if a landmark is set (valid)
        visible_count = 0
        for landmark in hand_landmarks.landmark:
            # Consider a landmark valid if it has reasonable x,y coordinates
            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                visible_count += 1
        
        return visible_count >= self.min_finger_landmarks
    
    def process_with_multiple_approaches(self, image_rgb):
        """
        Process an image with multiple detection approaches and use the best result
        
        Args:
            image_rgb: RGB image to process
            
        Returns:
            Best results from any of the detection methods
        """

        best_results = None
        visible_landmarks_count = 0
        
        # Try with holistic model first
        with self.mp_holistic.Holistic(
            static_image_mode=True, 
            model_complexity=2,
            min_detection_confidence=self.min_confidence,) as holistic:
            
            holistic_results = holistic.process(image_rgb)
            
            # Count detected hand landmarks
            holistic_visible_count = 0
            
            if holistic_results.left_hand_landmarks:
                for landmark in holistic_results.left_hand_landmarks.landmark:
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        holistic_visible_count += 1
            
            if holistic_results.right_hand_landmarks:
                for landmark in holistic_results.right_hand_landmarks.landmark:
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        holistic_visible_count += 1
            
            if holistic_visible_count > visible_landmarks_count:
                visible_landmarks_count = holistic_visible_count
                best_results = holistic_results
            else:
                del holistic_results
            gc.collect()
        
        # Try the dedicated hands model if we don't have enough landmarks
        if visible_landmarks_count < self.min_finger_landmarks * 2:  # Try to get more landmarks
            with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=self.min_hand_confidence) as hands:  # Lower threshold to catch partial hands
                
                hands_results = hands.process(image_rgb)
                
                # If hands were detected
                if hands_results.multi_hand_landmarks:
                    hands_visible_count = 0
                    
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                                hands_visible_count += 1
                    
                    # If we found more landmarks than before, incorporate these results
                    if hands_visible_count > visible_landmarks_count:
                        visible_landmarks_count = hands_visible_count
                        
                        # Create a holistic result-like structure with these hand landmarks
                        if best_results is None:
                            with self.mp_holistic.Holistic(static_image_mode=True) as dummy_holistic:
                                best_results = dummy_holistic.process(np.zeros_like(image_rgb))

                        
                        # Use existing best_results but add hand landmarks
                        if len(hands_results.multi_hand_landmarks) >= 1:
                            best_results.right_hand_landmarks = hands_results.multi_hand_landmarks[0]
                        
                        if len(hands_results.multi_hand_landmarks) >= 2:
                            best_results.left_hand_landmarks = hands_results.multi_hand_landmarks[1]
                del hands_results
            gc.collect()

        
        # Try with enhanced image if we still don't have enough landmarks
        if visible_landmarks_count < self.min_finger_landmarks * 2 and self.enhanced_detection:
            # Apply skin mask to isolate hands
            enhanced_image = self.apply_skin_mask(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
            
            with self.mp_holistic.Holistic(
                static_image_mode=True, 
                model_complexity=2,
                min_detection_confidence=self.min_hand_confidence) as holistic:  # Lower threshold for enhanced image
                
                enhanced_results = holistic.process(enhanced_image_rgb)
                
                # Count detected hand landmarks
                enhanced_visible_count = 0
                
                if enhanced_results.left_hand_landmarks:
                    for landmark in enhanced_results.left_hand_landmarks.landmark:
                        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                            enhanced_visible_count += 1
                
                if enhanced_results.right_hand_landmarks:
                    for landmark in enhanced_results.right_hand_landmarks.landmark:
                        if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                            enhanced_visible_count += 1
                
                if enhanced_visible_count > visible_landmarks_count:
                    visible_landmarks_count = enhanced_visible_count
                    best_results = enhanced_results
                else:
                    del enhanced_results

            del enhanced_image, enhanced_image_rgb
            gc.collect()
        
        return best_results
    
    def process_image(self, image_path, output_dir):
        """
        Process a single image file
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save the processed data
            
        Returns:
            Boolean indicating if processing was successful
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_landmarks.json")
        
        # Skip if already processed
        if os.path.exists(output_file):
            print("Data already exists. Skipping . . .")
            return True
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return False
        height, width, channels = image.shape
        if (height != self.target_height) or (width != self.target_width):
            image = self.resize_image(image)
        # Create debug images directory
        if self.debug_mode:
            debug_class_dir = os.path.join(self.debug_dir, os.path.basename(output_dir))
            os.makedirs(debug_class_dir, exist_ok=True)
        
        # Apply enhancement
        enhanced_image = self.enhance_image(image)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        if (height != self.target_height) or (width != self.target_width):
            image_rgb = self.resize_image(image_rgb)
        # Create sequence data with subtle noise
        sequence_data = []
        
        # Process with multiple approaches
        base_results = self.process_with_multiple_approaches(image_rgb)
        # Create debug image
        # if self.debug_mode and base_results:
        if self.debug_mode and base_results:
            debug_image = image.copy()
            print("Attempting to draw image")
            self.draw_landmarks_on_image(debug_image, base_results)
            cv2.imwrite(os.path.join(debug_class_dir, f"{base_name}_landmarks.jpg"), debug_image)
            del debug_image
        
        # Check if we have sufficient finger landmarks (or if we're forcing processing)
        left_has_enough = (base_results and self.has_enough_finger_landmarks(base_results.left_hand_landmarks))
        right_has_enough = (base_results and self.has_enough_finger_landmarks(base_results.right_hand_landmarks))
        
        if (not left_has_enough and not right_has_enough and not self.force_process):
            print(f"Not enough finger landmarks detected in {image_path}, skipping")
            return False
        
        if base_results == None:
            return False
        
        # Extract base landmarks
        base_landmarks = self.extract_landmarks(base_results)
        
        if self.debug_mode and base_landmarks:
            savepath = os.path.join(debug_class_dir, f"{base_name}_plot.jpg")
            self.save_plot(base_landmarks,savepath)

        # Generate frames with subtle noise
        for i in range(self.static_frames):
            # Add subtle noise to create variation between frames
            frame_landmarks = self.add_subtle_noise(base_landmarks)
            sequence_data.append(frame_landmarks)
        
        # Save the sequence data
        with open(output_file, 'w') as f:
            json.dump(sequence_data, f)
        del image,image_rgb,enhanced_image
        del sequence_data
        gc.collect()
        return True
    
    def process_video(self, video_path, output_dir):
        """
        Process a single video file
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the processed data
            
        Returns:
            Boolean indicating if processing was successful
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_landmarks.json")
        
        # Skip if already processed
        if os.path.exists(output_file):
            print("Data already exists. Skipping . . .")
            return True
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return False
        
        sequence_data = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        debug_frames = []
        
        # Create debug directory
        if self.debug_mode:
            debug_class_dir = os.path.join(self.debug_dir, os.path.basename(output_dir))
            os.makedirs(debug_class_dir, exist_ok=True)
        
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                height, width, channels = image.shape
                if (height != self.target_height) or (width != self.target_width):
                    image = self.resize_image(image)
                # Process every nth frame to improve speed for long videos
                if total_frames > 300 and frame_count % 2 != 0:
                    frame_count += 1
                    continue
                
                # Apply enhancement
                enhanced_image = self.enhance_image(image)
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
                height, width, channels = image_rgb.shape
                if (height != self.target_height) or (width != self.target_width):
                    image_rgb = self.resize_image(image_rgb)
                # Process the frame
                results = holistic.process(image_rgb)
                
                # If no hands detected with holistic or not enough landmarks, try the dedicated hands model
                left_has_enough = self.has_enough_finger_landmarks(results.left_hand_landmarks)
                right_has_enough = self.has_enough_finger_landmarks(results.right_hand_landmarks)
                
                if (not left_has_enough or not right_has_enough):
                    with self.mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=self.min_hand_confidence) as hands:  # Lower threshold to catch partial hands
                        
                        hands_results = hands.process(image_rgb)
                        
                        # If hands were detected
                        if hands_results.multi_hand_landmarks:
                            if len(hands_results.multi_hand_landmarks) >= 1:
                                # If the first model didn't find enough landmarks but this one does
                                if not right_has_enough and self.has_enough_finger_landmarks(hands_results.multi_hand_landmarks[0]):
                                    results.right_hand_landmarks = hands_results.multi_hand_landmarks[0]
                            
                            if len(hands_results.multi_hand_landmarks) >= 2:
                                if not left_has_enough and self.has_enough_finger_landmarks(hands_results.multi_hand_landmarks[1]):
                                    results.left_hand_landmarks = hands_results.multi_hand_landmarks[1]
                
                # Save debug frame if in debug mode (but only a few frames)
                if self.debug_mode and frame_count % 10 == 0 and len(debug_frames) < 5:
                    debug_frame = image.copy()
                    self.draw_landmarks_on_image(debug_frame, results)
                    debug_frames.append((frame_count, debug_frame))
                
                # Check if we have any hand landmarks
                left_has_enough = self.has_enough_finger_landmarks(results.left_hand_landmarks)
                right_has_enough = self.has_enough_finger_landmarks(results.right_hand_landmarks)
                
                # Skip frame if we don't have enough landmarks and not forcing process
                if not left_has_enough and not right_has_enough and not self.force_process:
                    frame_count += 1
                    continue
                
                # Extract landmarks
                frame_landmarks = self.extract_landmarks(results)
                sequence_data.append(frame_landmarks)
                
                frame_count += 1
                del image,image_rgb,enhanced_image
                gc.collect()
                
            cap.release()
        
        # Save debug frames if any
        if self.debug_mode and debug_frames:
            for frame_num, debug_frame in debug_frames:
                debug_path = os.path.join(debug_class_dir, f"{base_name}_frame{frame_num}_debug.jpg")
                cv2.imwrite(debug_path, debug_frame)
        
        # Save the sequence data if we have any frames
        if sequence_data:
            with open(output_file, 'w') as f:
                json.dump(sequence_data, f)
            del sequence_data
            return True
        else:
            print(f"No valid frames with sufficient hand landmarks found in {video_path}")
            return False
    
    def extract_landmarks(self, results):
        """
        Extract key landmarks from MediaPipe results
        
        Args:
            results: MediaPipe Holistic results
            
        Returns:
            Dictionary containing landmarks for face, pose, left hand, and right hand
        """
        landmarks = {
            'pose': [],
            'left_hand': [],
            'right_hand': []
        }
        
        # Extract pose landmarks (arms, chest, shoulders)
        # Selected landmarks for upper body
        pose_indices = [
            # Faces
            0,1,2,3,4,5,6,7,8,9,10,
            # Shoulders
            11, 12,
            # Arms
            13, 14, 15, 16,
            # Chest
            23, 24
        ]
        if results.pose_landmarks:
            
            for idx in pose_indices:
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    landmarks['pose'].append({
                        'indices': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': landmark.visibility
                    })
        else:
            # Set neutral state for pose
            landmarks['pose'] = [{'indices':indices,'x': 0.0, 'y': 0.0, 'visibility': 0.0} for indices in pose_indices]
        
        # Extract left hand landmarks - even if only partial
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                # Check if this landmark is valid
                is_valid = (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)
                
                if is_valid:
                    landmarks['left_hand'].append({
                        'indices':i,
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': 1.0
                    })
                else:
                    # For invalid landmarks, set with reduced visibility
                    landmarks['left_hand'].append({
                        'indices':i,
                        'x': 0.0,
                        'y': 0.0,
                        'visibility': 0.0
                    })
        else:
            # Set neutral state for left hand
            landmarks['left_hand'] = [{'indices':i,'x': 0.0, 'y': 0.0, 'visibility': 0.0} for i in range(21)]
        
        # Extract right hand landmarks - even if only partial
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                # Check if this landmark is valid
                is_valid = (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)
                
                if is_valid:
                    landmarks['right_hand'].append({
                        'indices':i,
                        'x': landmark.x,
                        'y': landmark.y,
                        'visibility': 1.0
                    })
                else:
                    # For invalid landmarks, set with reduced visibility
                    landmarks['right_hand'].append({
                        'indices':i,
                        'x': 0.0,
                        'y': 0.0,
                        'visibility': 0.0
                    })
        else:
            # Set neutral state for right hand
            landmarks['right_hand'] = [{'indices':i,'x': 0.0, 'y': 0.0, 'visibility': 0.0} for i in range(21)]
        
        return landmarks
    
    def add_subtle_noise(self, base_landmarks, noise_factor=0.002):
        """
        Add subtle noise to landmarks to create variation between frames
        
        Args:
            base_landmarks: Base landmarks to add noise to
            noise_factor: Factor to control the amount of noise
            
        Returns:
            New landmarks with added noise
        """
        noisy_landmarks = {
            'pose': [],
            'left_hand': [],
            'right_hand': []
        }
        
        for key in base_landmarks:
            for landmark in base_landmarks[key]:
                # Skip adding noise to landmarks with zero visibility
                if landmark['visibility'] < 0.1:
                    noisy_landmarks[key].append(landmark.copy())
                    continue
                
                # Add noise to x, y, z coordinates
                noise_x = random.uniform(-noise_factor, noise_factor)
                noise_y = random.uniform(-noise_factor, noise_factor)
                noise_z = random.uniform(-noise_factor, noise_factor)
                
                noisy_landmarks[key].append({
                    'indices': landmark['indices'],
                    'x': landmark['x'] + noise_x,
                    'y': landmark['y'] + noise_y,
                    'visibility': landmark['visibility']
                })
        
        return noisy_landmarks
    
    def draw_landmarks_on_image(self, image, results):
        """
        Draw landmarks on image for debugging
        
        Args:
            image: Image to draw on
            results: MediaPipe results
        """
        if results is None:
            return

        pose_landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
        pose_connection_style = self.mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=4)

        hand_landmark_style = self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=6, circle_radius=3)
        hand_connection_style = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)

        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_style,
                connection_drawing_spec=hand_connection_style
            )


def parse_frame(frame):
    """Parse a single frame of landmarks into a flat list of features"""
    keypoints = []
    for part in ['pose', 'left_hand', 'right_hand']:
        for landmark in frame.get(part, []):
            keypoints.append([landmark['x'], landmark['y'],landmark['visibility']])
    return keypoints

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess the JSON files into sequences.
    
    Args:
        data_dir: Directory containing the data
        sequence_length: Number of frames in each sequence
        
    Returns:
        sequences: array of shape (num_sequences, sequence_length, num_nodes * features)
        sequence_labels: array of class labels
        label_encoder: fitted LabelEncoder
    """
    frame_data = []
    raw_labels = []
    
    # Step 1: Collect all labels
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                label = os.path.basename(os.path.dirname(os.path.join(root, file)))
                raw_labels.append(label)
    
    # Step 2: Fit label encoder
    encoder = LabelEncoder()
    encoder.fit(raw_labels)
    label_map = {label: int(encoder.transform([label])[0]) for label in set(raw_labels)}
    
    sequences = []
    sequence_labels = []
    # Step 3: Parse frames and assign encoded labels
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                label = os.path.basename(os.path.dirname(os.path.join(root, file)))
                encoded_label = label_map[label]
                with open(os.path.join(root, file), 'r') as f:
                    frames = json.load(f)
                    features=[]
                    for frame in frames:
                        features.append(parse_frame(frame))
                    sequences.append(np.stack(features))
                    sequence_labels.append(encoded_label)
    idx_to_label = {v: k for k, v in label_map.items()}
    label_map = idx_to_label
    del idx_to_label
    gc.collect()
    
    return np.array(sequences), np.array(sequence_labels), label_map