import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pickle
import argparse
from collections import deque
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Constants
NUM_POSE_LANDMARKS = 19   
NUM_HAND_LANDMARKS = 21
NUM_NODES = NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS*2  # Total nodes in the graph
FEATURE_DIM = 3  # x, y coordinates + visibility
# GCN Layer definition from the notebook
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features,dropout=0.5):
        super(GCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (batch_size * seq_len, num_nodes, in_features)
        # adj: (num_nodes, num_nodes) - shared across all samples
        
        # Ensure adj is on the same device as x
        if adj.device != x.device:
            adj = adj.to(x.device)
            
        # Expand adj to match batch dimension
        batch_size_seq = x.size(0)
        adj_expanded = adj.unsqueeze(0).expand(batch_size_seq, -1, -1)
        
        # Graph convolution: first aggregate neighborhood features
        x = torch.bmm(adj_expanded, x)  # (batch_size * seq_len, num_nodes, in_features)
        
        # For Conv1d: input needs to be (batch, channels, length)
        # So we permute from (batch, nodes, features) to (batch, features, nodes)
        x = x.permute(0, 2, 1)  # -> (batch_size * seq_len, in_features, num_nodes)
        
        # Apply convolution
        x = self.conv(x)  # (batch_size * seq_len, out_features, num_nodes)
        
        # Permute back to original format
        x = x.permute(0, 2, 1)  # -> (batch_size * seq_len, num_nodes, out_features)
        
        x = F.gelu(x)
        x = self.dropout(x)
        return x

class GCNBiLSTM(nn.Module):
    def __init__(self, num_nodes=NUM_NODES, in_features=FEATURE_DIM, 
                 gcn_hidden=64, lstm_hidden=128, num_classes=10, 
                 num_gcn_layers=2, dropout=0.5, label_map=None):
        super(GCNBiLSTM, self).__init__()
        
        # Create multiple GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(in_features, gcn_hidden,dropout))
        
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNLayer(gcn_hidden, gcn_hidden,dropout))
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_nodes * gcn_hidden, 
            hidden_size=lstm_hidden, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_gcn_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.label_map = label_map
        self.num_nodes = num_nodes
        self.gcn_hidden = gcn_hidden

    def forward(self, x, adj):
        # x shape: (batch_size, seq_len, num_nodes * in_features)
        # Reshape to (batch_size, seq_len, num_nodes, in_features)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_nodes, -1)
        
        # Process each time step through GCN
        gcn_outputs = []
        for t in range(seq_len):
            # Get current time step data
            curr_x = x[:, t, :, :]  # (batch_size, num_nodes, in_features)
            
            # Process through GCN layers
            for gcn_layer in self.gcn_layers:
                curr_x = gcn_layer(curr_x, adj)
                curr_x = self.dropout(curr_x)
            
            # Flatten node features
            # curr_x = curr_x.view(batch_size, -1)  # (batch_size, num_nodes * gcn_hidden)
            curr_x = curr_x.contiguous().view(batch_size, -1)
            gcn_outputs.append(curr_x)
        
        # Stack outputs to (batch_size, seq_len, num_nodes * gcn_hidden)
        gcn_out = torch.stack(gcn_outputs, dim=1)
        
        # Process through BiLSTM
        lstm_out, _ = self.lstm(gcn_out)  # (batch_size, seq_len, lstm_hidden * 2)
        
        # Apply attention mechanism
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, lstm_hidden * 2)
        
        # Final classification
        output = self.classifier(context)
        
        return output
    
    def predict_label(self, x, adj):
        self.eval() 
        with torch.no_grad():
            logits = self.forward(x, adj)  # Forward pass
            pred_classes = torch.argmax(logits, dim=1)  # Get the predicted class (index)
            
            if self.label_map is not None:
                pred_labels = [self.label_map[int(idx)] for idx in pred_classes.cpu().numpy()]
                return pred_labels
            else:
                return pred_classes


# Create the normalized adjacency matrix
def create_norm_adjacency_matrix():
    """Create the adjacency matrix for the graph."""

    pose_connections = [
        # Mouth
        (9,10),
        # Left Eyes
        (1,2),(2,3),(3,7),
        # Right Eyes
        (4,5),(5,6),(6,8),
        # Nose
        (0,4),(0,1),
        # Shoulders
        (11, 12),
        # Connect shoulders to hip
        (11, 17), (12, 18),
        # Connect hip points
        (17, 18),
        # Left arm
        (11, 13), (13, 15),
        # Right arm
        (12, 14), (14, 16)
    ]

    hand_connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]
    
    def create_adj_matrix(num_nodes, connections):
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, j in connections:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        # Add self-loops
        for i in range(num_nodes):
            adj_matrix[i, i] = 1
        return adj_matrix

    pose_adj_matrix = create_adj_matrix(NUM_POSE_LANDMARKS, pose_connections)
    left_hand_adj_matrix = create_adj_matrix(NUM_HAND_LANDMARKS, hand_connections)
    right_hand_adj_matrix = create_adj_matrix(NUM_HAND_LANDMARKS, hand_connections)

    # Calculate the total number of nodes
    total_nodes = NUM_POSE_LANDMARKS + NUM_HAND_LANDMARKS + NUM_HAND_LANDMARKS

    # Initialize a global adjacency matrix
    global_adj_matrix = np.zeros((total_nodes, total_nodes))
    
    # start_pose = NUM_FACE_LANDMARKS
    start_pose=0
    end_pose = start_pose + NUM_POSE_LANDMARKS
    global_adj_matrix[start_pose:end_pose, start_pose:end_pose] = pose_adj_matrix
    
    start_lh = end_pose
    end_lh = start_lh + NUM_HAND_LANDMARKS
    global_adj_matrix[start_lh:end_lh, start_lh:end_lh] = left_hand_adj_matrix
    
    start_rh = end_lh
    end_rh = start_rh + NUM_HAND_LANDMARKS
    global_adj_matrix[start_rh:end_rh, start_rh:end_rh] = right_hand_adj_matrix
    
    # Connect pose to hands
    pose_hand_connections = [
        (start_pose + 15, start_lh),  # Left hand wrist to left hand base
        (start_pose + 16, start_rh),  # Right hand wrist to right hand base
    ]
    for i, j in pose_hand_connections:
        global_adj_matrix[i, j] = 1
        global_adj_matrix[j, i] = 1

    # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
    # Add identity matrix to include self-connections
    adj_matrix = global_adj_matrix + np.eye(total_nodes)
    
    # Calculate degree matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    
    # D^-0.5
    deg_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    
    # Normalized adjacency matrix
    normalized_adj_matrix = deg_inv_sqrt @ adj_matrix @ deg_inv_sqrt

    return torch.FloatTensor(normalized_adj_matrix)

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Sign Language Recognition')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--sequence_length', type=int, default=15, 
                        help='Number of frames to process for prediction (default: 15)')
    
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    args = parser.parse_args()
    
    # Initialize the processor
    processor = RealtimeSignLanguageProcessor(
        model_path=args.model,
        sequence_length=args.sequence_length
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Starting real-time sign language recognition...")
    if args.model:
        print(f"Model loaded from: {args.model}")
    else:
        print("No model loaded. Running in visualization-only mode.")
    
    # Calculate target frame interval
    target_interval = 1.0 / args.fps
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera.")
                break
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = processor.process_frame(frame)
            
            # Display FPS
            processing_time = time.time() - start_time
            fps = 1.0 / max(processing_time, 0.001)  # Avoid division by zero
            cv2.putText(
                processed_frame, 
                f"FPS: {fps:.1f}", 
                (processed_frame.shape[1] - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Display instructions
            cv2.putText(
                processed_frame,
                "Press 'q' to quit",
                (10, processed_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show the frame
            cv2.imshow('Real-time Sign Language Recognition', processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")



if __name__ == "__main__":
    main()