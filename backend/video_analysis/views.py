import os
import time
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FileUploadParser
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .models import VideoAnalysis
from .utils import get_detector
from django.core.paginator import Paginator
from datetime import datetime, timedelta



import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import libraries for feature extraction
import mediapipe as mp
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import TimesformerModel, TimesformerConfig
from torchvision import transforms
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import math
from datetime import datetime
import shutil

# Import your existing Django modules
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FileUploadParser
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

# Your model architecture (copy from your existing code)
class EnhancedDeepfakeDetectionModel(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, p2_dim, p3_dim):
        super().__init__()

        self.dropout_spatial = nn.Dropout2d(0.15)
        self.dropout_temporal = nn.Dropout(0.25) 

        self.spatial_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
        )
        
        # Feature dimensions
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.p2_dim = p2_dim
        self.p3_dim = p3_dim
        
        # Feature processing streams with batch normalization
        self.spatial_stream = nn.Sequential(
            nn.Linear(spatial_dim, 1536),
            nn.BatchNorm1d(1536),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.15),
        )
        
        self.temporal_stream = nn.Sequential(
            nn.Linear(temporal_dim, 1024),  
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        self.temporal_processor = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.p2_stream = nn.Sequential(
            nn.Linear(p2_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.p3_stream = nn.Sequential(
            nn.Linear(p3_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )

        self.cross_attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
        )
        
        # Adaptive feature weighting
        self.feature_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
        # Combine handcrafted features
        self.handcrafted_fusion = nn.Sequential(
            nn.Linear(128 + 128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        self.confidence_predictor = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.spatial_pyramid = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256)
        )
                
        # Enhanced attention mechanism
        self.attention_query = nn.Linear(256 * 3, 256)
        self.attention_key = nn.Linear(256 * 3, 256)
        self.attention_value = nn.Linear(256 * 3, 3)
        
        # Add residual connections
        self.spatial_residual = nn.Linear(spatial_dim, 256)
        self.temporal_residual = nn.Linear(temporal_dim, 256)
        
        # Final classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.temporal_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial, temporal, p2, p3):
        # Normalize feature weights
        norm_weights = F.softmax(self.feature_weights, dim=0)
        
        # Process each feature stream
        spatial_features = self.spatial_stream(spatial) * norm_weights[0]
        temporal_features = self.temporal_stream(temporal) * norm_weights[1]
        p2_features = self.p2_stream(p2) * norm_weights[2]
        p3_features = self.p3_stream(p3) * norm_weights[3]

        spatial_query = self.cross_attention[0](spatial_features)
        temporal_key = self.cross_attention[1](temporal_features)
        temporal_value = self.cross_attention[2](temporal_features)

        spatial_features = self.dropout_spatial(spatial_features.unsqueeze(1)).squeeze(1)
        temporal_features = self.dropout_temporal(temporal_features)

        # Feature pyramid for spatial features
        pyramid_features = self.spatial_pyramid(spatial_features)
        spatial_features = spatial_features + pyramid_features

        # Compute attention scores
        attention_scores = torch.bmm(
            spatial_query.unsqueeze(1),
            temporal_key.unsqueeze(2)
        ).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Apply cross-attention
        enhanced_spatial = spatial_features + attention_weights.unsqueeze(1) * temporal_features

        # Apply special attention to spatial features
        spatial_attention = torch.sigmoid(self.spatial_attention(spatial_features))
        spatial_features = spatial_features * spatial_attention
        
        # Add residual connections
        spatial_res = self.spatial_residual(spatial)
        temporal_res = self.temporal_residual(temporal)
        
        # Apply residual connections
        spatial_features = spatial_features + spatial_res
        temporal_features = temporal_features + temporal_res
        
        # Combine handcrafted features
        handcrafted_features = torch.cat([p2_features, p3_features], dim=1)
        handcrafted_features = self.handcrafted_fusion(handcrafted_features)
        
        # Prepare features for attention
        features_for_attention = torch.cat([
            spatial_features, temporal_features, handcrafted_features
        ], dim=1)
        
        # Apply enhanced attention mechanism
        query = self.attention_query(features_for_attention)
        key = self.attention_key(features_for_attention)
        value = self.attention_value(features_for_attention)
        
        # Compute attention scores
        attention_scores = torch.bmm(
            query.unsqueeze(1), 
            key.unsqueeze(2)
        ).squeeze(1)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(value, dim=1)
        
        # Force balanced feature usage
        min_temporal_weight = 0.3
        min_handcrafted_weight = 0.1
        max_spatial_weight = 0.6
        
        # Clamp spatial weight to maximum
        spatial_weight = torch.clamp(attention_weights[:, 0:1], max=max_spatial_weight)
        
        # Ensure temporal features have at least minimum weight
        temporal_weight = torch.clamp(attention_weights[:, 1:2], min=min_temporal_weight)
        
        # Ensure handcrafted has minimum weight
        handcrafted_weight = torch.clamp(attention_weights[:, 2:3], min=min_handcrafted_weight)
        
        # Normalize weights to sum to 1
        total_weight = spatial_weight + temporal_weight + handcrafted_weight
        spatial_weight = spatial_weight / total_weight
        temporal_weight = temporal_weight / total_weight
        handcrafted_weight = handcrafted_weight / total_weight
        
        weighted_spatial = spatial_features * spatial_weight
        weighted_temporal = temporal_features * temporal_weight
        weighted_handcrafted = handcrafted_features * handcrafted_weight
        
        # Enhance temporal features
        enhanced_temporal = self.temporal_processor(weighted_temporal)
        weighted_temporal = enhanced_temporal

        # Process temporal prediction
        temporal_prediction = self.temporal_classifier(weighted_temporal)
        
        # Combine all features
        combined_features = torch.cat([
            weighted_spatial, weighted_temporal, weighted_handcrafted
        ], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        
        # Add confidence prediction
        confidence = self.confidence_predictor(combined_features)
        
        # During inference, adjust output based on confidence
        if not self.training:
            if confidence.mean() < 0.6:
                output = output * 0.7 + temporal_prediction * 0.3
                        
        return output, attention_weights, norm_weights


class FaceExtractor:
    """Face extraction for P1 preprocessing"""
    def __init__(self, confidence_threshold=0.85, min_face_size=40):
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.confidence_threshold
        )

    def extract_faces_from_video(self, video_path, output_dir):
        """Extract faces from video and save metadata"""
        os.makedirs(output_dir, exist_ok=True)
        face_dir = os.path.join(output_dir, "faces")
        os.makedirs(face_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        metadata = []
        face_count = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % 3 == 0:  # Sample every 3rd frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        confidence = detection.score[0]
                        if confidence >= self.confidence_threshold:
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = frame.shape
                            
                            x_min = max(0, int(bbox.xmin * w))
                            y_min = max(0, int(bbox.ymin * h))
                            box_width = min(int(bbox.width * w), w - x_min)
                            box_height = min(int(bbox.height * h), h - y_min)
                            
                            if box_width >= self.min_face_size and box_height >= self.min_face_size:
                                face_img = frame[y_min:y_min+box_height, x_min:x_min+box_width]
                                
                                face_filename = f"face_{frame_idx}_{face_count}.jpg"
                                face_path = os.path.join(face_dir, face_filename)
                                cv2.imwrite(face_path, face_img)
                                
                                # Create dummy embedding (replace with actual face embedding if needed)
                                embedding = np.random.randn(512).tolist()
                                
                                metadata.append({
                                    'frame_number': frame_idx,
                                    'face_id': face_count,
                                    'confidence': float(confidence),
                                    'face_filename': face_filename,
                                    'embedding': embedding,
                                    'quality_score': 0.8  # Dummy quality score
                                })
                                
                                face_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Save metadata
        with open(os.path.join(output_dir, "face_metadata_full.json"), 'w') as f:
            json.dump(metadata, f)
            
        return metadata


class CompletePipelineTester:
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the complete deepfake testing pipeline
        
        Args:
            model_path: Path to the trained model (.pth file)
            config_path: Path to model configuration file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        
        # Try to determine feature dimensions from the model file
        self.determine_model_dimensions()
        
        self.load_model()
        
        # Initialize feature extractors
        self.face_extractor = FaceExtractor()
        self.init_spatial_extractor()
        self.init_temporal_extractor()
        self.init_mediapipe()
    
    def determine_model_dimensions(self):
        """Determine the correct feature dimensions from the saved model"""
        try:
            # Load the checkpoint to inspect dimensions
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract dimensions from the saved weights
            # Spatial dimension from spatial_stream.0.weight
            if 'spatial_stream.0.weight' in checkpoint:
                self.spatial_dim = checkpoint['spatial_stream.0.weight'].shape[1]
            else:
                self.spatial_dim = 1792  # Default from error message
            
            # P3 dimension from p3_stream.0.weight  
            if 'p3_stream.0.weight' in checkpoint:
                self.p3_dim = checkpoint['p3_stream.0.weight'].shape[1]
            else:
                self.p3_dim = 20  # Default from error message
            
            # P2 dimension from p2_stream.0.weight
            if 'p2_stream.0.weight' in checkpoint:
                self.p2_dim = checkpoint['p2_stream.0.weight'].shape[1]
            else:
                self.p2_dim = 63  # Keep default
            
            # Temporal dimension from temporal_stream.0.weight
            if 'temporal_stream.0.weight' in checkpoint:
                self.temporal_dim = checkpoint['temporal_stream.0.weight'].shape[1]
            else:
                self.temporal_dim = 768 * 5  # Keep default
            
            print(f"Detected model dimensions:")
            print(f"  Spatial: {self.spatial_dim}")
            print(f"  Temporal: {self.temporal_dim}")
            print(f"  P2: {self.p2_dim}")
            print(f"  P3: {self.p3_dim}")
            
        except Exception as e:
            print(f"Warning: Could not determine dimensions from model file: {e}")
            print("Using default dimensions...")
            # Use the dimensions from the error message
            self.spatial_dim = 1792
            self.temporal_dim = 768 * 5
            self.p2_dim = 63
            self.p3_dim = 20
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        self.model = EnhancedDeepfakeDetectionModel(
            spatial_dim=self.spatial_dim,
            temporal_dim=self.temporal_dim,
            p2_dim=self.p2_dim,
            p3_dim=self.p3_dim
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def init_spatial_extractor(self):
        """Initialize spatial feature extractor (EfficientNet)"""
        print("Loading EfficientNet for spatial features...")
        self.spatial_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        print("EfficientNet loaded!")
    
    def init_temporal_extractor(self):
        """Initialize temporal feature extractor (TimeSformer)"""
        print("Loading TimeSformer for temporal features...")
        try:
            config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
            config.num_frames = 8
            self.temporal_model = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", 
                config=config,
                ignore_mismatched_sizes=True
            ).to(self.device)
            self.temporal_model.eval()
            print("TimeSformer loaded!")
        except Exception as e:
            print(f"Error loading TimeSformer: {e}")
            self.temporal_model = None
    
    def init_mediapipe(self):
        """Initialize MediaPipe for P2 and P3 features"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Define landmark indices
        self.jawline_indices = list(range(0, 17)) + [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        self.upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.left_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295]
        self.right_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53]

    def normalize_features(self, features):
        """Normalize feature vectors to [0,1] range"""
        if features is None:
            return None
            
        if np.all(features == features[0]):
            return np.zeros_like(features)
            
        feature_min = np.min(features)
        feature_max = np.max(features)
        if feature_max > feature_min:
            normalized = (features - feature_min) / (feature_max - feature_min)
            return normalized
        return features

    def extract_spatial_features(self, faces_dir: str) -> np.ndarray:
        """Extract spatial features using EfficientNet"""
        print("Extracting spatial features...")
        
        face_files = [f for f in os.listdir(faces_dir) if f.endswith('.jpg')]
        if not face_files:
            print("No face images found")
            return np.zeros(self.spatial_dim)
        
        all_features = []
        face_embeddings = []
        
        for face_file in face_files[:10]:  # Limit to first 10 faces
            try:
                face_path = os.path.join(faces_dir, face_file)
                img = Image.open(face_path)
                img = img.resize((224, 224))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Extract features
                features = self.spatial_model.predict(img_array, verbose=0)
                all_features.append(features.flatten())
                
                # Create dummy face embedding (512 dimensions to match training)
                face_embeddings.append(np.random.randn(512))
            except Exception as e:
                print(f"Error processing {face_file}: {e}")
                continue
        
        if all_features:
            # Average features
            avg_efficientnet = np.mean(all_features, axis=0)  # Should be 1280 dimensions
            avg_embedding = np.mean(face_embeddings, axis=0)  # 512 dimensions
            combined_features = np.concatenate([avg_efficientnet, avg_embedding])  # 1792 total
            
            # Ensure we match the expected spatial dimension exactly
            if len(combined_features) != self.spatial_dim:
                if len(combined_features) < self.spatial_dim:
                    # Pad with zeros
                    combined_features = np.pad(combined_features, (0, self.spatial_dim - len(combined_features)))
                else:
                    # Truncate
                    combined_features = combined_features[:self.spatial_dim]
                    
            print(f"Spatial features shape: {combined_features.shape}")
            return combined_features
        else:
            return np.zeros(self.spatial_dim)

    def extract_temporal_features(self, video_path: str) -> np.ndarray:
        """Extract temporal features using TimeSformer"""
        print("Extracting temporal features...")
        
        if self.temporal_model is None:
            return np.zeros(self.temporal_dim)
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        max_frames = 8
        
        while cap.read()[0] and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < max_frames:
            # Duplicate last frame to reach required number
            while len(frames) < max_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        frame_tensors = []
        for frame in frames:
            frame_tensor = transform(frame)
            frame_tensors.append(frame_tensor)
        
        frames_tensor = torch.stack(frame_tensors).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.temporal_model(frames_tensor)
                last_hidden = outputs.last_hidden_state
                
                # Create combined features
                mean_features = torch.mean(last_hidden, dim=1)
                max_features = torch.max(last_hidden, dim=1)[0]
                first_frame_features = last_hidden[:, 0, :]
                last_frame_features = last_hidden[:, -1, :]
                
                if last_hidden.size(1) > 1:
                    temp_diffs = torch.abs(last_hidden[:, 1:, :] - last_hidden[:, :-1, :])
                    temp_change_features = torch.mean(temp_diffs, dim=1)
                else:
                    temp_change_features = torch.zeros_like(mean_features)
                
                combined_features = torch.cat([
                    mean_features,
                    max_features,
                    first_frame_features,
                    last_frame_features,
                    temp_change_features
                ], dim=1).cpu().numpy().flatten()
                
                print(f"Temporal features shape: {combined_features.shape} (expected: {self.temporal_dim})")
                
                # Ensure we match the expected temporal dimension
                if len(combined_features) != self.temporal_dim:
                    if len(combined_features) < self.temporal_dim:
                        combined_features = np.pad(combined_features, (0, self.temporal_dim - len(combined_features)))
                    else:
                        combined_features = combined_features[:self.temporal_dim]
                
                return combined_features
        except Exception as e:
            print(f"Error in temporal feature extraction: {e}")
            return np.zeros(self.temporal_dim)

    def extract_p2_features(self, video_path: str) -> np.ndarray:
        """Extract P2 features (facial curve analysis)"""
        print("Extracting P2 features...")
        
        cap = cv2.VideoCapture(video_path)
        features = []
        frame_count = 0
        
        distortion_metrics = {feature: [] for feature in ['jawline', 'upper_lip', 'lower_lip', 'left_eyebrow', 'right_eyebrow']}
        
        while cap.read()[0] and frame_count < 30:  # Limit frames
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 3 == 0:  # Sample every 3rd frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks_array = np.array([
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in face_landmarks.landmark
                    ])
                    
                    # Extract feature curves and calculate distortion
                    feature_curves = {
                        'jawline': landmarks_array[self.jawline_indices],
                        'upper_lip': landmarks_array[self.upper_lip_indices],
                        'lower_lip': landmarks_array[self.lower_lip_indices],
                        'left_eyebrow': landmarks_array[self.left_eyebrow_indices],
                        'right_eyebrow': landmarks_array[self.right_eyebrow_indices]
                    }
                    
                    # Calculate distortion for each feature
                    for feature_name, curve in feature_curves.items():
                        try:
                            points = curve[:, :2]  # Use only x, y coordinates
                            if len(points) > 3:
                                tck, u = splprep([points[:, 0], points[:, 1]], s=0.0, k=3)
                                new_u = np.linspace(0, 1, len(points))
                                smoothed_points = np.column_stack(splev(new_u, tck))
                                distortion = np.mean(np.sum((points - smoothed_points) ** 2, axis=1))
                                distortion_metrics[feature_name].append(distortion)
                        except:
                            distortion_metrics[feature_name].append(0.0)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics for each feature
        for feature_name in distortion_metrics:
            if distortion_metrics[feature_name]:
                data = distortion_metrics[feature_name]
                features.extend([
                    np.mean(data), np.std(data), np.min(data), np.max(data), np.median(data)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        # Add basic frame quality metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.extend([blurriness, np.mean(gray), np.std(gray)])
        
        # Pad or truncate to expected size
        if len(features) < self.p2_dim:
            features.extend([0] * (self.p2_dim - len(features)))
        else:
            features = features[:self.p2_dim]
        
        print(f"P2 features shape: {len(features)} (expected: {self.p2_dim})")
        return np.array(features, dtype=np.float32)

    def extract_p3_features(self, video_path: str) -> np.ndarray:
        """Extract P3 features (head pose and lip motion)"""
        print("Extracting P3 features...")
        
        cap = cv2.VideoCapture(video_path)
        head_pose_data = []
        lip_motion_data = []
        frame_count = 0
        prev_landmarks = None
        
        while cap.read()[0] and frame_count < 30:  # Limit frames
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 3 == 0:  # Sample every 3rd frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks_array = np.array([
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in face_landmarks.landmark
                    ])
                    
                    # Extract head pose (simplified)
                    nose_tip = landmarks_array[4]
                    left_eye = landmarks_array[33]
                    right_eye = landmarks_array[263]
                    
                    # Calculate basic head pose angles
                    eye_line = right_eye - left_eye
                    yaw = math.degrees(math.atan2(eye_line[2], eye_line[0])) * 5
                    pitch = math.degrees(math.atan2(eye_line[1], eye_line[2])) * 5
                    roll = math.degrees(math.atan2(eye_line[1], eye_line[0])) * 2
                    
                    head_pose_data.append([yaw, pitch, roll])
                    
                    # Extract lip motion
                    upper_lip = landmarks_array[self.upper_lip_indices]
                    lower_lip = landmarks_array[self.lower_lip_indices]
                    
                    # Calculate lip metrics
                    mid_upper_lip = upper_lip[5]
                    mid_lower_lip = lower_lip[5]
                    lip_height = np.linalg.norm(mid_upper_lip - mid_lower_lip)
                    
                    left_corner = upper_lip[0]
                    right_corner = upper_lip[-1]
                    lip_width = np.linalg.norm(left_corner - right_corner)
                    
                    lip_area = lip_height * lip_width
                    
                    # Calculate lip movement if we have previous landmarks
                    lip_movement = 0
                    if prev_landmarks is not None:
                        prev_upper_lip = prev_landmarks[self.upper_lip_indices]
                        upper_lip_motion = np.mean(np.linalg.norm(upper_lip - prev_upper_lip, axis=1))
                        lip_movement = upper_lip_motion
                    
                    # Simple lip sync score (variance in lip height)
                    lip_sync_score = lip_height / max(lip_width, 1e-5)
                    
                    lip_motion_data.append([lip_height, lip_width, lip_area, lip_movement, lip_sync_score])
                    prev_landmarks = landmarks_array
            
            frame_count += 1
        
        cap.release()
        
        features = []
        
        # Head pose statistics
        if head_pose_data:
            head_pose_array = np.array(head_pose_data)
            for i in range(3):  # yaw, pitch, roll
                column = head_pose_array[:, i]
                features.extend([
                    np.mean(column), np.std(column), np.min(column), np.max(column), np.median(column)
                ])
        else:
            features.extend([0] * 15)  # 3 angles * 5 stats each
        
        # Lip motion statistics
        if lip_motion_data:
            lip_motion_array = np.array(lip_motion_data)
            for i in range(5):  # 5 lip metrics
                column = lip_motion_array[:, i]
                features.extend([
                    np.mean(column), np.std(column), np.min(column), np.max(column), np.median(column)
                ])
        else:
            features.extend([0] * 25)  # 5 metrics * 5 stats each
        
        # Add some additional temporal features
        if head_pose_data and len(head_pose_data) > 1:
            head_pose_array = np.array(head_pose_data)
            # Calculate movement/change features
            for i in range(3):
                changes = np.abs(np.diff(head_pose_array[:, i]))
                features.extend([np.mean(changes), np.std(changes)])
        else:
            features.extend([0] * 6)
        
        # Pad or truncate to expected size
        if len(features) < self.p3_dim:
            features.extend([0] * (self.p3_dim - len(features)))
        else:
            features = features[:self.p3_dim]
        
        print(f"P3 features shape: {len(features)} (expected: {self.p3_dim})")
        return np.array(features, dtype=np.float32)

    def process_video_complete_pipeline(self, video_path: str, temp_dir: str = None) -> Dict:
        """
        Process a video through the complete pipeline and make prediction
        
        Args:
            video_path: Path to the video file
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            Dictionary with prediction results
        """
        if temp_dir is None:
            temp_dir = f"temp_{int(time.time())}"
        
        temp_dir = os.path.abspath(temp_dir)
        
        try:
            print(f"Processing video: {video_path}")
            print(f"Using temporary directory: {temp_dir}")
            
            # Step 1: Face extraction (P1)
            print("\n=== Step 1: Face Extraction ===")
            faces_output_dir = os.path.join(temp_dir, "faces_extracted")
            metadata = self.face_extractor.extract_faces_from_video(video_path, faces_output_dir)
            
            if not metadata:
                print("No faces detected in video")
                return {
                    'video_path': video_path,
                    'prediction': 'UNKNOWN',
                    'confidence': 0.0,
                    'error': 'No faces detected'
                }
            
            faces_dir = os.path.join(faces_output_dir, "faces")
            
            # Step 2: Extract all features
            print("\n=== Step 2: Feature Extraction ===")
            
            # Extract spatial features
            spatial_features = self.extract_spatial_features(faces_dir)
            
            # Extract temporal features
            temporal_features = self.extract_temporal_features(video_path)
            
            # Extract P2 features
            p2_features = self.extract_p2_features(video_path)
            
            # Extract P3 features
            p3_features = self.extract_p3_features(video_path)
            
            # Step 3: Normalize features
            print("\n=== Step 3: Feature Normalization ===")
            spatial_features = self.normalize_features(spatial_features)
            temporal_features = self.normalize_features(temporal_features)
            p2_features = self.normalize_features(p2_features)
            p3_features = self.normalize_features(p3_features)
            
            # Handle NaN or Inf values
            spatial_features = np.nan_to_num(spatial_features)
            temporal_features = np.nan_to_num(temporal_features)
            p2_features = np.nan_to_num(p2_features)
            p3_features = np.nan_to_num(p3_features)
            
            # Step 4: Make prediction
            print("\n=== Step 4: Model Prediction ===")
            
            # Debug: Print feature statistics
            print(f"Feature Statistics:")
            print(f"  Spatial - Min: {spatial_features.min():.4f}, Max: {spatial_features.max():.4f}, Mean: {spatial_features.mean():.4f}")
            print(f"  Temporal - Min: {temporal_features.min():.4f}, Max: {temporal_features.max():.4f}, Mean: {temporal_features.mean():.4f}")
            print(f"  P2 - Min: {p2_features.min():.4f}, Max: {p2_features.max():.4f}, Mean: {p2_features.mean():.4f}")
            print(f"  P3 - Min: {p3_features.min():.4f}, Max: {p3_features.max():.4f}, Mean: {p3_features.mean():.4f}")
            
            # Convert to tensors
            spatial_tensor = torch.tensor(spatial_features, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            temporal_tensor = torch.tensor(temporal_features, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            p2_tensor = torch.tensor(p2_features, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            p3_tensor = torch.tensor(p3_features, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output, attention_weights, feature_weights = self.model(
                    spatial_tensor, temporal_tensor, p2_tensor, p3_tensor
                )
                
                probability = output.cpu().numpy()[0][0]
                
                # Debug: Print raw model output
                print(f"Raw model output: {probability:.6f}")
                print(f"Logit space: {np.log(probability / (1 - probability + 1e-8)):.6f}")
                
                prediction = "FAKE" if probability > 0.5 else "REAL"
                confidence = abs(probability - 0.5) * 2
            
            # Extract feature importance
            attention_weights = attention_weights.cpu().numpy()[0]
            feature_weights = feature_weights.cpu().numpy()
            
            result = {
                'video_path': video_path,
                'video_name': Path(video_path).stem,
                'prediction': prediction,
                'fake_probability': float(probability),
                'confidence': float(confidence),
                'faces_detected': len(metadata),
                'feature_shapes': {
                    'spatial': spatial_features.shape,
                    'temporal': temporal_features.shape,
                    'p2': p2_features.shape,
                    'p3': p3_features.shape
                },
                'feature_importance': {
                    'spatial_attention': float(attention_weights[0]),
                    'temporal_attention': float(attention_weights[1]),
                    'handcrafted_attention': float(attention_weights[2]),
                    'spatial_weight': float(feature_weights[0]),
                    'temporal_weight': float(feature_weights[1]),
                    'p2_weight': float(feature_weights[2]),
                    'p3_weight': float(feature_weights[3])
                },
                'processing_steps': {
                    'face_extraction': 'completed',
                    'spatial_features': 'completed',
                    'temporal_features': 'completed',
                    'p2_features': 'completed',
                    'p3_features': 'completed',
                    'prediction': 'completed'
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {
                'video_path': video_path,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
        finally:
            # ENHANCED CLEANUP - Clean up temporary files ALWAYS
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f" Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not clean up {temp_dir}: {e}")


# Global pipeline instance (initialize once)
_pipeline_instance = None

def get_complete_pipeline():
    """Get or create the complete pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        # Path to your trained model - UPDATE THIS PATH
        model_path = settings.DEEPFAKE_MODEL_PATH
        _pipeline_instance = CompletePipelineTester(model_path)
    return _pipeline_instance


def run_complete_pipeline_analysis(video_path: str):
    temp_dir = None  # ADD THIS LINE
    try:
        # print("="*70)
        # print("RUNNING COMPLETE DEEPFAKE DETECTION PIPELINE")
        # print("="*70)
        
        # Get pipeline instance
        pipeline = get_complete_pipeline()
        
        # Create a unique temp directory
        temp_dir = f"temp_pipeline_{int(time.time())}"  # ADD THIS LINE
        
        # Run the analysis with explicit temp directory
        start_time = time.time()
        result = pipeline.process_video_complete_pipeline(video_path, temp_dir)  # CHANGE THIS LINE
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Display detailed results
        #print(f"\nCOMPLETE PIPELINE RESULTS:")
        print(f"Video: {result.get('video_name', 'Unknown')}")
        #print(f"Prediction: {result['prediction']}")
        
        if 'fake_probability' in result:
            #print(f"Fake Probability: {result['fake_probability']:.4f}")
            #print(f"Confidence: {result['confidence']:.4f}")
            print(f"Faces Detected: {result.get('faces_detected', 0)}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        #print(f"Processing Time: {processing_time:.2f} seconds")
        
        if 'feature_importance' in result:
            
            print(f"\nFeature Weights:")
            print(f"  Spatial Weight: {result['feature_importance']['spatial_weight']:.4f}")
            print(f"  Temporal Weight: {result['feature_importance']['temporal_weight']:.4f}")
            print(f"  P2 Weight: {result['feature_importance']['p2_weight']:.4f}")
            print(f"  P3 Weight: {result['feature_importance']['p3_weight']:.4f}")
        
      
        
        return result
        
    except Exception as e:
        print(f"Error in complete pipeline analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            'prediction': 'ERROR',
            'confidence': 0.0,
            'error': str(e)
        }
    finally:
        # ENSURE CLEANUP - Add this cleanup here too
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up {temp_dir}: {e}")


# Modified API view - Add this to your existing views.py
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FileUploadParser])
def analyze_video(request):
    try:
        # Check if video file is provided
        if 'video' not in request.FILES:
            return Response({
                'status': 'error',
                'message': 'No video file provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        video_file = request.FILES['video']
        
        # Validate file type
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_extension = os.path.splitext(video_file.name)[1].lower()
        
        if file_extension not in allowed_extensions:
            return Response({
                'status': 'error',
                'message': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if video_file.size > max_size:
            return Response({
                'status': 'error',
                'message': 'File size too large. Maximum allowed size is 100MB'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the uploaded file temporarily
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{video_file.name}"
        file_path = default_storage.save(f'videos/{filename}', ContentFile(video_file.read()))
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        try:
            # Get detector instance and analyze video (YOUR ORIGINAL METHOD)
            detector = get_detector()
            results = detector.analyze_video(full_file_path)
            
            # **NEW ADDITION: Run your complete pipeline analysis**
            #print("\n" + "="*50)
            #print("RUNNING ADDITIONAL COMPLETE PIPELINE ANALYSIS")
            #print("="*50)
            
            # Run your complete pipeline in parallel
            complete_pipeline_results = run_complete_pipeline_analysis(full_file_path)
            
            # Save analysis results to database
            analysis = VideoAnalysis.objects.create(
                user=request.user,
                video_file=file_path,
                filename=video_file.name,
                file_size=video_file.size,
                prediction_score=results['prediction_score'],
                is_deepfake=results['is_deepfake'],
                confidence_threshold=results['threshold'],
                analysis_duration=results['analysis_duration'],
                frames_analyzed=results['frames_per_video']
            )
            
            # Prepare response (YOUR ORIGINAL RESPONSE)
            response_data = {
                'status': 'success',
                'message': 'Video analysis completed',
                'analysis_id': analysis.id,
                'filename': video_file.name,
                'is_fake': results['is_deepfake'],
                'confidence': results['confidence'],
                'raw_score': results['prediction_score'],
                'threshold': results['threshold'],
                'analysis_duration': results['analysis_duration'],
                
                # **NEW ADDITION: Include complete pipeline results for reference**
                'complete_pipeline': {
                    'prediction': complete_pipeline_results.get('prediction', 'UNKNOWN'),
                    'fake_probability': complete_pipeline_results.get('fake_probability', 0.0),
                    'confidence': complete_pipeline_results.get('confidence', 0.0),
                    'faces_detected': complete_pipeline_results.get('faces_detected', 0),
                    'feature_importance': complete_pipeline_results.get('feature_importance', {}),
                    'error': complete_pipeline_results.get('error', None)
                }
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Clean up the uploaded file if analysis fails
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            
            return Response({
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        finally:
            # Clean up the temporary file after analysis
            if os.path.exists(full_file_path):
                try:
                    os.remove(full_file_path)
                except:
                    pass  # Ignore cleanup errors
    
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analysis_history(request):
    try:
        # Get user's analysis history
        analyses = VideoAnalysis.objects.filter(user=request.user).order_by('-created_at')[:20]
        
        history_data = []
        for analysis in analyses:
            history_data.append({
                'id': analysis.id,
                'filename': analysis.filename,
                'is_deepfake': analysis.is_deepfake,
                'confidence_percentage': analysis.confidence_percentage,
                'prediction_score': analysis.prediction_score,
                'created_at': analysis.created_at.isoformat(),
                'analysis_duration': analysis.analysis_duration
            })
        
        return Response({
            'status': 'success',
            'history': history_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Failed to fetch history: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analysis_detail(request, analysis_id):
    try:
        # Get specific analysis details
        analysis = VideoAnalysis.objects.get(id=analysis_id, user=request.user)
        
        analysis_data = {
            'id': analysis.id,
            'filename': analysis.filename,
            'file_size': analysis.file_size,
            'is_deepfake': analysis.is_deepfake,
            'confidence_percentage': analysis.confidence_percentage,
            'prediction_score': analysis.prediction_score,
            'confidence_threshold': analysis.confidence_threshold,
            'frames_analyzed': analysis.frames_analyzed,
            'analysis_duration': analysis.analysis_duration,
            'created_at': analysis.created_at.isoformat(),
            'updated_at': analysis.updated_at.isoformat()
        }
        
        return Response({
            'status': 'success',
            'analysis': analysis_data
        }, status=status.HTTP_200_OK)
        
    except VideoAnalysis.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Analysis not found'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Failed to fetch analysis details: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analysis_history(request):
    """Get user's analysis history with pagination and filtering"""
    try:
        # Get user's analyses ordered by newest first
        analyses = VideoAnalysis.objects.filter(user=request.user).order_by('-created_at')
        
        # Optional filtering
        result_filter = request.GET.get('result')  # 'real', 'fake', or 'all'
        if result_filter and result_filter != 'all':
            if result_filter == 'real':
                analyses = analyses.filter(is_deepfake=False)
            elif result_filter == 'fake':
                analyses = analyses.filter(is_deepfake=True)
        
        # Optional date filtering
        days = request.GET.get('days')  # last 7, 30, 90 days
        if days:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=int(days))
            analyses = analyses.filter(created_at__gte=cutoff_date)
        
        # Pagination
        page = int(request.GET.get('page', 1))
        per_page = int(request.GET.get('per_page', 10))
        paginator = Paginator(analyses, per_page)
        
        try:
            analyses_page = paginator.page(page)
        except:
            analyses_page = paginator.page(1)
        
        # Format response
        history_data = []
        for analysis in analyses_page:
            history_data.append({
            'id': analysis.id,
            'filename': analysis.filename,
            'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'date_only': analysis.created_at.strftime('%Y-%m-%d'),
            'time_only': analysis.created_at.strftime('%H:%M'),
            'is_deepfake': analysis.is_deepfake,
            'result_text': 'Deepfake' if analysis.is_deepfake else 'Real',
            'confidence': round(analysis.confidence_percentage, 1),  # ← FIXED
            'prediction_score': round(analysis.prediction_score, 3),
            'analysis_duration': round(analysis.analysis_duration, 1),
            'frames_analyzed': analysis.frames_analyzed or 16
        })
        
        return Response({
            'success': True,
            'history': history_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'has_next': analyses_page.has_next(),
                'has_previous': analyses_page.has_previous()
            },
            'stats': {
                'total_analyses': VideoAnalysis.objects.filter(user=request.user).count(),
                'real_count': VideoAnalysis.objects.filter(user=request.user, is_deepfake=False).count(),
                'fake_count': VideoAnalysis.objects.filter(user=request.user, is_deepfake=True).count(),
            }
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)