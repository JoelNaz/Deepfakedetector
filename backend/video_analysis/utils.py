# video_analysis/utils.py - Uses YOUR EXACT model loading logic
import os
import re
import time
import torch
from django.conf import settings
from .kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from training.zoo.classifiers import DeepFakeClassifier

class DeepfakeDetector:
    def __init__(self):
        self.models = []
        self.face_extractor = None
        self.input_size = 380
        self.frames_per_video = 16
        self._load_models()
        
    def _load_models(self):
        """Load models using flexible loading to match your exact saved structure"""
        try:
            model_path = settings.MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            #print(f"loading state dict {model_path}")
            
            # Load checkpoint first to see its structure
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location="cpu")
            
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            # Clean module. prefix
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                key = re.sub("^module.", "", k)
                cleaned_state_dict[key] = v
            
            # Check what keys we have to understand the model structure
            has_fc = any('fc.weight' in k for k in cleaned_state_dict.keys())
            has_classifier = any('classifier.weight' in k for k in cleaned_state_dict.keys())
            has_encoder_classifier = any('encoder.classifier' in k for k in cleaned_state_dict.keys())
            
            print(f"Model structure: fc={has_fc}, classifier={has_classifier}, encoder.classifier={has_encoder_classifier}")
            
            # Create base model 
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cpu")
            
            # Try loading with strict=False to see what works
            try:
                model.load_state_dict(cleaned_state_dict, strict=True)
                print("✅ Loaded model successfully!")
            except RuntimeError as e:
                #print(f"❌ Strict loading failed: {e}")
                
                # Let's try a more flexible approach
                #print("🔧 Trying flexible loading...")
                
                # Get model's current state dict
                model_dict = model.state_dict()
                
                # Try to match keys flexibly
                matched_dict = {}
                for model_key in model_dict.keys():
                    model_shape = model_dict[model_key].shape
                    
                    # Try exact match first
                    if model_key in cleaned_state_dict:
                        if cleaned_state_dict[model_key].shape == model_shape:
                            matched_dict[model_key] = cleaned_state_dict[model_key]
                            #print(f"✅ Exact match: {model_key}")
                        else:
                            print(f"❌ Shape mismatch for {model_key}: expected {model_shape}, got {cleaned_state_dict[model_key].shape}")
                    
                    # Try alternative names for fc vs classifier
                    elif 'fc.' in model_key:
                        alt_key = model_key.replace('fc.', 'classifier.')
                        if alt_key in cleaned_state_dict and cleaned_state_dict[alt_key].shape == model_shape:
                            matched_dict[model_key] = cleaned_state_dict[alt_key]
                            #print(f"✅ Mapped: {alt_key} -> {model_key}")
                    elif 'classifier.' in model_key:
                        alt_key = model_key.replace('classifier.', 'fc.')
                        if alt_key in cleaned_state_dict and cleaned_state_dict[alt_key].shape == model_shape:
                            matched_dict[model_key] = cleaned_state_dict[alt_key]
                            #print(f"✅ Mapped: {alt_key} -> {model_key}")
                
                # Update model with matched weights
                model_dict.update(matched_dict)
                model.load_state_dict(model_dict)
                #print(f"✅ Loaded {len(matched_dict)}/{len(model_dict)} layers")
            
            model.eval()
            del checkpoint
            self.models.append(model.half())
            
            # Your exact face extractor setup
            frames_per_video = 16
            video_reader = VideoReader()
            video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
            self.face_extractor = FaceExtractor(video_read_fn)
            
            #print(f"Successfully loaded {len(self.models)} model(s)")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def analyze_video(self, video_path):
        """Analyze video using YOUR EXACT prediction logic"""
        if not self.models or not self.face_extractor:
            raise RuntimeError("Models not loaded properly")
            
        start_time = time.time()
        
        try:
            # Use your EXACT predict_on_video function with YOUR parameters
            prediction_score = predict_on_video(
                face_extractor=self.face_extractor,
                video_path=video_path,
                batch_size=self.frames_per_video,
                input_size=self.input_size,
                models=self.models,
                strategy=confident_strategy,  # Your exact strategy
                apply_compression=False
            )
            
            analysis_duration = time.time() - start_time
            threshold = 0.5  # Your exact threshold
            is_deepfake = prediction_score > threshold
            
            if is_deepfake:
                confidence = prediction_score
            else:
                confidence = 1 - prediction_score
                
            results = {
                'prediction_score': float(prediction_score),
                'is_deepfake': is_deepfake,
                'confidence': float(confidence),
                'threshold': threshold,
                'analysis_duration': analysis_duration,
                'frames_per_video': self.frames_per_video
            }
            
            return results
            
        except Exception as e:
            print(f"Error analyzing video {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

# Global detector instance (loads once per server startup)
detector = None

def get_detector():
    """Get or create global detector instance - loads model only once"""
    global detector
    if detector is None:
        print("Initializing deepfake detector using YOUR original code...")
        detector = DeepfakeDetector()
        #print("Detector ready!")
    return detector