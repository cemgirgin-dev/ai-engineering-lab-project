"""
Few-Shot Learning Module for AI Object Counting Application
Implements advanced mode for counting objects not in predefined set
"""

import os
import json
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime
import cv2
from collections import defaultdict
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FewShotDataset(Dataset):
    """Enhanced dataset for few-shot learning with data augmentation"""
    
    def __init__(self, images: List[str], labels: List[str], transform=None, augment=True):
        self.images = images
        self.labels = labels
        self.augment = augment
        
        # Base transform
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Augmentation transforms for training
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose(base_transforms)
    
    def __len__(self):
        return len(self.images) * (2 if self.augment else 1)
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.images)
        image_path = self.images[original_idx]
        label = self.labels[original_idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image if loading fails
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label

class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor - keeping original architecture"""
    
    def __init__(self, feature_dim=512):
        super(ResNetFeatureExtractor, self).__init__()
        
        # ResNet-like architecture (simplified but effective)
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # ResNet block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # ResNet block 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Initialize weights like ResNet
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        # L2 normalize features for better similarity computation
        features = F.normalize(features, p=2, dim=1)
        return features

class FewShotLearner:
    """
    Few-shot learning system using ResNet architecture - FIXED VERSION
    """
    
    def __init__(self, model_dir: str = "few_shot_models", feature_dim: int = 512):
        self.model_dir = model_dir
        self.feature_dim = feature_dim
        self.known_objects = {}
        self.feature_extractor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # EXTREME parameters to prevent overcounting 
        self.similarity_threshold = 0.98  # EXTREMELY high
        self.confidence_threshold = 0.97  # EXTREMELY high 
        self.min_window_size = 96
        self.max_window_size = 128  
        self.window_stride = 96  # Window stride = window size (no overlap)
        self.nms_threshold = 0.1   # VERY aggressive NMS
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize ResNet-based feature extractor
        self._initialize_feature_extractor()
        
        logger.info(f"ResNet-based few-shot learner initialized on device: {self.device}")
    
    def _initialize_feature_extractor(self):
        """Initialize ResNet-based feature extractor"""
        try:
            self.feature_extractor = ResNetFeatureExtractor(self.feature_dim).to(self.device)
            self.feature_extractor.eval()
            logger.info("ResNet feature extractor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ResNet feature extractor: {str(e)}")
            raise
    
    def learn_new_object(self, object_name: str, training_images: List[str], 
                        validation_images: List[str] = None) -> Dict:
        """
        Learn new object with ResNet features
        """
        try:
            logger.info(f"Learning new object type: {object_name}")
            logger.info(f"Training images: {len(training_images)}")
            
            if len(training_images) < 2:
                raise ValueError("At least 2 training images are required for few-shot learning")
            
            # Prepare training data with light augmentation
            train_dataset = FewShotDataset(training_images, [object_name] * len(training_images), augment=True)
            train_loader = DataLoader(train_dataset, batch_size=min(4, len(training_images) * 2), shuffle=True)
            
            # Extract features with ResNet
            features = self._extract_features(train_loader)
            
            if len(features) == 0:
                raise Exception("Failed to extract features from training images")
            
            # Create strict representation to avoid false positives
            object_representation = self._create_strict_representation(features, object_name)
            
            # Store learned object
            self.known_objects[object_name] = {
                'representation': object_representation,
                'training_images': training_images,
                'learned_at': datetime.now().isoformat(),
                'feature_dim': self.feature_dim
            }
            
            # Validation
            validation_results = {}
            if validation_images:
                validation_results = self._validate_object(object_name, validation_images)
            
            # Save model
            self._save_object_model(object_name)
            
            results = {
                'object_name': object_name,
                'training_images_count': len(training_images),
                'validation_images_count': len(validation_images) if validation_images else 0,
                'feature_dim': self.feature_dim,
                'features_extracted': len(features),
                'learning_successful': True,
                'validation_results': validation_results,
                'learned_at': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully learned object type: {object_name} with {len(features)} ResNet features")
            return results
            
        except Exception as e:
            logger.error(f"Error learning new object {object_name}: {str(e)}")
            return {
                'object_name': object_name,
                'learning_successful': False,
                'error': str(e),
                'learned_at': datetime.now().isoformat()
            }
    
    def _extract_features(self, data_loader: DataLoader) -> np.ndarray:
        """Extract ResNet features"""
        self.feature_extractor.eval()
        features = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                try:
                    images = images.to(self.device)
                    batch_features = self.feature_extractor(images)
                    features.append(batch_features.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
                    continue
        
        if features:
            return np.vstack(features)
        else:
            return np.array([])
    
    def _create_strict_representation(self, features: np.ndarray, object_name: str) -> Dict:
        """Create strict representation using ResNet features"""
        if len(features) == 0:
            raise Exception("No features to create representation from")
        
        # Use both mean and median for robustness
        mean_features = np.mean(features, axis=0)
        median_features = np.median(features, axis=0)
        std_features = np.std(features, axis=0)
        
        # Calculate internal similarity to set strict thresholds
        similarity_matrix = cosine_similarity(features)
        np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
        min_internal_similarity = np.min(similarity_matrix[similarity_matrix > 0])
        avg_internal_similarity = np.mean(similarity_matrix[similarity_matrix > 0])
        
        return {
            'mean_features': mean_features,
            'median_features': median_features,
            'std_features': std_features,
            'min_internal_similarity': min_internal_similarity,
            'avg_internal_similarity': avg_internal_similarity,
            'all_features': features,
            'feature_count': len(features),
            'object_name': object_name
        }
    
    def _validate_object(self, object_name: str, validation_images: List[str]) -> Dict:
        """Validate with strict criteria"""
        try:
            val_dataset = FewShotDataset(validation_images, [object_name] * len(validation_images), augment=False)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
            
            val_features = self._extract_features(val_loader)
            
            if len(val_features) == 0:
                return {'validation_successful': False, 'error': 'No validation features extracted'}
            
            representation = self.known_objects[object_name]['representation']
            mean_features = representation['mean_features']
            
            # Calculate similarities
            similarities = []
            for val_feature in val_features:
                similarity = cosine_similarity(
                    val_feature.reshape(1, -1),
                    mean_features.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            
            # Very strict validation criteria
            validation_successful = (
                avg_similarity > 0.85 and 
                min_similarity > 0.75
            )
            
            return {
                'avg_similarity': float(avg_similarity),
                'min_similarity': float(min_similarity),
                'validation_images_count': len(validation_images),
                'validation_successful': validation_successful
            }
            
        except Exception as e:
            logger.error(f"Error validating object {object_name}: {str(e)}")
            return {
                'validation_successful': False,
                'error': str(e)
            }
    
    def recognize_object(self, image_path: str, threshold: float = 0.92) -> Dict:
        """ResNet-based object recognition with strict threshold"""
        try:
            if not self.known_objects:
                return {
                    'recognized': False,
                    'message': 'No objects learned yet',
                    'similarities': {}
                }
            
            # Load and process image
            try:
                image = Image.open(image_path).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(image_tensor)
                    image_features = features.cpu().numpy()[0]
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                return {
                    'recognized': False,
                    'message': f'Error processing image: {str(e)}',
                    'similarities': {}
                }
            
            # Compare with all learned objects using strict criteria
            similarities = {}
            for obj_name, obj_data in self.known_objects.items():
                try:
                    representation = obj_data['representation']
                    mean_features = representation['mean_features']
                    median_features = representation['median_features']
                    avg_internal_sim = representation.get('avg_internal_similarity', 0.8)
                    
                    # Calculate similarity to both mean and median
                    mean_similarity = cosine_similarity(
                        image_features.reshape(1, -1),
                        mean_features.reshape(1, -1)
                    )[0][0]
                    
                    median_similarity = cosine_similarity(
                        image_features.reshape(1, -1),
                        median_features.reshape(1, -1)
                    )[0][0]
                    
                    # Use average but penalize if too different from internal consistency
                    avg_similarity = (mean_similarity + median_similarity) / 2
                    
                    # Penalize if similarity is much lower than internal consistency
                    if avg_similarity < avg_internal_sim * 0.8:
                        avg_similarity *= 0.7  # Apply penalty
                    
                    similarities[obj_name] = float(avg_similarity)
                    
                except Exception as e:
                    logger.warning(f"Error calculating similarity for {obj_name}: {e}")
                    similarities[obj_name] = 0.0
            
            if not similarities:
                return {
                    'recognized': False,
                    'message': 'No similarities calculated',
                    'similarities': {}
                }
            
            # Find best match with strict threshold
            best_match = max(similarities.items(), key=lambda x: x[1])
            best_object, best_similarity = best_match
            
            recognized = best_similarity >= threshold
            
            return {
                'recognized': recognized,
                'best_match': best_object if recognized else None,
                'best_similarity': float(best_similarity),
                'similarities': similarities,
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"Error recognizing objects in {image_path}: {str(e)}")
            return {
                'recognized': False,
                'message': f'Error: {str(e)}',
                'similarities': {}
            }
    
    def count_learned_objects(self, image_path: str, object_name: str) -> Dict:
        """FIXED counting that prevents overcounting and reports correct segments"""
        try:
            if object_name not in self.known_objects:
                return {
                    'count': 0,
                    'confidence': 0.0,
                    'segments_found': 0,
                    'segments_analyzed': 0,
                    'error': f'Object type "{object_name}" not learned yet',
                    'details': {
                        'segments_found': 0,
                        'target_segments': 0,
                        'confidence_scores': [],
                        'segment_details': []
                    }
                }
            
            # Load image
            try:
                image = Image.open(image_path).convert('RGB')
                width, height = image.size
                logger.info(f"Processing image {image_path}: {width}x{height}")
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return {
                    'count': 0,
                    'confidence': 0.0,
                    'segments_found': 0,
                    'segments_analyzed': 0,
                    'error': f'Error loading image: {str(e)}',
                    'details': {'segments_found': 0, 'target_segments': 0}
                }
            
            # CONSERVATIVE sliding window approach
            detections = []
            total_windows = 0
            
            # Only use one window size to reduce false positives
            window_size = self.min_window_size
            
            # Calculate number of windows
            x_windows = max(1, (width - window_size) // self.window_stride + 1)
            y_windows = max(1, (height - window_size) // self.window_stride + 1)
            expected_windows = x_windows * y_windows
            
            logger.info(f"Will analyze {expected_windows} windows of size {window_size}x{window_size}")
            
            for y_idx in range(y_windows):
                for x_idx in range(x_windows):
                    y = min(y_idx * self.window_stride, height - window_size)
                    x = min(x_idx * self.window_stride, width - window_size)
                    
                    try:
                        # Extract window
                        window = image.crop((x, y, x + window_size, y + window_size))
                        
                        # Save window temporarily
                        temp_path = f"temp_window_{x}_{y}.png"
                        window.save(temp_path)
                        total_windows += 1
                        
                        try:
                            # Recognize object in window with VERY HIGH threshold
                            recognition_result = self.recognize_object(temp_path, threshold=self.similarity_threshold)
                            
                            if (recognition_result.get('recognized', False) and 
                                recognition_result.get('best_match') == object_name):
                                
                                similarity = recognition_result.get('best_similarity', 0.0)
                                
                                # DOUBLE CHECK with confidence threshold
                                if similarity >= self.confidence_threshold:
                                    detection = {
                                        'x': x,
                                        'y': y,
                                        'width': window_size,
                                        'height': window_size,
                                        'confidence': similarity,
                                        'similarity': similarity
                                    }
                                    detections.append(detection)
                        
                        except Exception as e:
                            logger.warning(f"Error recognizing window at ({x}, {y}): {e}")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    
                    except Exception as e:
                        logger.warning(f"Error processing window at ({x}, {y}): {e}")
                        continue
            
            logger.info(f"Analyzed {total_windows} windows, found {len(detections)} initial detections")
            
            # Aggressive NMS to prevent overcounting
            final_detections = self._aggressive_nms(detections)
            
            # Calculate final metrics
            object_count = len(final_detections)
            
            if final_detections:
                avg_confidence = np.mean([d['confidence'] for d in final_detections])
                confidence_scores = [d['confidence'] for d in final_detections]
            else:
                avg_confidence = 0.0
                confidence_scores = []
            
            logger.info(f"Final count after aggressive NMS: {object_count}, avg confidence: {avg_confidence:.3f}")
            
            # FIXED: Return correct segment information
            result = {
                'count': object_count,
                'confidence': float(avg_confidence),
                'avg_similarity': float(avg_confidence),
                'segments_found': total_windows,  # FIXED: Report actual windows analyzed
                'segments_analyzed': total_windows,  # FIXED: Same value
                'detections': final_detections,
                'windows_checked': total_windows,
                'object_name': object_name,
                'details': {
                    'segments_found': total_windows,  # FIXED: Consistent with above
                    'target_segments': object_count,
                    'confidence_scores': confidence_scores,
                    'segment_details': [
                        {
                            'id': i,
                            'label': object_name,
                            'predicted': object_name,
                            'is_target': True,
                            'confidence': d['confidence']
                        }
                        for i, d in enumerate(final_detections)
                    ]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error counting objects in {image_path}: {str(e)}")
            return {
                'count': 0,
                'confidence': 0.0,
                'segments_found': 0,
                'segments_analyzed': 0,
                'error': str(e),
                'details': {'segments_found': 0, 'target_segments': 0}
            }
    
    def _aggressive_nms(self, detections: List[Dict]) -> List[Dict]:
        """FIXED aggressive NMS that actually works"""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        removed_count = 0
        
        for current in detections:
            # Check if this detection overlaps significantly with any kept detection
            should_keep = True
            
            for kept in keep:
                overlap = self._calculate_overlap(current, kept)
                if overlap > self.nms_threshold:  # If overlap is too high, remove it
                    should_keep = False
                    removed_count += 1
                    break
            
            if should_keep:
                keep.append(current)
        
        logger.info(f"FIXED NMS: kept {len(keep)} detections, removed {removed_count}")
        return keep
    
    def _calculate_overlap(self, det1: Dict, det2: Dict) -> float:
        """Calculate IoU overlap between two detections"""
        try:
            x1_1, y1_1 = det1['x'], det1['y']
            x2_1, y2_1 = x1_1 + det1['width'], y1_1 + det1['height']
            
            x1_2, y1_2 = det2['x'], det2['y']
            x2_2, y2_2 = x1_2 + det2['width'], y1_2 + det2['height']
            
            # Calculate intersection
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            
            # Calculate union
            area1 = det1['width'] * det1['height']
            area2 = det2['width'] * det2['height']
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating overlap: {e}")
            return 0.0
    
    def _save_object_model(self, object_name: str):
        """Save the learned object model to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{object_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.known_objects[object_name], f)
            logger.info(f"Saved ResNet model for {object_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model for {object_name}: {str(e)}")
    
    def load_object_model(self, object_name: str) -> bool:
        """Load a learned object model from disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{object_name}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.known_objects[object_name] = pickle.load(f)
                logger.info(f"Loaded ResNet model for {object_name} from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found for {object_name}")
                return False
        except Exception as e:
            logger.error(f"Error loading model for {object_name}: {str(e)}")
            return False
    
    def list_learned_objects(self) -> List[Dict]:
        """List all learned objects"""
        objects = []
        for obj_name, obj_data in self.known_objects.items():
            representation = obj_data.get('representation', {})
            objects.append({
                'name': obj_name,
                'training_images_count': len(obj_data['training_images']),
                'learned_at': obj_data['learned_at'],
                'feature_dim': obj_data['feature_dim'],
                'feature_count': representation.get('feature_count', 0)
            })
        return objects
    
    def delete_object(self, object_name: str) -> bool:
        """Delete a learned object"""
        try:
            if object_name in self.known_objects:
                del self.known_objects[object_name]
                
                # Remove model file
                model_path = os.path.join(self.model_dir, f"{object_name}_model.pkl")
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                logger.info(f"Deleted object: {object_name}")
                return True
            else:
                logger.warning(f"Object {object_name} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting object {object_name}: {str(e)}")
            return False

# Global few-shot learner instance
few_shot_learner = FewShotLearner()
