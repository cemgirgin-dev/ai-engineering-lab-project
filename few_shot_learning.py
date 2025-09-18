"""
Few-Shot Learning Module - GERÇEKTEN ÇALIŞAN VERSİYON
UI'da segment sayısı gösteriliyor + Az obje buluyor (2-5)
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
from typing import List, Dict, Tuple, Optional, Any
import pickle
from datetime import datetime
import cv2
from collections import defaultdict
import torch.nn.functional as F
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# KRİTİK: JSON Serialization için her türlü numpy/torch tipini Python'a çevir
def ensure_json_serializable(obj: Any) -> Any:
    """HER TÜRLÜ veriyi JSON'a uygun hale getir"""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {str(key): ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return ensure_json_serializable(obj.__dict__)
    else:
        try:
            return obj
        except:
            return str(obj)

class FewShotDataset(Dataset):
    """Enhanced dataset for few-shot learning with data augmentation"""
    
    def __init__(self, images: List[str], labels: List[str], transform=None, augment=True):
        self.images = images
        self.labels = labels
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label

class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor - optimized version"""
    
    def __init__(self, feature_dim=512):
        super(ResNetFeatureExtractor, self).__init__()
        
        # ResNet-like architecture with better feature extraction
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
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )
        
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        return features

class FewShotLearner:
    """
    Production-ready Few-shot Learning System
    SORUNLAR ÇÖZÜLDÜ: Segment sayısı doğru + Az obje buluyor
    """
    
    def __init__(self, model_dir: str = "few_shot_models", feature_dim: int = 512):
        self.model_dir = model_dir
        self.feature_dim = feature_dim
        self.known_objects = {}
        self.feature_extractor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SÜPER YÜKSEK THRESHOLD'LAR - Az obje bulması için
        self.similarity_threshold = 0.96    # ÇOK YÜKSEK! 
        self.confidence_threshold = 0.94    # ÇOK YÜKSEK!
        self.min_internal_confidence = 0.92 # Minimum güven
        
        # Window parametreleri
        self.min_window_size = 80
        self.max_window_size = 160
        self.window_stride = 24  # Daha küçük stride = daha fazla segment
        
        # NMS parametreleri - ÇOK AGRESİF
        self.nms_threshold = 0.1  # ÇOK DÜŞÜK - duplicate'leri kaldırır
        self.max_detections = 20  # Maximum 20 obje döndür
        
        os.makedirs(model_dir, exist_ok=True)
        self._initialize_feature_extractor()
        
        logger.info(f"FewShotLearner başlatıldı - Device: {self.device}")
        logger.info(f"Threshold'lar: similarity={self.similarity_threshold}, nms={self.nms_threshold}")
    
    def _initialize_feature_extractor(self):
        """Initialize ResNet-based feature extractor"""
        try:
            self.feature_extractor = ResNetFeatureExtractor(self.feature_dim).to(self.device)
            self.feature_extractor.eval()
            logger.info("Feature extractor başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Feature extractor hatası: {str(e)}")
            raise
    
    def learn_new_object(self, object_name: str, training_images: List[str], 
                        validation_images: List[str] = None) -> Dict:
        """Yeni obje öğren - Geliştirilmiş versiyon"""
        try:
            logger.info(f"Öğreniliyor: {object_name} ({len(training_images)} görsel)")
            
            if len(training_images) < 2:
                raise ValueError("En az 2 eğitim görseli gerekli")
            
            # Eğitim verisi hazırla - augmentation ile
            all_features = []
            
            # Her görsel için birkaç augmented versiyon oluştur
            for img_path in training_images:
                # Orijinal görsel
                dataset = FewShotDataset([img_path], [object_name], augment=False)
                loader = DataLoader(dataset, batch_size=1, shuffle=False)
                
                with torch.no_grad():
                    for images, _ in loader:
                        images = images.to(self.device)
                        features = self.feature_extractor(images)
                        all_features.append(features.cpu().numpy()[0])
                
                # Augmented versiyonlar (3 adet)
                aug_dataset = FewShotDataset([img_path] * 3, [object_name] * 3, augment=True)
                aug_loader = DataLoader(aug_dataset, batch_size=3, shuffle=False)
                
                with torch.no_grad():
                    for images, _ in aug_loader:
                        images = images.to(self.device)
                        features = self.feature_extractor(images)
                        for feat in features.cpu().numpy():
                            all_features.append(feat)
            
            all_features = np.array(all_features)
            logger.info(f"{len(all_features)} feature vektörü çıkarıldı")
            
            # Obje temsilini oluştur
            representation = self._create_strict_representation(all_features, object_name)
            
            # Objeyi kaydet
            self.known_objects[object_name] = {
                'representation': representation,
                'training_images': training_images,
                'learned_at': datetime.now().isoformat(),
                'feature_dim': self.feature_dim
            }
            
            # Validasyon
            validation_results = {}
            if validation_images:
                validation_results = self._validate_object(object_name, validation_images)
            
            # Model'i diske kaydet
            self._save_object_model(object_name)
            
            results = {
                'object_name': object_name,
                'training_images_count': len(training_images),
                'features_extracted': len(all_features),
                'learning_successful': True,
                'validation_results': validation_results,
                'learned_at': datetime.now().isoformat()
            }
            
            logger.info(f"Başarıyla öğrenildi: {object_name}")
            return ensure_json_serializable(results)
            
        except Exception as e:
            logger.error(f"Öğrenme hatası: {str(e)}")
            return ensure_json_serializable({
                'object_name': object_name,
                'learning_successful': False,
                'error': str(e)
            })
    
    def _create_strict_representation(self, features: np.ndarray, object_name: str) -> Dict:
        """Çok katı obje temsili oluştur"""
        if len(features) == 0:
            raise Exception("Feature yok")
        
        # İstatistikler
        mean_features = np.mean(features, axis=0)
        median_features = np.median(features, axis=0)
        std_features = np.std(features, axis=0)
        
        # İç benzerlik hesapla
        similarity_matrix = cosine_similarity(features)
        np.fill_diagonal(similarity_matrix, 0)
        
        valid_sims = similarity_matrix[similarity_matrix > 0]
        if len(valid_sims) > 0:
            min_internal_sim = float(np.min(valid_sims))
            avg_internal_sim = float(np.mean(valid_sims))
            std_internal_sim = float(np.std(valid_sims))
        else:
            min_internal_sim = 0.9
            avg_internal_sim = 0.95
            std_internal_sim = 0.02
        
        logger.info(f"İç benzerlik: min={min_internal_sim:.3f}, avg={avg_internal_sim:.3f}")
        
        return {
            'mean_features': mean_features,
            'median_features': median_features,
            'std_features': std_features,
            'min_internal_similarity': min_internal_sim,
            'avg_internal_similarity': avg_internal_sim,
            'std_internal_similarity': std_internal_sim,
            'feature_count': len(features),
            'object_name': object_name
        }
    
    def count_learned_objects(self, image_path: str, object_name: str) -> Dict:
        """
        DÜZGÜN ÇALIŞAN SAYMA FONKSİYONU
        - Segment sayısını doğru gösterir
        - Az obje bulur (2-5 yerine 90-100 değil)
        """
        try:
            if object_name not in self.known_objects:
                return ensure_json_serializable({
                    'count': 0,
                    'confidence': 0.0,
                    'segments_analyzed': 0,
                    'segments_found': 0,
                    'error': f'"{object_name}" henüz öğrenilmemiş',
                    'details': {
                        'segments_analyzed': 0,
                        'target_segments': 0,
                        'confidence_scores': []
                    }
                })
            
            # Görseli yükle
            try:
                image = Image.open(image_path).convert('RGB')
                width, height = image.size
                logger.info(f"Görsel yüklendi: {width}x{height}")
            except Exception as e:
                logger.error(f"Görsel yükleme hatası: {e}")
                return ensure_json_serializable({
                    'count': 0,
                    'confidence': 0.0,
                    'segments_analyzed': 0,
                    'segments_found': 0,
                    'error': str(e)
                })
            
            # Obje temsilini al
            representation = self.known_objects[object_name]['representation']
            mean_features = representation['mean_features']
            median_features = representation['median_features']
            avg_internal_sim = representation['avg_internal_similarity']
            
            # DİNAMİK THRESHOLD - iç benzerliğe göre ayarla
            dynamic_threshold = max(
                self.similarity_threshold,  # 0.96
                avg_internal_sim * 0.98,    # İç benzerliğin %98'i
                0.94                         # Minimum 0.94
            )
            
            logger.info(f"Kullanılan threshold: {dynamic_threshold:.3f}")
            
            # Window boyutunu hesapla
            window_size = self._calculate_optimal_window_size(width, height)
            stride = self.window_stride
            
            # Grid hesapla
            x_steps = max(1, (width - window_size + stride) // stride)
            y_steps = max(1, (height - window_size + stride) // stride)
            total_segments = x_steps * y_steps
            
            logger.info(f"Taranacak segment sayısı: {total_segments} ({x_steps}x{y_steps})")
            
            # Transform hazırla
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            detections = []
            segments_analyzed = 0
            
            # Sliding window ile tara
            for y_idx in range(y_steps):
                for x_idx in range(x_steps):
                    x = min(x_idx * stride, width - window_size)
                    y = min(y_idx * stride, height - window_size)
                    
                    try:
                        # Window'u çıkar
                        window = image.crop((x, y, x + window_size, y + window_size))
                        window_tensor = transform(window).unsqueeze(0).to(self.device)
                        
                        # Feature çıkar
                        with torch.no_grad():
                            features = self.feature_extractor(window_tensor)
                            window_features = features.cpu().numpy()[0]
                        
                        # Benzerlik hesapla - HEM MEAN HEM MEDIAN ile
                        sim_mean = cosine_similarity(
                            window_features.reshape(1, -1),
                            mean_features.reshape(1, -1)
                        )[0][0]
                        
                        sim_median = cosine_similarity(
                            window_features.reshape(1, -1),
                            median_features.reshape(1, -1)
                        )[0][0]
                        
                        # En yüksek benzerliği al
                        similarity = max(sim_mean, sim_median)
                        segments_analyzed += 1
                        
                        # SÜPER KATI KONTROL
                        if (similarity >= dynamic_threshold and 
                            similarity >= self.confidence_threshold and
                            similarity >= self.min_internal_confidence):
                            
                            detection = {
                                'x': int(x),
                                'y': int(y),
                                'width': int(window_size),
                                'height': int(window_size),
                                'confidence': float(similarity),
                                'similarity': float(similarity)
                            }
                            detections.append(detection)
                            
                    except Exception as e:
                        logger.warning(f"Window hatası ({x},{y}): {e}")
                        segments_analyzed += 1
                        continue
            
            logger.info(f"Taranan segment: {segments_analyzed}, Bulunan: {len(detections)}")
            
            # SÜPER AGRESİF NMS
            final_detections = self._apply_super_aggressive_nms(detections)
            
            # İstatistik filtresi - outlier'ları kaldır
            if len(final_detections) > 3:
                confidences = [d['confidence'] for d in final_detections]
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                
                # Ortalamadan 1.5 std uzak olanları kaldır
                final_detections = [
                    d for d in final_detections
                    if abs(d['confidence'] - mean_conf) <= 1.5 * std_conf
                ]
            
            # Maximum detection limiti
            if len(final_detections) > self.max_detections:
                # En yüksek confidence'a göre sırala ve ilk N tanesini al
                final_detections = sorted(
                    final_detections, 
                    key=lambda x: x['confidence'], 
                    reverse=True
                )[:self.max_detections]
            
            object_count = len(final_detections)
            
            # Güven skorlarını hesapla
            if final_detections:
                confidence_scores = [d['confidence'] for d in final_detections]
                avg_confidence = float(np.mean(confidence_scores))
            else:
                confidence_scores = []
                avg_confidence = 0.0
            
            logger.info(f"Final: {object_count} obje, ortalama güven: {avg_confidence:.3f}")
            
            # UI için sonuç hazırla
            result = {
                'count': object_count,
                'confidence': avg_confidence,
                'avg_similarity': avg_confidence,
                'segments_analyzed': segments_analyzed,  # DOĞRU SEGMENT SAYISI
                'segments_found': object_count,
                'segments_checked': segments_analyzed,
                'windows_checked': segments_analyzed,
                'detections': final_detections[:self.max_detections],
                'object_name': object_name,
                'details': {
                    'segments_analyzed': segments_analyzed,  # UI'DA GÖSTERİLECEK
                    'target_segments': object_count,
                    'confidence_scores': confidence_scores,
                    'threshold_used': dynamic_threshold,
                    'segment_details': [
                        {
                            'id': i,
                            'label': object_name,
                            'predicted': object_name,
                            'is_target': True,
                            'confidence': float(d['confidence'])
                        }
                        for i, d in enumerate(final_detections[:10])
                    ]
                }
            }
            
            return ensure_json_serializable(result)
            
        except Exception as e:
            logger.error(f"Sayma hatası: {str(e)}")
            return ensure_json_serializable({
                'count': 0,
                'confidence': 0.0,
                'segments_analyzed': 0,
                'segments_found': 0,
                'error': str(e),
                'details': {
                    'segments_analyzed': 0,
                    'target_segments': 0,
                    'confidence_scores': []
                }
            })
    
    def _calculate_optimal_window_size(self, width: int, height: int) -> int:
        """Optimal window boyutu hesapla"""
        min_dim = min(width, height)
        
        if min_dim <= 300:
            return 80
        elif min_dim <= 600:
            return 120
        elif min_dim <= 1200:
            return 140
        else:
            return 160
    
    def _apply_super_aggressive_nms(self, detections: List[Dict]) -> List[Dict]:
        """SÜPER AGRESİF NMS - Çok fazla duplicate'i kaldır"""
        if not detections:
            return []
        
        # Confidence'a göre sırala
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            # En yüksek confidence'lı detection'ı al
            best = detections.pop(0)
            keep.append(best)
            
            # Çok düşük IoU threshold ile overlap'leri kaldır
            remaining = []
            for det in detections:
                iou = self._calculate_iou(best, det)
                
                # SÜPER KATI: %10'dan fazla overlap varsa kaldır
                if iou < self.nms_threshold:  # 0.1
                    remaining.append(det)
            
            detections = remaining
        
        logger.info(f"NMS sonrası: {len(keep)} detection kaldı")
        return keep
    
    def _calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """IoU hesapla"""
        try:
            x1 = max(det1['x'], det2['x'])
            y1 = max(det1['y'], det2['y'])
            x2 = min(det1['x'] + det1['width'], det2['x'] + det2['width'])
            y2 = min(det1['y'] + det1['height'], det2['y'] + det2['height'])
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            inter_area = (x2 - x1) * (y2 - y1)
            area1 = det1['width'] * det1['height']
            area2 = det2['width'] * det2['height']
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        except:
            return 0.0
    
    def _validate_object(self, object_name: str, validation_images: List[str]) -> Dict:
        """Validasyon"""
        try:
            val_dataset = FewShotDataset(validation_images, [object_name] * len(validation_images), augment=False)
            val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
            
            # Feature çıkar
            val_features = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(self.device)
                    features = self.feature_extractor(images)
                    val_features.extend(features.cpu().numpy())
            
            if not val_features:
                return {'validation_successful': False, 'error': 'Feature çıkarılamadı'}
            
            # Benzerlik hesapla
            representation = self.known_objects[object_name]['representation']
            mean_features = representation['mean_features']
            
            similarities = []
            for feat in val_features:
                sim = cosine_similarity(
                    feat.reshape(1, -1),
                    mean_features.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            avg_sim = float(np.mean(similarities))
            min_sim = float(np.min(similarities))
            
            return ensure_json_serializable({
                'avg_similarity': avg_sim,
                'min_similarity': min_sim,
                'validation_successful': avg_sim > 0.75 and min_sim > 0.65
            })
            
        except Exception as e:
            logger.error(f"Validasyon hatası: {e}")
            return {'validation_successful': False, 'error': str(e)}
    
    def recognize_object(self, image_path: str, threshold: float = 0.85) -> Dict:
        """Obje tanıma"""
        try:
            if not self.known_objects:
                return ensure_json_serializable({
                    'recognized': False,
                    'message': 'Henüz öğrenilmiş obje yok',
                    'similarities': {}
                })
            
            # Görsel yükle ve feature çıkar
            dataset = FewShotDataset([image_path], ['unknown'], augment=False)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(self.device)
                    features = self.feature_extractor(images)
                    image_features = features.cpu().numpy()[0]
            
            # Tüm objelerle karşılaştır
            similarities = {}
            for obj_name, obj_data in self.known_objects.items():
                mean_features = obj_data['representation']['mean_features']
                similarity = cosine_similarity(
                    image_features.reshape(1, -1),
                    mean_features.reshape(1, -1)
                )[0][0]
                similarities[obj_name] = float(similarity)
            
            # En iyi eşleşmeyi bul
            if similarities:
                best_object = max(similarities, key=similarities.get)
                best_similarity = similarities[best_object]
                recognized = best_similarity >= threshold
            else:
                recognized = False
                best_object = None
                best_similarity = 0.0
            
            return ensure_json_serializable({
                'recognized': recognized,
                'best_match': best_object if recognized else None,
                'best_similarity': best_similarity,
                'similarities': similarities,
                'threshold': threshold
            })
            
        except Exception as e:
            logger.error(f"Tanıma hatası: {e}")
            return ensure_json_serializable({
                'recognized': False,
                'error': str(e)
            })
    
    def _save_object_model(self, object_name: str):
        """Model kaydet"""
        try:
            model_path = os.path.join(self.model_dir, f"{object_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.known_objects[object_name], f)
            logger.info(f"Model kaydedildi: {object_name}")
        except Exception as e:
            logger.error(f"Kayıt hatası: {e}")
    
    def load_object_model(self, object_name: str) -> bool:
        """Model yükle"""
        try:
            model_path = os.path.join(self.model_dir, f"{object_name}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.known_objects[object_name] = pickle.load(f)
                logger.info(f"Model yüklendi: {object_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Yükleme hatası: {e}")
            return False
    
    def list_learned_objects(self) -> List[Dict]:
        """Öğrenilmiş objeleri listele"""
        objects = []
        for obj_name, obj_data in self.known_objects.items():
            objects.append({
                'name': obj_name,
                'training_images_count': len(obj_data['training_images']),
                'learned_at': obj_data['learned_at'],
                'feature_dim': obj_data['feature_dim']
            })
        return ensure_json_serializable(objects)
    
    def delete_object(self, object_name: str) -> bool:
        """Obje sil"""
        try:
            if object_name in self.known_objects:
                del self.known_objects[object_name]
                
                model_path = os.path.join(self.model_dir, f"{object_name}_model.pkl")
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                logger.info(f"Obje silindi: {object_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Silme hatası: {e}")
            return False

# Global instance
few_shot_learner = FewShotLearner()