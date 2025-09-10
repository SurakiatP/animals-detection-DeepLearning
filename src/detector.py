import cv2
import yaml
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np
from datetime import datetime

class SimpleAnimalDetector:
    def __init__(self, config_path="config/config.yaml"):
        # load configuration
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # load model
        self.model = YOLO(self.config['model']['path'])
        self.confidence_threshold = self.config['model']['confidence_threshold']
        
        # build animal mapping from COCO classes
        self.animal_classes = {}
        self.colors = {}
        self.coco_ids = set()
        
        for animal in self.config['animals']['classes']:
            name = animal['name']
            self.animal_classes[name] = animal['name']
            self.colors[name] = tuple(animal['color'])
            self.coco_ids.add(animal['coco_id'])

        print(f"🦁 Loaded {len(self.animal_classes)} animal classes:")
        for name, display_name in self.animal_classes.items():
            print(f"   - {name}")

        # Variables for counting animals
        self.current_counts = defaultdict(int)
        self.max_counts = defaultdict(int)
        
    def detect_frame(self, frame):
        """Detect animals in 1 frame"""
        results = self.model(frame, conf=self.confidence_threshold)
        
        detections = []
        frame_counts = defaultdict(int)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Check if it is the animal we want. (COCO ID 17-23)
                    if class_id in self.coco_ids and class_name in self.animal_classes:
                        # Count the animals in the frame
                        frame_counts[class_name] += 1
                        
                        # Collect detection data
                        detections.append({
                            'class_name': class_name,
                            'display_name': self.animal_classes[class_name],
                            'confidence': confidence,
                            'coco_id': class_id,
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.now()
                        })

        # Update maximum number
        for animal, count in frame_counts.items():
            self.max_counts[animal] = max(self.max_counts[animal], count)
        
        self.current_counts = frame_counts
        return detections, dict(frame_counts)
    
    def draw_detections(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            display_name = detection['display_name']
            confidence = detection['confidence']
            coco_id = detection['coco_id']
        
            color = self.colors.get(class_name, (255, 255, 255))
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
            label = f"{class_name}: {confidence:.1%}"
        
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
            id_label = f"COCO ID: {coco_id}"
            cv2.putText(frame, id_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
        return frame

    def draw_statistics(self, frame):
        stats_height = 60 + len(self.animal_classes) * 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
        cv2.putText(frame, "Animal Detection", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Animals: horse, sheep, cow, elephant, bear, zebra, giraffe", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
        y_pos = 75
        total_current = 0
        total_max = 0
    
        for animal, display_name in self.animal_classes.items():
            current = self.current_counts.get(animal, 0)
            maximum = self.max_counts.get(animal, 0)
            color = self.colors[animal]
        
            total_current += current
            total_max += maximum
        
            text = f"{animal}: {current} (Max: {maximum})"
            cv2.putText(frame, text, (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_pos += 25
    
        cv2.putText(frame, f"Total: {total_current} (Max Total: {total_max})", (20, y_pos + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        return frame
    
    def get_summary(self):
        """Return animal count summary"""
        return {
            'current_counts': dict(self.current_counts),
            'max_counts': dict(self.max_counts),
            'total_animals': sum(self.current_counts.values()),
            'total_max': sum(self.max_counts.values()),
            'animal_types': len(self.animal_classes)
        }