import cv2
import time
import yaml
import os
import sys

# เพิ่ม project root path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import จาก src package
from src import SimpleAnimalDetector, SimpleInfluxDB, load_config

class COCOAnimalDetectionApp:
    def __init__(self, config_path="config/config.yaml"):
        # โหลด configuration
        self.config = load_config(config_path)
        
        # สร้าง components
        self.detector = SimpleAnimalDetector(config_path)
        self.database = SimpleInfluxDB(config_path)
        
        # ตัวแปรสำหรับ video
        self.video_source = self.config['video']['source']
        self.save_output = self.config['video']['save_output']
        self.output_path = self.config['video']['output_path']
        
        # ตัวแปรสำหรับ performance
        self.frame_count = 0
        self.last_save_time = time.time()
        
        print("COCO Animal Detection System Initialized")
        print(f"Video Source: {self.video_source}")
        print(f"Target Animals: {len(self.detector.animal_classes)} classes")
        
    def run(self):
        """เริ่มการตรวจจับสัตว์"""
        print("\nStarting COCO Animal Detection System...")
        print("Detecting: Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe")
        print("Press 'q' to quit, 's' to save current stats, 'i' to show info")
        
        # เปิด video source
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # ตั้งค่า video writer (ถ้าต้องการบันทึก)
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # สร้างโฟลเดอร์ output ถ้าไม่มี
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            print(f"Recording to: {self.output_path}")
        
        # Performance tracking
        fps_history = []
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print(" End of video stream")
                    break
                
                # ตรวจจับสัตว์
                detections, frame_counts = self.detector.detect_frame(frame)
                
                # วาดผลลัพธ์
                frame = self.detector.draw_detections(frame, detections)
                frame = self.detector.draw_statistics(frame)
                
                # คำนวณและแสดง FPS
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                fps_history.append(fps)
                
                if len(fps_history) > 30:  # เก็บ 30 frames ล่าสุด
                    fps_history.pop(0)
                
                avg_fps = sum(fps_history) / len(fps_history)
                
                # แสดง FPS และข้อมูลเพิ่มเติม
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                           (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {self.frame_count}", 
                           (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Detections: {len(detections)}", 
                           (frame.shape[1] - 150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # บันทึกข้อมูลลง database ทุก 15 วินาที
                current_time = time.time()
                if current_time - self.last_save_time >= 15:
                    if frame_counts:
                        self.database.save_animal_counts(frame_counts)
                        print(f"Saved to database: {frame_counts}")
                    self.last_save_time = current_time
                
                # แสดงผล
                cv2.imshow('COCO Animal Detection System', frame)
                
                # บันทึก video output
                if self.save_output:
                    out.write(frame)
                
                # จัดการ keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested")
                    break
                elif key == ord('s'):
                    self.save_stats()
                elif key == ord('i'):
                    self.show_detection_info(detections)
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            # ปิดทุกอย่าง
            cap.release()
            if self.save_output:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"Processed {self.frame_count} frames")
            print(f"Average FPS: {sum(fps_history) / len(fps_history):.2f}")
            self.save_stats()
    
    def save_stats(self):
        """บันทึกและแสดงสถิติ"""
        summary = self.detector.get_summary()
        
        print("\nCOCO Animal Detection Summary:")
        print("=" * 50)
        print(f"Target Classes: COCO ID 17-23 (7 animals)")
        print(f"Animal Types Detected: {summary['animal_types']}")
        print("-" * 50)
        
        for animal, display_name in self.detector.animal_classes.items():
            current = summary['current_counts'].get(animal, 0)
            maximum = summary['max_counts'].get(animal, 0)
    
            coco_id = next((item['coco_id'] for item in self.config['animals']['classes'] 
                   if item['name'] == animal), '?')
    
        print(f"{animal} [COCO {coco_id}]: {current} (Max: {maximum})")  
        
        print("-" * 50)
        print(f"Total Current: {summary['total_animals']}")
        print(f"Max Total Ever: {summary['total_max']}")
        print(f"  Frames Processed: {self.frame_count}")
        print("=" * 50)
    
    def show_detection_info(self, detections):
        """แสดงข้อมูลการตรวจจับปัจจุบัน"""
        if not detections:
            print("  No animals detected in current frame")
            return
        
        print(f"\n Current Frame Detections ({len(detections)} animals):")
        print("-" * 60)
        for i, detection in enumerate(detections, 1):
            print(f"{i}. {detection['class_name']} "
                f"- Confidence: {detection['confidence']:.1%} "
                f"- COCO ID: {detection['coco_id']}")  
        print("-" * 60)

if __name__ == "__main__":
    app = COCOAnimalDetectionApp()
    app.run()