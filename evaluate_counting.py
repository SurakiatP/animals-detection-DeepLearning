# evaluate_counting.py
import cv2
import json
import os
from collections import defaultdict
from src import SimpleAnimalDetector, load_config

class CountingEvaluator:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.detector = SimpleAnimalDetector(config_path)
        self.results = []
    
    def evaluate_video(self, video_path, ground_truth_counts):
        """ประเมินความแม่นยำการนับในวิดีโอ"""
        print(f"\nEvaluating: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # ตัวแปรเก็บจำนวนสูงสุด
        max_counts = defaultdict(int)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            detections, counts = self.detector.detect_frame(frame)
            
            # เก็บค่าสูงสุด
            for animal, count in counts.items():
                max_counts[animal] = max(max_counts[animal], count)
            
            frame_count += 1
        
        cap.release()
        
        # คำนวณ metrics
        result = self.calculate_metrics(
            dict(max_counts), 
            ground_truth_counts, 
            video_path
        )
        
        self.results.append(result)
        return result
    
    def calculate_metrics(self, predicted, ground_truth, video_name):
        """คำนวณ MAE, MAPE, Accuracy"""
        
        # รวม animals ทั้งหมดที่มีใน GT
        all_animals = set(ground_truth.keys())
        
        errors = []
        percentage_errors = []
        correct = 0
        total = 0
        
        for animal in all_animals:
            pred_count = predicted.get(animal, 0)
            gt_count = ground_truth[animal]
            
            # MAE
            error = abs(pred_count - gt_count)
            errors.append(error)
            
            # MAPE (เฉพาะ GT > 0)
            if gt_count > 0:
                percentage_errors.append((error / gt_count) * 100)
            
            # Exact match
            if pred_count == gt_count:
                correct += 1
            total += 1
        
        mae = sum(errors) / len(errors) if errors else 0
        mape = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'video': os.path.basename(video_name),
            'predicted': predicted,
            'ground_truth': ground_truth,
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'accuracy': round(accuracy, 2),
            'correct_counts': correct,
            'total_classes': total
        }
    
    def evaluate_all(self, ground_truth_file):
        """ประเมินทั้งหมดจาก ground truth file"""
        
        # โหลด ground truth
        with open(ground_truth_file, 'r') as f:
            all_ground_truths = json.load(f)
        
        print("="*60)
        print("COUNTING ACCURACY EVALUATION")
        print("="*60)
        
        # ประเมินแต่ละวิดีโอ
        for video_path, gt_counts in all_ground_truths.items():
            # ถ้าเป็น relative path
            if not os.path.isabs(video_path):
                video_path = os.path.join('data/videos', video_path)
            
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            
            self.evaluate_video(video_path, gt_counts)
        
        # แสดงสรุปผล
        self.print_summary()
        
        # บันทึกผลลัพธ์
        self.save_results()
    
    def print_summary(self):
        """แสดงสรุปผลลัพธ์"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(f"\n Video: {result['video']}")
            print(f"   MAE:      {result['mae']:.2f} animals")
            print(f"   MAPE:     {result['mape']:.2f}%")
            print(f"   Accuracy: {result['accuracy']:.2f}% ({result['correct_counts']}/{result['total_classes']} exact)")
            
            print("\n   Details:")
            for animal in result['ground_truth'].keys():
                pred = result['predicted'].get(animal, 0)
                gt = result['ground_truth'][animal]
                diff = pred - gt
                status = "✓" if diff == 0 else "✗"
                print(f"     {status} {animal:10s}: Predicted={pred:2d}, Ground Truth={gt:2d}, Diff={diff:+3d}")
        
        # คำนวณค่าเฉลี่ยทั้งหมด
        avg_mae = sum(r['mae'] for r in self.results) / len(self.results)
        avg_mape = sum(r['mape'] for r in self.results) / len(self.results)
        avg_accuracy = sum(r['accuracy'] for r in self.results) / len(self.results)
        
        print("\n" + "="*60)
        print("OVERALL METRICS")
        print("="*60)
        print(f"Average MAE:      {avg_mae:.2f} animals")
        print(f"Average MAPE:     {avg_mape:.2f}%")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print("="*60)
    
    def save_results(self, output_file="evaluation/counting_results.json"):
        """บันทึกผลลัพธ์"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Counting Accuracy')
    parser.add_argument('--ground-truth', type=str, 
                       default='evaluation/ground_truth_counts.json',
                       help='Path to ground truth counts JSON')
    
    args = parser.parse_args()
    
    # รันการประเมินผล
    evaluator = CountingEvaluator()
    evaluator.evaluate_all(args.ground_truth)