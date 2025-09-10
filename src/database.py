import yaml
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
import pytz
import os

THAILAND_TZ = pytz.timezone('Asia/Bangkok')

class SimpleInfluxDB:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize InfluxDB connection"""
        # โหลด configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        self.db_config = config['influxdb']
        
        # สร้าง InfluxDB client
        try:
            self.client = InfluxDBClient(
                url=self.db_config['url'],
                token=self.db_config['token'],
                org=self.db_config['org']
            )
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.bucket = self.db_config['bucket']
            self.org = self.db_config['org']
            
            print(f"InfluxDB connected: {self.db_config['url']}")
            
        except Exception as e:
            print(f"InfluxDB connection failed: {e}")
            print("Make sure InfluxDB is running and config is correct")
            self.client = None
    
    def is_connected(self):
        """Check if InfluxDB is connected"""
        return self.client is not None
    
    def save_animal_counts(self, animal_counts, source="camera_1", location="detection_area"):
        """บันทึกจำนวนสัตว์แต่ละประเภท"""
        if not self.is_connected():
            print("InfluxDB not connected, skipping save")
            return False
            
        if not animal_counts:
            return True
        
        timestamp = datetime.now(THAILAND_TZ)
        timestamp_utc = timestamp.astimezone(timezone.utc)
        
        points = []
        
        try:
            # บันทึกจำนวนแต่ละประเภทสัตว์
            for animal_type, count in animal_counts.items():
                point = Point("animal_counts") \
                    .tag("animal_type", animal_type) \
                    .tag("source", source) \
                    .tag("location", location) \
                    .field("count", int(count)) \
                    .field("detection_time", timestamp.isoformat()) \
                    .time(timestamp)
                points.append(point)
            
            # บันทึกจำนวนรวมทั้งหมด
            total_count = sum(animal_counts.values())
            total_point = Point("total_animals") \
                .tag("source", source) \
                .tag("location", location) \
                .field("total_count", int(total_count)) \
                .field("unique_types", len(animal_counts)) \
                .time(timestamp)
            points.append(total_point)
            
            # เขียนข้อมูลลง database
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            
            return True
            
        except Exception as e:
            print(f"Database save error: {e}")
            return False
    
    def save_detection_details(self, detections, source="camera_1", location="detection_area"):
        """บันทึกรายละเอียดการตรวจจับแต่ละตัว"""
        if not self.is_connected() or not detections:
            return False
        
        timestamp = datetime.now(timezone.utc)
        points = []
        
        try:
            for i, detection in enumerate(detections):
                point = Point("detection_details") \
                    .tag("animal_type", detection['class_name']) \
                    .tag("source", source) \
                    .tag("location", location) \
                    .field("confidence", float(detection['confidence'])) \
                    .field("coco_id", int(detection['coco_id'])) \
                    .field("bbox_x1", int(detection['bbox'][0])) \
                    .field("bbox_y1", int(detection['bbox'][1])) \
                    .field("bbox_x2", int(detection['bbox'][2])) \
                    .field("bbox_y2", int(detection['bbox'][3])) \
                    .field("detection_id", i) \
                    .time(timestamp)
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            return True
            
        except Exception as e:
            print(f"Detection details save error: {e}")
            return False
    
    def save_system_performance(self, fps, processing_time, frame_count, source="camera_1"):
        """บันทึกข้อมูล performance ของระบบ"""
        if not self.is_connected():
            return False
        
        try:
            point = Point("system_performance") \
                .tag("source", source) \
                .field("fps", float(fps)) \
                .field("processing_time_ms", float(processing_time * 1000)) \
                .field("frame_count", int(frame_count)) \
                .field("timestamp", datetime.now(timezone.utc).isoformat()) \
                .time(datetime.now(timezone.utc))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
            
        except Exception as e:
            print(f"Performance data save error: {e}")
            return False
    
    def get_animal_history(self, hours=1, animal_type=None):
        """ดึงประวัติการนับสัตว์"""
        if not self.is_connected():
            return None
        
        try:
            if animal_type:
                query = f'''
                from(bucket: "{self.bucket}")
                  |> range(start: -{hours}h)
                  |> filter(fn: (r) => r["_measurement"] == "animal_counts")
                  |> filter(fn: (r) => r["animal_type"] == "{animal_type}")
                  |> filter(fn: (r) => r["_field"] == "count")
                  |> aggregateWindow(every: 2m, fn: mean, createEmpty: false)
                  |> yield(name: "mean")
                '''
            else:
                query = f'''
                from(bucket: "{self.bucket}")
                  |> range(start: -{hours}h)
                  |> filter(fn: (r) => r["_measurement"] == "animal_counts")
                  |> filter(fn: (r) => r["_field"] == "count")
                  |> group(columns: ["animal_type"])
                  |> aggregateWindow(every: 2m, fn: mean, createEmpty: false)
                  |> yield(name: "mean")
                '''
            
            result = self.query_api.query(org=self.org, query=query)
            return result
            
        except Exception as e:
            print(f"Query error: {e}")
            return None
    
    def get_total_history(self, hours=1):
        """ดึงประวัติจำนวนสัตว์รวม"""
        if not self.is_connected():
            return None
        
        try:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -{hours}h)
              |> filter(fn: (r) => r["_measurement"] == "total_animals")
              |> filter(fn: (r) => r["_field"] == "total_count")
              |> aggregateWindow(every: 2m, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            
            result = self.query_api.query(org=self.org, query=query)
            return result
            
        except Exception as e:
            print(f"Total query error: {e}")
            return None
    
    def get_detection_summary(self, hours=24):
        """ดึงสรุปการตรวจจับใน x ชั่วโมงที่ผ่านมา"""
        if not self.is_connected():
            return None
        
        try:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -{hours}h)
              |> filter(fn: (r) => r["_measurement"] == "detection_details")
              |> group(columns: ["animal_type"])
              |> count()
              |> yield(name: "count")
            '''
            
            result = self.query_api.query(org=self.org, query=query)
            return result
            
        except Exception as e:
            print(f"Summary query error: {e}")
            return None
    
    def get_performance_stats(self, hours=1):
        """ดึงสถิติ performance ของระบบ"""
        if not self.is_connected():
            return None
        
        try:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -{hours}h)
              |> filter(fn: (r) => r["_measurement"] == "system_performance")
              |> filter(fn: (r) => r["_field"] == "fps" or r["_field"] == "processing_time_ms")
              |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
              |> yield(name: "mean")
            '''
            
            result = self.query_api.query(org=self.org, query=query)
            return result
            
        except Exception as e:
            print(f"Performance query error: {e}")
            return None
    
    def test_connection(self):
        """ทดสอบการเชื่อมต่อ database"""
        if not self.is_connected():
            return False
        
        try:
            # ลองเขียนข้อมูลทดสอบ
            test_point = Point("connection_test") \
                .field("test_value", 1) \
                .time(datetime.now(timezone.utc))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=test_point)
            print("Database connection test successful")
            return True
            
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False
    
    def close(self):
        """ปิดการเชื่อมต่อ"""
        if self.client:
            self.client.close()
            print(" InfluxDB connection closed")

# Utility function for easy import
def create_database(config_path="config/config.yaml"):
    """Factory function to create database instance"""
    return SimpleInfluxDB(config_path)

# Test function
def test_database_connection():
    """Function to test database connection"""
    db = create_database()
    return db.test_connection() if db.is_connected() else False

if __name__ == "__main__":
    # Test the database connection
    print("Testing InfluxDB connection...")
    test_database_connection()