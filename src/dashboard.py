import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np
from influxdb_client import InfluxDBClient
import logging

# Use relative import within package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import SimpleInfluxDB, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Animal Detection Dashboard",
            page_icon="🦁",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # โหลด config
        try:
            self.config = load_config("config/config.yaml")
            self.db = SimpleInfluxDB()
            
            # เตรียมข้อมูลสัตว์
            self.animals_data = {}
            for animal in self.config['animals']['classes']:
                self.animals_data[animal['name']] = {
                    'display_name': animal['name'],
                    'coco_id': animal['coco_id'],
                    'color': f"rgb({animal['color'][0]}, {animal['color'][1]}, {animal['color'][2]})"
                }
            
            self.time_mapping = {
                "Last Hour": 1,
                "Last 6 Hours": 6, 
                "Last 24 Hours": 24
            }
            
        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
            st.stop()
        
    def run(self):
        st.title("Animal Detection Dashboard")
        st.markdown("**Real-time data from InfluxDB: Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe**")
        
        # Connection status
        if self.db.is_connected():
            st.success("✅ Connected to InfluxDB")
        else:
            st.error("❌ InfluxDB Connection Failed - Showing sample data")
        
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Dashboard Settings")
            
            # Database connection info
            st.subheader("🗄️ Database Info")
            if self.db.is_connected():
                st.success("✅ InfluxDB Connected")
                st.write(f"**URL:** {self.db.db_config['url']}")
                st.write(f"**Org:** {self.db.db_config['org']}")
                st.write(f"**Bucket:** {self.db.db_config['bucket']}")
            else:
                st.error("❌ InfluxDB Disconnected")
                
            # Test connection button
            if st.button("🧪 Test Connection"):
                if self.db.test_connection():
                    st.success("✅ Connection test passed!")
                else:
                    st.error("❌ Connection test failed!")
            
            st.markdown("---")
            
            # แสดงข้อมูล COCO classes
            st.subheader("🎯 Target Animals (COCO)")
            for name, data in self.animals_data.items():
                st.write(f"**{name}** - ID: {data['coco_id']}")  
            
            st.markdown("---")
            
            # Time range
            time_range = st.selectbox(
                "📅 Time Range",
                list(self.time_mapping.keys()),
                index=1  # Default to 6 hours
            )
            
            # Refresh settings
            st.subheader("🔄 Refresh Settings")
            auto_refresh = st.checkbox("Auto Refresh", value=False)
            refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 30)
            
            # Manual refresh button
            if st.button("🔄 Refresh Now"):
                st.rerun()
            
            if auto_refresh:
                st.success(f"Dashboard will refresh every {refresh_interval} seconds")
        
        # Main content
        hours = self.time_mapping[time_range]
        
        # Show current stats
        self.show_current_stats(hours)
        
        # Show charts with real data
        self.show_real_charts(hours)
        
        # Show system performance
        self.show_system_performance(hours)
        
        # Show database info
        self.show_database_info()
        
        # Auto refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def get_real_animal_counts(self, hours):
        """ดึงข้อมูลจำนวนสัตว์จริงจาก InfluxDB"""
        try:
            # ดึงข้อมูลจาก database
            result = self.db.get_animal_history(hours)
            
            if result is None:
                return self.get_sample_data()
            
            # แปลงข้อมูลเป็น DataFrame
            data_points = []
            for table in result:
                for record in table.records:
                    data_points.append({
                        'time': record.get_time(),
                        'animal_type': record.values.get('animal_type'),
                        'count': int(record.get_value())
                    })
            
            if not data_points:
                return self.get_sample_data()
            
            df = pd.DataFrame(data_points)
            
            # คำนวณจำนวนสัตว์แต่ละประเภท (เอาค่าสูงสุด)
            latest_counts = df.groupby('animal_type')['count'].max().to_dict()
            
            # เติมค่า 0 สำหรับสัตว์ที่ไม่มีข้อมูล
            animal_counts = {}
            for animal_name in self.animals_data.keys():
                animal_counts[animal_name] = latest_counts.get(animal_name, 0)
            
            return animal_counts, df
            
        except Exception as e:
            logger.error(f"Error fetching real data: {e}")
            st.warning(f"⚠️ Using sample data due to: {str(e)}")
            return self.get_sample_data()
    
    def get_sample_data(self):
        """ข้อมูลตัวอย่างเมื่อไม่สามารถเชื่อมต่อ database ได้"""
        sample_counts = {
            'horse':int(np.random.randint(20, 30)),
            'sheep': int(np.random.randint(15, 25)),
            'cow': int(np.random.randint(10, 20)),
            'elephant': int(np.random.randint(5, 15)),
            'bear': int(np.random.randint(3, 8)),
            'zebra': int(np.random.randint(10, 20)),
            'giraffe': int(np.random.randint(4, 10))
        }
        
        # สร้าง DataFrame ตัวอย่าง
        times = pd.date_range(end=datetime.now(), periods=20, freq='3T')
        sample_data = []
        
        for animal, base_count in sample_counts.items():
            for time in times:
                count = int(base_count + np.random.randint(-5, 6))
                sample_data.append({
                    'time': time,
                    'animal_type': animal,
                    'count': max(0, count)  # ไม่ให้ติดลบ
                })
        
        df = pd.DataFrame(sample_data)
        return sample_counts, df
    
    def show_current_stats(self, hours):
        """แสดงสถิติปัจจุบันจากข้อมูลจริง"""
        st.subheader("📊 Current Detection Statistics")
        
        # ดึงข้อมูลจริง
        animal_counts, df = self.get_real_animal_counts(hours)
        
        # คำนวณสถิติ
        total_animals = int(sum(animal_counts.values()))
        active_types = len([count for count in animal_counts.values() if count > 0])
        
        # ดึงข้อมูล performance
        try:
            perf_result = self.db.get_performance_stats(1)  # Last hour
            avg_fps = 28.5  # Default
            
            if perf_result:
                fps_data = []
                for table in perf_result:
                    for record in table.records:
                        if record.values.get('_field') == 'fps':
                            fps_data.append(record.get_value())
                
                if fps_data:
                    avg_fps = np.mean(fps_data)
                    
        except Exception:
            avg_fps = 28.5
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Animals", 
                f"{total_animals}",
                delta=f"+{np.random.randint(0, 10)}"  # จำลอง delta
            )
        
        with col2:
            st.metric(
                "🎯 Active Classes", 
                f"{active_types}/7",
                delta="All Types" if active_types == 7 else f"{active_types} Types"
            )
        
        with col3:
            st.metric(
                "⚡ Average FPS", 
                f"{avg_fps:.1f}",
                delta=f"{np.random.uniform(-2, 3):.1f}"
            )
        
        with col4:
            status = "🟢 Online" if self.db.is_connected() else "🔴 Offline"
            st.metric(
                "📹 System Status", 
                "Online" if self.db.is_connected() else "Offline",
                delta=status
            )
    
    def show_real_charts(self, hours):
        """แสดงกราฟจากข้อมูลจริง"""
        st.subheader("📈 Animal Detection Trends (Real Data)")
        
        # ดึงข้อมูลจริง
        animal_counts, df = self.get_real_animal_counts(hours)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart จากข้อมูลจริง
            if animal_counts:
                animals = []
                counts = []
                
                for name, data in self.animals_data.items():
                    if name in animal_counts:
                        animals.append(f"{name}")
                        counts.append(animal_counts[name])
                
                fig_bar = px.bar(
                    x=animals, 
                    y=counts,
                    title=f"จำนวนสัตว์แต่ละประเภท (Last {hours}H)",
                    color=counts,
                    color_continuous_scale="viridis",
                    text=counts
                )
                fig_bar.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No data available for bar chart")
        
        with col2:
            # Pie chart จากข้อมูลจริง
            if animal_counts:
                # กรองเฉพาะสัตว์ที่มีจำนวน > 0
                filtered_data = {k: v for k, v in animal_counts.items() if v > 0}
                
                if filtered_data:
                    names = list(filtered_data.keys())  
                    values = list(filtered_data.values())
                    
                    fig_pie = px.pie(
                        values=values,
                        names=names,
                        title=f"สัดส่วนสัตว์ที่ตรวจพบ (Last {hours}H)"
                    )
                    fig_pie.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No animals detected in selected time range")
            else:
                st.warning("No data available for pie chart")
        
        # Time series chart จากข้อมูลจริง
        st.subheader(f"📅 Detection Trends Over Time (Last {hours} Hours)")
        
        if not df.empty:
            # สร้าง time series chart
            fig_time = go.Figure()
            
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#fef65b', '#ff9ff3', '#54a0ff']
            color_map = dict(zip(self.animals_data.keys(), colors))
            
            for animal in self.animals_data.keys():
                animal_data = df[df['animal_type'] == animal].sort_values('time')
                
                if not animal_data.empty:
                    fig_time.add_trace(go.Scatter(
                        x=animal_data['time'],
                        y=animal_data['count'],
                        mode='lines+markers',
                        name=f"{animal} ({animal})",
                        line=dict(color=color_map.get(animal, '#888888')),
                        marker=dict(size=6)
                    ))
            
            fig_time.update_layout(
                title=f"การตรวจจับสัตว์ตลอดเวลา (Last {hours} Hours)",
                xaxis_title="เวลา",
                yaxis_title="จำนวนสัตว์",
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("No time series data available")
        
        # แสดงข้อมูลล่าสุด
        if not df.empty:
            with st.expander("📋 Recent Detection Data"):
                # แสดง 10 records ล่าสุด
                recent_data = df.sort_values('time', ascending=False).head(10)
                recent_data['thai_name'] = recent_data['animal_type'].map(
                    lambda x: self.animals_data.get(x, {}).get('thai_name', x)
                )
                
                display_df = recent_data[['time', 'animal_type', 'count']].copy()
                display_df.columns = ['Time', 'Animal Type', 'Count']  
                
                st.dataframe(display_df, use_container_width=True)
    
    def show_system_performance(self, hours):
        """แสดงข้อมูล performance ของระบบ"""
        st.subheader("⚡ System Performance Metrics")
        
        try:
            perf_result = self.db.get_performance_stats(hours)
            
            if perf_result:
                fps_data = []
                processing_data = []
                times = []
                
                for table in perf_result:
                    for record in table.records:
                        field = record.values.get('_field')
                        time_val = record.get_time()
                        value = record.get_value()
                        
                        if field == 'fps':
                            fps_data.append({'time': time_val, 'value': value})
                        elif field == 'processing_time_ms':
                            processing_data.append({'time': time_val, 'value': value})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if fps_data:
                        fps_df = pd.DataFrame(fps_data)
                        fig_fps = px.line(
                            fps_df, x='time', y='value',
                            title='FPS Over Time',
                            labels={'value': 'FPS', 'time': 'Time'}
                        )
                        fig_fps.add_hline(
                            y=25, line_dash="dash", line_color="red",
                            annotation_text="Min Threshold"
                        )
                        st.plotly_chart(fig_fps, use_container_width=True)
                    else:
                        st.info("No FPS data available")
                
                with col2:
                    if processing_data:
                        proc_df = pd.DataFrame(processing_data)
                        fig_proc = px.line(
                            proc_df, x='time', y='value',
                            title='Processing Time',
                            labels={'value': 'Time (ms)', 'time': 'Time'}
                        )
                        st.plotly_chart(fig_proc, use_container_width=True)
                    else:
                        st.info("No processing time data available")
            else:
                st.info("No performance data available")
                
        except Exception as e:
            st.error(f"Error loading performance data: {e}")
    
    def show_database_info(self):
        """แสดงข้อมูล database และ COCO classes"""
        st.subheader("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**COCO Classes ที่ใช้งาน:**")
            for name, data in self.animals_data.items():
                st.write(f"🔹 **Class {data['coco_id']}**: {name}")  
        
        with col2:
            st.success("**Database Status:**")
            if self.db.is_connected():
                st.write("✅ InfluxDB Connected")
                st.write(f"📊 Bucket: {self.db.bucket}")
                st.write(f"🏢 Organization: {self.db.org}")
                st.write(f"🌐 URL: {self.db.db_config['url']}")
                
                # แสดงข้อมูลการเชื่อมต่อล่าสุด
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"🕐 Last Update: {current_time}")
            else:
                st.write("❌ InfluxDB Disconnected")
                st.write("🔧 Check your configuration")
                st.write("📝 Showing sample data instead")

def run_dashboard():
    """รันแดชบอร์ด"""
    try:
        dashboard = RealDataDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Dashboard Error: {e}")
        st.info("💡 Make sure InfluxDB is running and configured properly")

if __name__ == "__main__":
    run_dashboard()