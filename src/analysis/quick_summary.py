#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để chạy báo cáo nhanh
"""

from data_analysis_report import DataAnalysisReport
import pandas as pd

def quick_summary():
    """Báo cáo tóm tắt nhanh"""
    print("🚀 BÁO CÁO NHANH - DỮ LIỆU COMMENT YOUTUBE")
    print("=" * 60)
    
    analyzer = DataAnalysisReport()
    analyzer.load_data()
    
    # Thống kê nhanh
    total_raw = sum(len(df) for df in analyzer.raw_data.values())
    total_processed = sum(len(df) for df in analyzer.processed_data.values())
    
    print(f"📊 TỔNG KẾT:")
    print(f"  • Raw data:       {total_raw:,} comments")
    print(f"  • Processed data: {total_processed:,} comments")
    print(f"  • Loại bỏ:        {total_raw - total_processed:,} comments ({(total_raw - total_processed)/total_raw*100:.1f}%)")
    print(f"  • Giữ lại:       {total_processed/total_raw*100:.1f}%")
    
    if hasattr(analyzer, 'combined_data'):
        combined_file = analyzer.base_path / "combined_processed.csv"
        if combined_file.exists():
            combined = pd.read_csv(combined_file)
            print(f"  • Combined data:  {len(combined):,} records")
            
            # Độ dài trung bình
            combined['text_length'] = combined['text'].astype(str).str.len()
            avg_length = combined['text_length'].mean()
            print(f"  • Độ dài TB:      {avg_length:.1f} ký tự")
    
    print("\n📹 TOP 3 VIDEO CÓ NHIỀU COMMENT NHẤT (sau xử lý):")
    video_counts = [(name, len(df)) for name, df in analyzer.processed_data.items()]
    video_counts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (video, count) in enumerate(video_counts[:3], 1):
        print(f"  {i}. {video:<20}: {count:,} comments")
    
    print(f"\n✨ Hoàn thành! Để xem báo cáo chi tiết, chạy: python run_report.py")

if __name__ == "__main__":
    try:
        quick_summary()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Hãy đảm bảo dữ liệu tồn tại trong thư mục data/")