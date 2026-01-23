#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Báo cáo phân tích dữ liệu comment YouTube trước và sau khi xử lý
Author: Data Analysis Team
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from pathlib import Path

# Set up matplotlib để hiển thị tiếng Việt
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class DataAnalysisReport:
    def __init__(self, base_path="../../data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path
        
        # Danh sách các file dữ liệu
        self.video_files = [
            "eLcYOrgGl-A",
            "IzSYlr3VI1A", 
            "QBvL3Ffn1u4",
            "qGRU3sRbaYw",
            "x5CNJUMJcII",
            "Ym6h7ZLtdhE"
        ]
        
    def load_data(self):
        """Load dữ liệu raw và processed"""
        print("🔄 Đang load dữ liệu...")
        
        self.raw_data = {}
        self.processed_data = {}
        
        # Load raw data
        for video in self.video_files:
            try:
                raw_file = self.raw_path / f"{video}.csv"
                if raw_file.exists():
                    self.raw_data[video] = pd.read_csv(raw_file)
                    print(f"  ✓ Loaded raw data: {video}.csv ({len(self.raw_data[video])} records)")
            except Exception as e:
                print(f"  ❌ Error loading {video}.csv: {e}")
        
        # Load processed data
        for video in self.video_files:
            try:
                processed_file = self.processed_path / f"{video}_processed.csv"
                if processed_file.exists():
                    self.processed_data[video] = pd.read_csv(processed_file)
                    print(f"  ✓ Loaded processed data: {video}_processed.csv ({len(self.processed_data[video])} records)")
            except Exception as e:
                print(f"  ❌ Error loading {video}_processed.csv: {e}")
        
        # Load combined data
        try:
            combined_file = self.processed_path / "combined_processed.csv"
            if combined_file.exists():
                self.combined_data = pd.read_csv(combined_file)
                print(f"  ✓ Loaded combined data: combined_processed.csv ({len(self.combined_data)} records)")
        except Exception as e:
            print(f"  ❌ Error loading combined_processed.csv: {e}")
            
        print("✅ Load dữ liệu hoàn tất!\n")

    def analyze_text_length(self, text):
        """Phân tích độ dài text"""
        if pd.isna(text):
            return 0
        return len(str(text))
    
    def analyze_word_count(self, text):
        """Đếm số từ"""
        if pd.isna(text):
            return 0
        return len(str(text).split())
    
    def detect_spam_keywords(self, text):
        """Phát hiện spam keywords (đơn giản)"""
        if pd.isna(text):
            return False
        
        spam_patterns = [
            r'like.*subscribe', r'sub.*back', r'view.*tăng', 
            r'quảng.*cáo', r'bán.*hàng', r'liên.*hệ.*mua',
            r'http[s]?://', r'www\.', r'\.com'
        ]
        
        text_lower = str(text).lower()
        for pattern in spam_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def generate_basic_statistics(self):
        """Tạo thống kê cơ bản"""
        print("📊 THỐNG KÊ CƠ BẢN")
        print("=" * 60)
        
        # Tổng hợp số liệu từng video
        raw_stats = []
        processed_stats = []
        
        for video in self.video_files:
            if video in self.raw_data:
                raw_df = self.raw_data[video]
                raw_count = len(raw_df)
                raw_avg_length = raw_df['text'].apply(self.analyze_text_length).mean()
                raw_avg_words = raw_df['text'].apply(self.analyze_word_count).mean()
                raw_spam_count = raw_df['text'].apply(self.detect_spam_keywords).sum()
                
                raw_stats.append({
                    'video': video,
                    'count': raw_count,
                    'avg_length': raw_avg_length,
                    'avg_words': raw_avg_words,
                    'spam_count': raw_spam_count,
                    'spam_rate': (raw_spam_count / raw_count * 100) if raw_count > 0 else 0
                })
            
            if video in self.processed_data:
                proc_df = self.processed_data[video]
                proc_count = len(proc_df)
                proc_avg_length = proc_df['text'].apply(self.analyze_text_length).mean()
                proc_avg_words = proc_df['text'].apply(self.analyze_word_count).mean()
                proc_spam_count = proc_df['text'].apply(self.detect_spam_keywords).sum()
                
                processed_stats.append({
                    'video': video,
                    'count': proc_count,
                    'avg_length': proc_avg_length,
                    'avg_words': proc_avg_words,
                    'spam_count': proc_spam_count,
                    'spam_rate': (proc_spam_count / proc_count * 100) if proc_count > 0 else 0
                })
        
        # Hiển thị bảng thống kê
        print("\n📈 TRƯỚC KHI XỬ LÝ (Raw Data):")
        print("-" * 100)
        print(f"{'Video':<20} {'Số comments':<12} {'Độ dài TB':<12} {'Số từ TB':<12} {'Spam':<8} {'Tỷ lệ spam':<12}")
        print("-" * 100)
        
        total_raw = 0
        total_spam_raw = 0
        
        for stat in raw_stats:
            total_raw += stat['count']
            total_spam_raw += stat['spam_count']
            print(f"{stat['video']:<20} {stat['count']:<12} {stat['avg_length']:<12.1f} "
                  f"{stat['avg_words']:<12.1f} {stat['spam_count']:<8} {stat['spam_rate']:<12.2f}%")
        
        print("-" * 100)
        total_spam_rate_raw = (total_spam_raw / total_raw * 100) if total_raw > 0 else 0
        print(f"{'TỔNG':<20} {total_raw:<12} {'-':<12} {'-':<12} {total_spam_raw:<8} {total_spam_rate_raw:<12.2f}%")
        
        print("\n📈 SAU KHI XỬ LÝ (Processed Data):")
        print("-" * 100)
        print(f"{'Video':<20} {'Số comments':<12} {'Độ dài TB':<12} {'Số từ TB':<12} {'Spam':<8} {'Tỷ lệ spam':<12}")
        print("-" * 100)
        
        total_processed = 0
        total_spam_processed = 0
        
        for stat in processed_stats:
            total_processed += stat['count']
            total_spam_processed += stat['spam_count']
            print(f"{stat['video']:<20} {stat['count']:<12} {stat['avg_length']:<12.1f} "
                  f"{stat['avg_words']:<12.1f} {stat['spam_count']:<8} {stat['spam_rate']:<12.2f}%")
        
        print("-" * 100)
        total_spam_rate_processed = (total_spam_processed / total_processed * 100) if total_processed > 0 else 0
        print(f"{'TỔNG':<20} {total_processed:<12} {'-':<12} {'-':<12} {total_spam_processed:<8} {total_spam_rate_processed:<12.2f}%")
        
        # So sánh trước và sau
        print("\n📊 SO SÁNH TRƯỚC VÀ SAU KHI XỬ LÝ:")
        print("-" * 60)
        print(f"Tổng comment raw:        {total_raw:,}")
        print(f"Tổng comment processed:  {total_processed:,}")
        print(f"Số comment bị loại bỏ:   {total_raw - total_processed:,}")
        print(f"Tỷ lệ giữ lại:          {(total_processed/total_raw*100):.2f}%" if total_raw > 0 else "N/A")
        print(f"Spam trước xử lý:       {total_spam_raw} ({total_spam_rate_raw:.2f}%)")
        print(f"Spam sau xử lý:         {total_spam_processed} ({total_spam_rate_processed:.2f}%)")
        
        return raw_stats, processed_stats

    def analyze_text_quality(self):
        """Phân tích chất lượng văn bản"""
        print("\n\n🔍 PHÂN TÍCH CHẤT LƯỢNG VẦN BẢN")
        print("=" * 60)
        
        if hasattr(self, 'combined_data'):
            df = self.combined_data
            
            # Phân tích độ dài
            df['text_length'] = df['text'].apply(self.analyze_text_length)
            df['word_count'] = df['text'].apply(self.analyze_word_count)
            
            print(f"\n📏 THỐNG KÊ ĐỘ DÀI (Combined processed data):")
            print(f"Số comment:              {len(df):,}")
            print(f"Độ dài trung bình:       {df['text_length'].mean():.1f} ký tự")
            print(f"Độ dài median:           {df['text_length'].median():.1f} ký tự")
            print(f"Số từ trung bình:        {df['word_count'].mean():.1f} từ")
            print(f"Số từ median:            {df['word_count'].median():.1f} từ")
            
            print(f"\n📊 PHÂN BỐ ĐỘ DÀI:")
            length_ranges = [
                (0, 20, "Rất ngắn"),
                (21, 50, "Ngắn"),
                (51, 100, "Trung bình"),
                (101, 200, "Dài"),
                (201, float('inf'), "Rất dài")
            ]
            
            for min_len, max_len, category in length_ranges:
                if max_len == float('inf'):
                    count = len(df[df['text_length'] > min_len])
                else:
                    count = len(df[(df['text_length'] >= min_len) & (df['text_length'] <= max_len)])
                percentage = count / len(df) * 100
                print(f"{category:<12}: {count:>6,} ({percentage:>5.1f}%)")
            
            # Top từ khóa xuất hiện nhiều nhất
            print(f"\n🔥 TOP 20 TỪ XUẤT HIỆN NHIỀU NHẤT:")
            all_text = ' '.join(df['text'].astype(str))
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter(words)
            
            for i, (word, freq) in enumerate(word_freq.most_common(20), 1):
                print(f"{i:>2}. {word:<15}: {freq:>6,} lần")

    def analyze_video_distribution(self):
        """Phân tích phân bố theo video"""
        print("\n\n📹 PHÂN BỐ THEO VIDEO")
        print("=" * 60)
        
        if hasattr(self, 'combined_data'):
            source_counts = self.combined_data['source'].value_counts()
            
            print("\n📊 SỐ LƯỢNG COMMENT THEO VIDEO:")
            print("-" * 50)
            
            total = len(self.combined_data)
            for source, count in source_counts.items():
                video_name = source.replace('_processed', '')
                percentage = count / total * 100
                print(f"{video_name:<20}: {count:>6,} ({percentage:>5.1f}%)")
            
            print("-" * 50)
            print(f"{'TỔNG':<20}: {total:>6,} (100.0%)")

    def detect_preprocessing_changes(self):
        """Phát hiện những thay đổi do preprocessing"""
        print("\n\n🔧 PHÂN TÍCH THAY ĐỔI DO PREPROCESSING")
        print("=" * 60)
        
        changes_detected = {
            'underscores_added': 0,
            'emojis_processed': 0,
            'length_reduced': 0,
            'records_removed': 0
        }
        
        for video in self.video_files:
            if video in self.raw_data and video in self.processed_data:
                raw_df = self.raw_data[video]
                proc_df = self.processed_data[video]
                
                print(f"\n📝 {video}:")
                print(f"  Raw records:        {len(raw_df):,}")
                print(f"  Processed records:  {len(proc_df):,}")
                print(f"  Records removed:    {len(raw_df) - len(proc_df):,}")
                
                changes_detected['records_removed'] += len(raw_df) - len(proc_df)
                
                # Phân tích một số mẫu để xem thay đổi
                if len(proc_df) > 0:
                    sample_texts = proc_df['text'].head(5)
                    underscores = sum(1 for text in sample_texts if '_' in str(text))
                    changes_detected['underscores_added'] += underscores
                    
                    # Tính toán độ dài trung bình
                    if len(raw_df) > 0:
                        raw_avg_len = raw_df['text'].apply(self.analyze_text_length).mean()
                        proc_avg_len = proc_df['text'].apply(self.analyze_text_length).mean()
                        
                        if proc_avg_len < raw_avg_len:
                            changes_detected['length_reduced'] += 1
                        
                        print(f"  Average length change: {raw_avg_len:.1f} → {proc_avg_len:.1f}")
        
        print(f"\n📊 TỔNG KẾT THAY ĐỔI:")
        print(f"  Records bị loại bỏ:     {changes_detected['records_removed']:,}")
        print(f"  Videos có giảm độ dài:  {changes_detected['length_reduced']}")
        print(f"  Phát hiện underscore:   {changes_detected['underscores_added']} mẫu")

    def create_visualizations(self):
        """Tạo các biểu đồ trực quan"""
        print("\n\n📈 TẠO BIỂU ĐỒ TRỰC QUAN")
        print("=" * 60)
        
        try:
            # Tạo thư mục reports nếu chưa có
            reports_dir = Path("../../reports")
            reports_dir.mkdir(exist_ok=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Báo cáo phân tích dữ liệu comment YouTube', fontsize=16, fontweight='bold')
            
            # 1. Biểu đồ so sánh số lượng raw vs processed
            videos = []
            raw_counts = []
            processed_counts = []
            
            for video in self.video_files:
                if video in self.raw_data and video in self.processed_data:
                    videos.append(video.replace('_processed', ''))
                    raw_counts.append(len(self.raw_data[video]))
                    processed_counts.append(len(self.processed_data[video]))
            
            x = np.arange(len(videos))
            width = 0.35
            
            axes[0,0].bar(x - width/2, raw_counts, width, label='Raw Data', alpha=0.8, color='lightcoral')
            axes[0,0].bar(x + width/2, processed_counts, width, label='Processed Data', alpha=0.8, color='lightblue')
            axes[0,0].set_xlabel('Video')
            axes[0,0].set_ylabel('Số lượng comments')
            axes[0,0].set_title('So sánh số lượng Raw vs Processed')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(videos, rotation=45)
            axes[0,0].legend()
            
            # 2. Phân bố độ dài text (combined data)
            if hasattr(self, 'combined_data'):
                text_lengths = self.combined_data['text'].apply(self.analyze_text_length)
                axes[0,1].hist(text_lengths, bins=50, alpha=0.7, color='green', edgecolor='black')
                axes[0,1].set_xlabel('Độ dài text (ký tự)')
                axes[0,1].set_ylabel('Tần suất')
                axes[0,1].set_title('Phân bố độ dài text')
                axes[0,1].axvline(text_lengths.mean(), color='red', linestyle='--', 
                                label=f'Trung bình: {text_lengths.mean():.1f}')
                axes[0,1].legend()
                
                # 3. Phân bố theo video
                source_counts = self.combined_data['source'].value_counts()
                video_names = [name.replace('_processed', '') for name in source_counts.index]
                axes[1,0].pie(source_counts.values, labels=video_names, autopct='%1.1f%%', startangle=90)
                axes[1,0].set_title('Phân bố comments theo video')
                
                # 4. Phân bố số từ
                word_counts = self.combined_data['text'].apply(self.analyze_word_count)
                axes[1,1].hist(word_counts, bins=30, alpha=0.7, color='purple', edgecolor='black')
                axes[1,1].set_xlabel('Số từ')
                axes[1,1].set_ylabel('Tần suất')
                axes[1,1].set_title('Phân bố số từ trong comment')
                axes[1,1].axvline(word_counts.mean(), color='red', linestyle='--',
                                label=f'Trung bình: {word_counts.mean():.1f}')
                axes[1,1].legend()
            
            plt.tight_layout()
            chart_file = reports_dir / "data_analysis_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✅ Đã lưu biểu đồ: {chart_file}")
            
            # Hiển thị biểu đồ
            plt.show()
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo biểu đồ: {e}")

    def export_summary_report(self):
        """Xuất báo cáo tổng kết ra file"""
        print("\n\n💾 XUẤT BÁO CÁO TỔNG KẾT")
        print("=" * 60)
        
        try:
            reports_dir = Path("../../reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_file = reports_dir / f"data_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("BÁOCÁO PHÂN TÍCH DỮ LIỆU COMMENT YOUTUBE\n")
                f.write("=" * 60 + "\n")
                f.write(f"Ngày tạo: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Thống kê tổng quan
                total_raw = sum(len(df) for df in self.raw_data.values())
                total_processed = sum(len(df) for df in self.processed_data.values())
                
                f.write("THỐNG KÊ TỔNG QUAN:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Tổng comment raw:        {total_raw:,}\n")
                f.write(f"Tổng comment processed:  {total_processed:,}\n")
                f.write(f"Tỷ lệ giữ lại:          {(total_processed/total_raw*100):.2f}%\n")
                f.write(f"Số video phân tích:     {len(self.video_files)}\n\n")
                
                # Chi tiết từng video
                f.write("CHI TIẾT TỪNG VIDEO:\n")
                f.write("-" * 30 + "\n")
                for video in self.video_files:
                    if video in self.raw_data and video in self.processed_data:
                        raw_count = len(self.raw_data[video])
                        proc_count = len(self.processed_data[video])
                        retention_rate = (proc_count / raw_count * 100) if raw_count > 0 else 0
                        
                        f.write(f"{video}:\n")
                        f.write(f"  Raw: {raw_count:,}, Processed: {proc_count:,}, Tỷ lệ: {retention_rate:.1f}%\n")
                
                # Thống kê combined data
                if hasattr(self, 'combined_data'):
                    f.write(f"\nDỮ LIỆU TỔNG HỢP:\n")
                    f.write("-" * 30 + "\n")
                    df = self.combined_data
                    df['text_length'] = df['text'].apply(self.analyze_text_length)
                    df['word_count'] = df['text'].apply(self.analyze_word_count)
                    
                    f.write(f"Tổng records: {len(df):,}\n")
                    f.write(f"Độ dài TB: {df['text_length'].mean():.1f} ký tự\n")
                    f.write(f"Số từ TB: {df['word_count'].mean():.1f} từ\n")
                    f.write(f"Độ dài min/max: {df['text_length'].min()}/{df['text_length'].max()}\n")
            
            print(f"✅ Đã xuất báo cáo: {report_file}")
            
        except Exception as e:
            print(f"❌ Lỗi khi xuất báo cáo: {e}")

    def run_full_analysis(self):
        """Chạy phân tích đầy đủ"""
        print("🚀 BẮT ĐẦU PHÂN TÍCH DỮ LIỆU")
        print("=" * 60)
        
        # Load dữ liệu
        self.load_data()
        
        # Các bước phân tích
        self.generate_basic_statistics()
        self.analyze_text_quality()
        self.analyze_video_distribution()
        self.detect_preprocessing_changes()
        
        # Tạo visualizations
        self.create_visualizations()
        
        # Xuất báo cáo
        self.export_summary_report()
        
        print("\n\n✅ HOÀN THÀNH PHÂN TÍCH DỮ LIỆU!")
        print("📁 Kiểm tra thư mục reports/ để xem kết quả chi tiết.")


if __name__ == "__main__":
    # Khởi chạy phân tích
    analyzer = DataAnalysisReport()
    analyzer.run_full_analysis()