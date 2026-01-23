#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script đơn giản để tạo báo cáo nhanh về dữ liệu
"""

import sys
import os

# Thêm đường dẫn để import module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_analysis_report import DataAnalysisReport

def main():
    print("=" * 70)
    print("           BÁO CÁO PHÂN TÍCH DỮ LIỆU COMMENT YOUTUBE")
    print("=" * 70)
    
    try:
        # Khởi tạo analyzer
        analyzer = DataAnalysisReport()
        
        # Chạy phân tích đầy đủ
        analyzer.run_full_analysis()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã dừng phân tích theo yêu cầu người dùng.")
    except Exception as e:
        print(f"\n\n❌ Lỗi trong quá trình phân tích: {e}")
        print("💡 Hãy kiểm tra lại đường dẫn dữ liệu và thử lại.")
    
    print("\n🔚 Kết thúc chương trình.")

if __name__ == "__main__":
    main()