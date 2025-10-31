#!/usr/bin/env python3
"""
test_benchmark.py — รันทดสอบทันที
แสดงผลใน console + สร้าง results/
"""

import os
from main import main as run_benchmark

if __name__ == "__main__":
    print("=== NEXUS v6.6.0 LOCAL BENCHMARK ===")
    print("กำลังรัน benchmark บน Electricity dataset...")
    print("ผลลัพธ์จะแสดงใน console และบันทึกใน results/\n")

    run_benchmark()

    results_dir = "results"
    if os.path.exists(results_dir):
        print(f"\nผลลัพธ์บันทึกในโฟลเดอร์: {results_dir}/")
        for f in os.listdir(results_dir):
            print(f"   • {f}")
    else:
        print("\nไม่พบโฟลเดอร์ results/ — เกิดข้อผิดพลาด!")

    print("\n=== เสร็จสิ้น ===")
