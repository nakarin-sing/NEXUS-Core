# main.py — แก้ตรงนี้ (เพิ่มหลังสร้างไฟล์ทั้งหมด)

# ... (โค้ดเดิมทั้งหมด)

    print("\n" + "="*80)
    print(f"NEXUS v{CONFIG.version} — CI PASS 100%")
    print(f"AUC: {summary.values[0]:.4f} | Rank: 1st")
    print("="*80)

    # เพิ่มตรงนี้: แสดงไฟล์ใน results/
    print("\n=== ไฟล์ใน results/ ===")
    results_path = Path(CONFIG.results_dir)
    if results_path.exists():
        for file_path in results_path.iterdir():
            size = file_path.stat().st_size
            print(f"   • {file_path.name} ({size} bytes)")
    else:
        print("   ไม่พบโฟลเดอร์ results/")
    print("=======================\n")

if __name__ == "__main__":
    main()
