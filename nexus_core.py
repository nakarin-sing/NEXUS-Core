# File: nexus_core.py
# อัปเดตเพื่อให้รองรับการเปลี่ยนแปลง API ของ river.datasets ในเวอร์ชัน >= 0.18

from river import datasets
import logging

# ตั้งค่า logger สำหรับการแสดงข้อความเตือน
logger = logging.getLogger(__name__)

# แผนที่ชุดข้อมูล (Dataset Map) ที่อัปเดตแล้ว
# ใช้ชุดข้อมูลที่ยืนยันว่ายังมีอยู่ใน River เวอร์ชันล่าสุด
DATASET_MAP = {}

# รายการชุดข้อมูลใหม่ที่พบว่าใช้งานได้และถูกนำมาใช้แทนชุดข้อมูลเก่า
# (SEA, Mushroom, Airlines, Covertype, Bnk ถูกลบออกแล้ว)
candidate_datasets = {
    "Phishing": datasets.Phishing,
    "Bikes": datasets.Bikes,
    "Higgs": datasets.Higgs,
    "Electricity": datasets.Elec2,
    # "Bank": datasets.Bnk, # ลบออกเนื่องจากเกิด AttributeError: module 'river.datasets' has no attribute 'Bnk'
    # หากต้องการเพิ่มชุดข้อมูลอื่นกลับมาในอนาคต:
    # "SEA": datasets.SEA,
}

# ตรวจสอบว่า dataset มีอยู่จริงก่อนเพิ่มลงใน map 
# เพื่อความเข้ากันได้ย้อนหลัง (Backward Compatibility)
for name, dataset_class in candidate_datasets.items():
    try:
        # ตรวจสอบว่าคลาสถูกโหลดจาก river.datasets จริงๆ และสามารถสร้างอินสแตนซ์ได้
        # เนื่องจากเรา import เข้ามาโดยตรง เราแค่ตรวจสอบชื่อคลาสในโมดูล datasets
        if hasattr(datasets, dataset_class.__name__):
            DATASET_MAP[name] = dataset_class
        else:
            logger.warning(f"Dataset class {dataset_class.__name__} not found in river.datasets. Skipping.")
    except Exception as e:
        logger.error(f"Error processing dataset {name}: {e}. Skipping.")


# แสดงผลลัพธ์ (สำหรับการดีบัก)
print(f"--- Dataset Map Status ---")
print(f"Dataset Map Updated. Currently available datasets: {list(DATASET_MAP.keys())}")
print(f"--------------------------")

# ตัวอย่างฟังก์ชันสำหรับโหลด dataset
def load_dataset(dataset_name):
    """โหลดชุดข้อมูลจาก DATASET_MAP"""
    if dataset_name in DATASET_MAP:
        # สร้างอินสแตนซ์ของคลาส dataset และส่งคืน
        return DATASET_MAP[dataset_name]()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets are: {list(DATASET_MAP.keys())}")

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # ทดสอบการโหลด
    try:
        phishing_data = load_dataset("Phishing")
        # พิมพ์รายการแรกเพื่อยืนยันว่าโหลดสำเร็จ
        x, y = next(phishing_data)
        print(f"Successfully loaded Phishing dataset.")
        print(f"First feature vector (x): {x}")
        print(f"First target value (y): {y}")
    except ValueError as e:
        print(f"Error loading dataset: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    try:
        unknown_data = load_dataset("Mushroom") # ชุดข้อมูลที่ไม่รองรับแล้ว
    except ValueError as e:
        print(f"Expected error for old dataset: {e}")
