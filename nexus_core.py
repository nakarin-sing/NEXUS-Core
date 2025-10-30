# File: nexus_core.py
# อัปเดตเพื่อให้รองรับการเปลี่ยนแปลง API ของ river.datasets ในเวอร์ชัน >= 0.18
# และเพิ่ม NEXUS_River, CONFIG เพื่อแก้ไข ImportError

from river import datasets
import logging

# ตั้งค่า logger สำหรับการแสดงข้อความเตือน
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# 1. เพิ่ม CONFIG เพื่อแก้ไข ImportError
# ----------------------------------------------------
CONFIG = {
    "model_version": "1.0",
    "default_dataset": "Phishing",
    "log_level": "INFO"
}

# แผนที่ชุดข้อมูล (Dataset Map) ที่อัปเดตแล้ว
DATASET_MAP = {}

# รายการชุดข้อมูลใหม่ที่พบว่าใช้งานได้และถูกนำมาใช้แทนชุดข้อมูลเก่า
candidate_datasets = {
    "Phishing": datasets.Phishing,
    "Bikes": datasets.Bikes,
    "Higgs": datasets.Higgs,
    "Electricity": datasets.Elec2,
}

# ตรวจสอบว่า dataset มีอยู่จริงก่อนเพิ่มลงใน map
for name, dataset_class in candidate_datasets.items():
    try:
        if hasattr(datasets, dataset_class.__name__):
            DATASET_MAP[name] = dataset_class
        else:
            logger.warning(f"Dataset class {dataset_class.__name__} not found in river.datasets. Skipping.")
    except Exception as e:
        logger.error(f"Error processing dataset {name}: {e}. Skipping.")

# ----------------------------------------------------
# 2. ฟังก์ชันสำหรับโหลด dataset (ยังคงเดิม)
# ----------------------------------------------------
def load_dataset(dataset_name):
    """โหลดชุดข้อมูลจาก DATASET_MAP"""
    if dataset_name in DATASET_MAP:
        return DATASET_MAP[dataset_name]()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets are: {list(DATASET_MAP.keys())}")

# ----------------------------------------------------
# 3. เพิ่ม NEXUS_River Class เพื่อแก้ไข ImportError
# ----------------------------------------------------
class NEXUS_River:
    """
    คลาสหลักสำหรับการรวม (Integration) River เข้ากับ NEXUS Core
    """
    def __init__(self, dataset_name=CONFIG["default_dataset"], model=None):
        self.dataset_name = dataset_name
        self.model = model
        print(f"NEXUS_River initialized with dataset: {self.dataset_name}")

    def get_data_stream(self):
        """ส่งกลับ iterator ของ data stream โดยใช้ load_dataset"""
        return load_dataset(self.dataset_name)

    def train_and_test(self):
        """ฟังก์ชัน Placeholder สำหรับการฝึกและทดสอบโมเดล"""
        print(f"Starting training and testing process for {self.dataset_name}...")
        if not self.model:
            print("Warning: Model is None. Skipping training.")
            return

# ----------------------------------------------------
# 4. แสดงผลลัพธ์ (สำหรับการดีบัก) และตัวอย่างการใช้งาน
# ----------------------------------------------------

print(f"--- Dataset Map Status ---")
print(f"Dataset Map Updated. Currently available datasets: {list(DATASET_MAP.keys())}")
print(f"--------------------------")

if __name__ == "__main__":
    # ทดสอบการโหลด
    try:
        phishing_data = load_dataset("Phishing")
        # พิมพ์รายการแรกเพื่อยืนยันว่าโหลดสำเร็จ
        x, y = next(phishing_data)
        print(f"Successfully loaded Phishing dataset.")
        
        # ทดสอบการสร้าง NEXUS_River
        nexus_instance = NEXUS_River()
        stream = nexus_instance.get_data_stream()
        print(f"Successfully got data stream from NEXUS_River.")

    except ValueError as e:
        print(f"Error loading dataset: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
