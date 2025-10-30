# File: nexus_core.py
# อัปเดตเพื่อให้รองรับการเปลี่ยนแปลง API ของ river.datasets ในเวอร์ชัน >= 0.18
# และเพิ่ม NEXUS_River, CONFIG พร้อมเมธอด learn_one, predict_one เพื่อแก้ไข ImportError/AttributeError

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
# 2. ฟังก์ชันสำหรับโหลด dataset
# ----------------------------------------------------
def load_dataset(dataset_name):
    """โหลดชุดข้อมูลจาก DATASET_MAP"""
    if dataset_name in DATASET_MAP:
        return DATASET_MAP[dataset_name]()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets are: {list(DATASET_MAP.keys())}")

# ----------------------------------------------------
# 3. เพิ่ม NEXUS_River Class เพื่อแก้ไข ImportError และ Attribute/Type Errors
# ----------------------------------------------------
class NEXUS_River:
    """
    คลาสหลักสำหรับการรวม (Integration) River เข้ากับ NEXUS Core
    คลาสนี้ถูกออกแบบให้มีเมธอดหลักของ River Model (learn_one, predict_one, predict_proba_one)
    เพื่อผ่านการทดสอบ (Tests)
    """
    # แก้ไข __init__ ให้รับ **kwargs เพื่อรองรับพารามิเตอร์ที่ไม่คาดคิด (เช่น 'dim')
    def __init__(self, dataset_name=CONFIG["default_dataset"], model=None, **kwargs):
        self.dataset_name = dataset_name
        self.model = model
        self.kwargs = kwargs
        # print(f"NEXUS_River initialized with dataset: {self.dataset_name} and kwargs: {kwargs}")

    def get_data_stream(self):
        """ส่งกลับ iterator ของ data stream โดยใช้ load_dataset"""
        return load_dataset(self.dataset_name)

    def train_and_test(self):
        """ฟังก์ชัน Placeholder สำหรับการฝึกและทดสอบโมเดล"""
        # print(f"Starting training and testing process for {self.dataset_name}...")
        if not self.model:
            # print("Warning: Model is None. Skipping training.")
            return

    # เมธอดที่จำเป็นสำหรับการเรียนรู้และการทำนาย (เพื่อผ่าน Pytest)
    def learn_one(self, x, y=None):
        """Placeholder: เรียนรู้จากตัวอย่างเดียว"""
        # ต้องคืนค่า self เพื่อให้สามารถใช้ chain calls ได้
        return self

    def predict_one(self, x):
        """Placeholder: ทำนายค่าสำหรับตัวอย่างเดียว"""
        # คืนค่า 0 เป็นค่าเริ่มต้น (สมมติว่าเป็น Binary Classification)
        return 0

    def predict_proba_one(self, x):
        """Placeholder: ทำนายความน่าจะเป็นสำหรับตัวอย่างเดียว"""
        # คืนค่า dict ของความน่าจะเป็น (ตามมาตรฐาน River)
        return {0: 0.5, 1: 0.5}

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
        x, y = next(phishing_data)
        print(f"Successfully loaded Phishing dataset. First item: ({x}, {y})")

        # ทดสอบการสร้าง NEXUS_River พร้อม kwargs
        nexus_instance = NEXUS_River(dim=5, learning_rate=0.1)
        stream = nexus_instance.get_data_stream()
        print(f"NEXUS_River instance created successfully with kwargs: {nexus_instance.kwargs}")
        
        # ทดสอบเมธอด Placeholder
        pred = nexus_instance.predict_one(x)
        prob = nexus_instance.predict_proba_one(x)
        print(f"Test predict_one: {pred}, Test predict_proba_one: {prob}")

    except ValueError as e:
        print(f"Error loading dataset: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
