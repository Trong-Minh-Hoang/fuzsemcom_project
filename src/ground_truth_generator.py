import pandas as pd
from sklearn.model_selection import train_test_split
from .fuzzy_engine import create_fuzzy_system, infer, get_semantic_output

def generate_ground_truth(
    input_csv_path, 
    output_train_path, 
    output_test_path, 
    test_size=0.2, 
    random_state=42,
    use_gaussian=False
):
    """
    Generates semantic labels for raw sensor data using the updated MAMDANI FIS.
    
    ✅ FIXES:
    - Chia train/test TRƯỚC khi tạo fuzzy system (tránh overfitting)
    - Sử dụng infer() để lấy irrigation value
    - Gọi get_semantic_output() với output value (không sim)
    - Hỗ trợ Gaussian MFs
    - Xử lý exception đúng từ sim.compute()
    
    Args:
        input_csv_path: Path to raw data
        output_train_path: Path to save training data with labels
        output_test_path: Path to save test data with labels
        test_size: Proportion of test data (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        use_gaussian: Use Gaussian MFs instead of Triangular (default False)
    """
    
    # ✅ BƯỚC 1: ĐỌC DỮ LIỆU
    df = pd.read_csv(input_csv_path)
    print(f"📊 Loaded {len(df)} samples from {input_csv_path}")

    # ✅ BƯỚC 2: AUTO COLUMN MAPPING
    column_map = {
        'Moisture': 'soil_moisture',
        'pH': 'pH',
        'N': 'nitrogen',
        'Temperature': 'temperature',
        'Humidity': 'humidity'
    }

    for old, new in column_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # ✅ BƯỚC 3: VALIDATE REQUIRED COLUMNS
    required_cols = ['soil_moisture', 'temperature', 'humidity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    # ✅ BƯỚC 4: CHIA TRAIN/TEST TRƯỚC KHI TẠO FUZZY SYSTEM
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"📈 Data split: Train={len(train_df)} samples, Test={len(test_df)} samples")

    # ✅ BƯỚC 5: TẠO FUZZY SYSTEM CHỈ TỪ TRAIN DATA (adaptive scaling)
    fuzzy_sim, _ = create_fuzzy_system(
        use_gaussian=use_gaussian, 
        train_data=train_df
    )
    print(f"✨ Fuzzy system created (use_gaussian={use_gaussian})")
    
    # ✅ BƯỚC 6: HÀM PHỤ TRỢ ĐỂ GÁN NHÃN
    def assign_labels(data_df, sim):
        """
        Gán nhãn ngữ nghĩa cho dữ liệu sử dụng fuzzy inference.
        
        Args:
            data_df: DataFrame cần gán nhãn
            sim: Fuzzy system simulation
        
        Returns:
            List of semantic labels
        """
        labels = []
        errors = 0
        
        for index, row in data_df.iterrows():
            try:
                # Gọi infer() để lấy irrigation value
                irrigation_value, semantic_label = infer(
                    sim,
                    soil_moisture_val=float(row['soil_moisture']),
                    temperature_val=float(row['temperature']),
                    humidity_val=float(row['humidity'])
                )
                labels.append(semantic_label)
                
            except Exception as e:
                # Fallback: gán nhãn 'other' nếu có lỗi
                labels.append('other')
                errors += 1
        
        if errors > 0:
            print(f"⚠️  {errors} inference errors (fallback to 'other')")
        
        return labels

    # ✅ BƯỚC 7: GÁN NHÃN CHO TRAIN DATA
    print("🔄 Assigning labels to training data...")
    train_df["semantic_label"] = assign_labels(train_df, fuzzy_sim)
    train_df.to_csv(output_train_path, index=False)
    print(f"✅ Training data saved to {output_train_path}")
    print(f"   Label distribution:\n{train_df['semantic_label'].value_counts()}\n")

    # ✅ BƯỚC 8: GÁN NHÃN CHO TEST DATA
    print("🔄 Assigning labels to test data...")
    test_df["semantic_label"] = assign_labels(test_df, fuzzy_sim)
    test_df.to_csv(output_test_path, index=False)
    print(f"✅ Test data saved to {output_test_path}")
    print(f"   Label distribution:\n{test_df['semantic_label'].value_counts()}\n")


if __name__ == "__main__":
    generate_ground_truth(
        input_csv_path="data/raw/Agriculture_dataset_with_metadata.csv",
        output_train_path="data/processed/semantic_dataset_train.csv",
        output_test_path="data/processed/semantic_dataset_test.csv",
        test_size=0.2,
        random_state=42,
        use_gaussian=False  # Set to True để dùng Gaussian MFs
    )