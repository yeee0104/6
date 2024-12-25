import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from io import BytesIO
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 禁用 TensorFlow 警告（可選）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 訓練模型並保存
def train_and_save_model(train_dir, val_dir, model_path="saved_model.keras"):
    # 建立資料增強
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # 加載資料集
    train_dataset = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
    )
    val_dataset = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode="binary"
    )

    # 加載預訓練 VGG16 模型
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # 凍結卷積層權重

    # 添加自定義分類層
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid"),  # 2 類分類
    ])

    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 添加學習率調整回調
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 訓練模型
    print("Start training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        batch_size=32,
        callbacks=[reduce_lr],
    )
    print("Training complete.")

    # 保存模型
    model.save(model_path, save_format="keras")
    print(f"Model saved to {model_path}")

    # 繪製訓練過程曲線
    plt.figure(figsize=(12, 5))

    # 繪製損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # 繪製準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.show()

# 測試圖片分類
def test_image(image_url, model_path="saved_model.keras"):
    # 確保模型存在
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        return

    # 加載模型
    model = load_model(model_path)

    # 測試圖片
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 檢查是否下載成功
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the image: {e}")
        return

    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 預測圖片分類
    prediction = model.predict(img_array)
    predicted_class = "without_mask" if prediction[0] > 0.5 else "with_mask"

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    plt.show()

# 主程式入口
if __name__ == "__main__":
    # 設置數據集路徑
    train_dir = "train"
    val_dir = "valid"

    # 確認是否需要訓練
    train_model = input("Do you want to train the model? (yes/no): ").strip().lower()
    if train_model == "yes":
        train_and_save_model(train_dir=train_dir, val_dir=val_dir, model_path="saved_model.keras")

    # 測試圖片分類
    image_url = input("Enter image URL to classify: ").strip()
    test_image(image_url, model_path="saved_model.keras")

