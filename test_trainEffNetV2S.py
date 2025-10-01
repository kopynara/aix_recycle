import os
import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    Callback, CSVLogger, ProgbarLogger
)
from tensorflow.keras.optimizers.legacy import Adam   # ✅ 핵심 수정

# 로그 설정
log_file = "train_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()]
)
logging.info("🚀 테스트 학습 시작")

# 디바이스 확인
print("🔎 디바이스 확인:", tf.config.list_physical_devices())
if tf.config.list_physical_devices("GPU"):
    logging.info("✅ Apple Silicon GPU(Metal) 사용 가능")
else:
    logging.warning("⚠️ GPU 없음 → CPU 모드로 실행")

# 환경 설정 (테스트 → 소량 샘플 데이터셋)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16 if tf.config.list_physical_devices("GPU") else 32
DATASET_PATH = "datasets/sample_dataset"    # ✅ 샘플 데이터셋
MODEL_NAME = 'trash_classifier_efficientnetv2S_test'
LEARNING_RATE_STEP1 = 1e-3
LEARNING_RATE_STEP2 = 1e-5

# 데이터 준비
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
)
val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
NUM_CLASSES = len(class_names)
logging.info(f"총 클래스 수: {NUM_CLASSES}, 클래스 이름: {class_names}")

# 클래스 가중치 로드
class_weight = None
if os.path.exists("class_weights.json"):
    with open("class_weights.json", "r") as f:
        class_weight_name = json.load(f)
    class_indices = train_generator.class_indices
    class_weight = {
        idx: float(class_weight_name[cls_name])
        for cls_name, idx in class_indices.items()
        if cls_name in class_weight_name
    }
    logging.info("✅ 클래스 가중치 로드 및 매핑 완료")
else:
    logging.warning("⚠️ class_weights.json 없음 → class_weight 미적용")

# 모델 구성
def build_efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetV2S(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)   # 테스트 버전 → 경량화
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = build_efficientnet_model(IMAGE_SIZE + (3,), NUM_CLASSES)
model.summary()

# 커스텀 콜백
class EarlyStopOnStableMetrics(Callback):
    def __init__(self, monitor_acc='val_accuracy', monitor_loss='val_loss',
                 patience=2, delta_acc=1e-4, delta_loss=1e-4):
        super().__init__()
        self.monitor_acc = monitor_acc
        self.monitor_loss = monitor_loss
        self.patience = patience
        self.delta_acc = delta_acc
        self.delta_loss = delta_loss
        self.acc_history = []
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get(self.monitor_acc)
        current_loss = logs.get(self.monitor_loss)
        if current_acc is None or current_loss is None:
            return
        self.acc_history.append(current_acc)
        self.loss_history.append(current_loss)
        if len(self.acc_history) >= self.patience:
            recent_acc = self.acc_history[-self.patience:]
            recent_loss = self.loss_history[-self.patience:]
            acc_stable = max(recent_acc) - min(recent_acc) < self.delta_acc
            loss_stable = max(recent_loss) - min(recent_loss) < self.delta_loss
            if acc_stable and loss_stable:
                logging.info(f"⚠️ {self.patience}번 연속 안정 → 학습 중단")
                self.model.stop_training = True

# 학습 Step1
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_NAME}_best_step1.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    EarlyStopOnStableMetrics(patience=2),
    CSVLogger("train_history_step1_test.csv", append=False),
    ProgbarLogger(count_mode='steps')
]
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_STEP1),   # ✅ legacy.Adam 사용
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_step1 = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weight
)
model.load_weights(f'{MODEL_NAME}_best_step1.keras')

# 학습 Step2
num_layers_to_train = int(len(base_model.layers) * 0.2)
for layer in base_model.layers[-num_layers_to_train:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_STEP2),   # ✅ legacy.Adam 사용
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
callbacks_ft = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f'{MODEL_NAME}_best_final.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    EarlyStopOnStableMetrics(patience=2),
    CSVLogger("train_history_step2_test.csv", append=False),
    ProgbarLogger(count_mode='steps')
]
history_step2 = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    callbacks=callbacks_ft,
    class_weight=class_weight
)

model.load_weights(f'{MODEL_NAME}_best_final.keras')
logging.info(f"✅ 최종 테스트 모델 저장 완료: {MODEL_NAME}_best_final.keras")
