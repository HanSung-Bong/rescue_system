# test_load.py
import tensorrt as trt
import os

# 1. 파일 경로 확인
engine_path = "/home/hansung/kroc/src/rescue_system/models/yolo11n_pc.engine" # 경로 꼭 확인!

if not os.path.exists(engine_path):
    print(f"❌ 파일이 없습니다: {engine_path}")
    exit()

print(f"📂 파일 크기: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")

# 2. 로드 시도
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

try:
    with open(engine_path, "rb") as f:
        print("🔄 엔진 로드 중...")
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine:
        print("✅ 엔진 로드 성공! (버전 문제 없음)")
    else:
        print("❌ 엔진 로드 실패 (파일은 읽었으나 객체 생성 실패 - 버전 문제 가능성)")

except Exception as e:
    print(f"❌ 에러 발생: {e}")