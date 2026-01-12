import tensorrt as trt
import os

# 설정
ONNX_FILE_PATH = 'yolo11n.onnx'
ENGINE_FILE_PATH = 'yolo11n.engine'

# 로거 생성
logger = trt.Logger(trt.Logger.INFO)

def build_engine():
    builder = trt.Builder(logger)
    
    # 1. 네트워크 정의 생성 (Explicit Batch 필수)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 2. ONNX 파일 파싱
    if not os.path.exists(ONNX_FILE_PATH):
        print(f"❌ ONNX 파일이 없습니다: {ONNX_FILE_PATH}")
        return

    print(f"🔄 ONNX 파일 파싱 중... ({ONNX_FILE_PATH})")
    with open(ONNX_FILE_PATH, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ ONNX 파싱 실패!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 3. 빌드 설정 (Config)
    config = builder.create_builder_config()
    
    # 메모리 풀 설정 (최신 API 대응)
    # 구버전에서는 config.max_workspace_size 였으나 최신 버전은 아래와 같이 씁니다.
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    except:
        print("⚠️ 메모리 설정 경고: 구버전 API이거나 설정 실패 (무시 가능)")
        pass

    # FP16 사용 (가능한 경우)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("⚡ FP16 모드 활성화")

    # 4. 동적 형상(Dynamic Shape) 프로파일 설정 ★중요★
    profile = builder.create_optimization_profile()
    
    # 입력 텐서 이름 찾기 (보통 'images')
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"ℹ️ 입력 텐서 이름: {input_name}")

    # (Min, Opt, Max) 설정: (배치, 채널, 높이, 너비)
    # 최소: 1장, 640x640
    # 최적: 1장, 640x640
    # 최대: 8장, 1280x1280 (필요시 조절)
    profile.set_shape(input_name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 1280, 1280))
    config.add_optimization_profile(profile)

    # 5. 엔진 빌드 및 직렬화
    print("🚀 엔진 빌드 시작... (시간이 좀 걸립니다)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine:
        with open(ENGINE_FILE_PATH, "wb") as f:
            f.write(serialized_engine)
        print(f"✅ 엔진 생성 완료! 저장됨: {ENGINE_FILE_PATH}")
    else:
        print("❌ 엔진 빌드 실패")

if __name__ == "__main__":
    build_engine()