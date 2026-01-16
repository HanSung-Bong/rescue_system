import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2  # [추가] 이미지 처리를 위해 필요

class YoloTRT:
    def __init__(self, engine_path, logger_severity=trt.Logger.WARNING):
        # 1. Logger & Runtime 생성
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)

        # 2. 엔진 파일 로드
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        # 3. Context 생성
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 4. 메모리 할당 및 텐서 정보 파싱
        self.inputs = {}
        self.outputs = {}
        self.allocations = [] 

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            dtype = self.engine.get_tensor_dtype(name)
            
            # Shape 가져오기
            shape = self.engine.get_tensor_shape(name)
            real_shape = []
            
            # 동적 쉐이프(-1) 대응을 위해 Profile 0번의 Max Shape 사용
            if -1 in shape:
                print("Dynamic Shape Detected")
                try:
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                    if profile_shape and len(profile_shape) >= 3:
                        max_dims = profile_shape[2]
                        real_shape = [max_dims[d] for d in range(len(shape))]
                    else:
                        raise ValueError("Profile not found") 
                except Exception as e:
                    print(f"Warn: Failed to read profile for {name}, utilizing feedback ({e})")
                    if is_input:
                        real_shape = [1,3,1640,1232]
                    else:
                        real_shape = [1,84,33600]
            else:
                real_shape = list(shape)
            
            print(f"DEBUG: Tensor {name} resolved shape: {real_shape}")

            # 입력 텐서(images)의 H, W 저장 (전처리용)
            if is_input:
                self.input_h = real_shape[2]
                self.input_w = real_shape[3]

            # 메모리 크기 계산
            size = trt.volume(real_shape) * np.dtype(trt.nptype(dtype)).itemsize

            # Host/Device 메모리 할당
            host_mem = cuda.pagelocked_empty(trt.volume(real_shape), dtype=trt.nptype(dtype))
            device_mem = cuda.mem_alloc(size)
            self.allocations.append(device_mem)

            tensor_info = {
                "name": name,
                "host": host_mem,
                "device": device_mem,
                "shape": real_shape, # [1, 3, 1280, 1280] 등
                "dtype": dtype
            }

            if is_input:
                self.inputs[name] = tensor_info
            else:
                self.outputs[name] = tensor_info

    def preprocess(self, img):
        """
        Resize 대신 Center Crop 수행
        """
        # 1. Center Crop 계산 (Resizing 아님)
        # 들어오는 이미지(1640x1232) -> 모델 입력(1632x1216)
        orig_h, orig_w, _ = img.shape
        
        # 잘라낼 시작점 계산 (중앙 정렬)
        x_start = (orig_w - self.input_w) // 2  # (1640-1632)/2 = 4
        y_start = (orig_h - self.input_h) // 2  # (1232-1216)/2 = 8
        
        # [수정] 이미지 크롭 (Slicing)
        img_cropped = img[y_start : y_start + self.input_h, 
                          x_start : x_start + self.input_w]
        
        # 2. BGR -> RGB
        img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
        
        # 3. HWC -> CHW 변환 및 정규화 (0~1)
        img_transposed = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        # 4. 배치 차원 추가 및 1차원 flatten
        img_input = np.expand_dims(img_transposed, axis=0)
        return np.ascontiguousarray(img_input).ravel()

    def postprocess(self, output_tensor, original_shape):
        """
        YOLO 출력 텐서([1, 84, 8400])를 파싱하여 BBox 리스트로 변환
        """
        # output_tensor shape: (1, 84, 8400) -> (cx, cy, w, h, class_probs...)
        # 후처리를 위해 전치: (1, 8400, 84)
        output = output_tensor[0].transpose()
        
        boxes = []
        confidences = []
        class_ids = []
        
        orig_h, orig_w = original_shape[:2]

        # [수정] 스케일링 비율(Scale) 대신 오프셋(Offset) 계산
        # 전처리 때 잘라낸 만큼(4px, 8px) 더해줘야 함
        offset_x = (orig_w - self.input_w) // 2
        offset_y = (orig_h - self.input_h) // 2

        CONF_THRESH = 0.4
        
        rows = output.shape[0]
        for i in range(rows):
            classes_scores = output[i, 4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence > CONF_THRESH:
                cx, cy, w, h = output[i, 0], output[i, 1], output[i, 2], output[i, 3]
                
                # [수정] 좌표 복원: 비율 곱하기가 아니라 오프셋 더하기
                # 모델이 본 좌표 (Crop 기준) + 잘려나간 여백 (Offset)
                center_x_real = cx + offset_x
                center_y_real = cy + offset_y
                
                # 좌상단 좌표 계산
                left = int(center_x_real - w/2)
                top = int(center_y_real - h/2)
                width = int(w)  # Resizing을 안 했으므로 크기는 그대로
                height = int(h) # Resizing을 안 했으므로 크기는 그대로
                
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, 0.45)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "class_id": class_ids[i],
                    "bbox": boxes[i], # [x, y, w, h] (원본 해상도 기준)
                    "conf": confidences[i]
                })
        
        return results
    def inference(self, raw_img):   
        """
        외부에서 호출하는 메인 함수
        raw_img: cv2 image (BGR)
        """
        # 1. 전처리 (Resize & Normalize)
        input_data = self.preprocess(raw_img)
        
        # 2. 데이터 Host -> Device 복사
        for name, info in self.inputs.items():
            self.context.set_input_shape(name, info["shape"])
            np.copyto(info["host"], input_data)
            cuda.memcpy_htod_async(info["device"], info["host"], self.stream)
            self.context.set_tensor_address(name, int(info["device"]))

        for name, info in self.outputs.items():
            self.context.set_tensor_address(name, int(info["device"]))

        # 3. 추론 실행
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 4. 결과 회수 Device -> Host
        raw_results = {}
        for name, info in self.outputs.items():
            cuda.memcpy_dtoh_async(info["host"], info["device"], self.stream)
            raw_results[name] = info["host"].reshape(info["shape"])

        self.stream.synchronize()
        
        # 5. 후처리 (BBox Parsing)
        # 보통 output0이 결과 텐서임. 모델마다 이름 다를 수 있으니 확인 필요
        output_tensor = list(raw_results.values())[0] # 첫 번째 출력 텐서 사용
        
        return self.postprocess(output_tensor, raw_img.shape)