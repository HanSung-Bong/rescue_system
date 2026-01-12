import vpi
import numpy as np
import cv2

class VPIStereoDepth:
    def __init__(self, width, height):
        # 입력 이미지 크기 
        self.width = width
        self.height = height
        
        # VPI Backend 설정
        self.backend = vpi.Backend.CUDA
        #JETSON 사용시 변경
        #self.backend = vpi.Backend.PVA
        
        # Stereo Disparity Estimator 생성
        self.stereo = vpi.StereoDisparityEstimator(
            (width, height), vpi.Format.U8, downscale=1, max_disparity=64
        )
        
        # 결과 담을 버퍼 (Disparity Map)
        self.disparity = vpi.Image((width, height), vpi.Format.U16)

        # 카메라 파라미터 (필수: Baseline과 Focal Length)
        self.focal_length = 500.0 
        self.baseline = 0.12 # 12cm (카메라 간 거리)

    def estimate(self, left_img_np, right_img_np):
        """
        Numpy 이미지를 받아 Depth Map(미터 단위)을 반환
        """
        # 1. Numpy -> VPI Image 변환
        left_vpi = vpi.asimage(left_img_np).convert(vpi.Format.U8, backend=self.backend)
        right_vpi = vpi.asimage(right_img_np).convert(vpi.Format.U8, backend=self.backend)

        # 2. Disparity 추정 (VPI stream 내에서 실행)
        with vpi.Backend.CUDA:
            self.disparity = self.stereo(left_vpi, right_vpi)

        # 3. VPI -> Numpy 변환 (CPU로 가져오기)
        # Disparity 포맷은 보통 Q10.5 혹은 Q11.5 형식이므로 float 변환 필요
        disp_map = self.disparity.cpu()
        disp_map = disp_map.astype(np.float32) / 32.0 # VPI 포맷 스케일링 (설정에 따라 다름)

        # 4. Disparity -> Depth 변환 (Z = f * B / d)
        # 0으로 나누기 방지
        disp_map[disp_map == 0] = 0.1
        depth_map = (self.focal_length * self.baseline) / disp_map
        
        return depth_map