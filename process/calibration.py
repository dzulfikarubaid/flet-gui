import json
import numpy as np
from .coin_detect import  detect_cv, detect_onnx

# image_path = ['images/tes3/baby_up.jpeg', 'images/tes3/baby_side.jpeg'] ### GANTI PATH INI

def calibration(img1, img2):
    def save_to_json(data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file)
    COIN_COEFFICIENT1 = detect_onnx(img1)
    COIN_COEFFICIENT2 = detect_onnx(img2)
    # COIN_COEFFICIENT1 = detect_cv(img1) #komputasi lebih cepat
    # COIN_COEFFICIENT2 = detect_cv(img2)
    coef = np.array([COIN_COEFFICIENT1, COIN_COEFFICIENT2])
    print(coef)

    #SIMPAN DALAM NPY
    save_to_json([COIN_COEFFICIENT1, COIN_COEFFICIENT2],'coin_coeffs.json')
    #SIMPAN DALAM TXT
    # np.savetxt('coin_coeffs.txt', coef, fmt='%.6f', header='COIN_COEFFS')

    return coef

# calibration("teskoin.png", "koin.jpeg")



