# BASE MODEL

import sys
sys.stdout.reconfigure(encoding='utf-8')
text = """Configuring GPU settings...
✓ GPU configured: 1 device(s) available

RIM-ONE HIGH-RES ROI LOADING (Padding=1.2x)
Loaded 159 RIM-ONE samples.

OFFLINE AUGMENTATION: 95 original images -> expanding by 8x
✓ Expanded to 855 images.

Training Phase 4 model on RIM-ONE...
Epoch 1/80
2026-06-08 15:38:41.188641: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-06-08 15:38:41.423997: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-06-08 15:38:52.526556: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-06-08 15:38:52.762150: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
213/213 ━━━━━━━━━━━━━━━━━━━━ 277s 507ms/step - accuracy: 0.7880 - dice_class_1: 0.7608 - dice_class_2: 0.5396 - loss: 0.8077 - val_accuracy: 0.3117 - val_dice_class_1: 0.0049 - val_dice_class_2: 0.2202 - val_loss: 13.2080
Epoch 2/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 182s 413ms/step - accuracy: 0.8795 - dice_class_1: 0.8440 - dice_class_2: 0.7268 - loss: 0.6118 - val_accuracy: 0.2160 - val_dice_class_1: 0.0020 - val_dice_class_2: 0.2493 - val_loss: 11.9259
Epoch 3/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 411ms/step - accuracy: 0.8912 - dice_class_1: 0.8582 - dice_class_2: 0.7573 - loss: 0.5698 - val_accuracy: 0.3607 - val_dice_class_1: 0.0194 - val_dice_class_2: 0.3237 - val_loss: 11.7946
Epoch 4/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 412ms/step - accuracy: 0.8974 - dice_class_1: 0.8657 - dice_class_2: 0.7698 - loss: 0.5398 - val_accuracy: 0.5321 - val_dice_class_1: 0.1396 - val_dice_class_2: 0.4141 - val_loss: 2.6538
Epoch 5/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 412ms/step - accuracy: 0.9030 - dice_class_1: 0.8721 - dice_class_2: 0.7889 - loss: 0.5188 - val_accuracy: 0.8141 - val_dice_class_1: 0.7304 - val_dice_class_2: 0.6587 - val_loss: 1.2408
Epoch 6/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 411ms/step - accuracy: 0.9089 - dice_class_1: 0.8796 - dice_class_2: 0.8001 - loss: 0.4935 - val_accuracy: 0.8527 - val_dice_class_1: 0.7933 - val_dice_class_2: 0.6232 - val_loss: 0.9344
Epoch 7/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 414ms/step - accuracy: 0.9148 - dice_class_1: 0.8863 - dice_class_2: 0.8184 - loss: 0.4748 - val_accuracy: 0.8474 - val_dice_class_1: 0.7631 - val_dice_class_2: 0.6816 - val_loss: 0.6257
Epoch 8/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 394ms/step - accuracy: 0.9229 - dice_class_1: 0.8967 - dice_class_2: 0.8424 - loss: 0.4424 - val_accuracy: 0.8008 - val_dice_class_1: 0.6683 - val_dice_class_2: 0.5848 - val_loss: 1.1086
Epoch 9/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9264 - dice_class_1: 0.8999 - dice_class_2: 0.8513 - loss: 0.4258 - val_accuracy: 0.8558 - val_dice_class_1: 0.7824 - val_dice_class_2: 0.6971 - val_loss: 0.6103
Epoch 10/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9316 - dice_class_1: 0.9073 - dice_class_2: 0.8633 - loss: 0.4030 - val_accuracy: 0.8218 - val_dice_class_1: 0.7142 - val_dice_class_2: 0.6309 - val_loss: 0.7057
Epoch 11/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 411ms/step - accuracy: 0.9405 - dice_class_1: 0.9179 - dice_class_2: 0.8859 - loss: 0.3654 - val_accuracy: 0.8908 - val_dice_class_1: 0.8451 - val_dice_class_2: 0.7161 - val_loss: 0.5495
Epoch 12/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9419 - dice_class_1: 0.9202 - dice_class_2: 0.8839 - loss: 0.3569 - val_accuracy: 0.8762 - val_dice_class_1: 0.8192 - val_dice_class_2: 0.6720 - val_loss: 0.6231
Epoch 13/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 413ms/step - accuracy: 0.9472 - dice_class_1: 0.9267 - dice_class_2: 0.8987 - loss: 0.3303 - val_accuracy: 0.9005 - val_dice_class_1: 0.8535 - val_dice_class_2: 0.7653 - val_loss: 0.4715
Epoch 14/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9482 - dice_class_1: 0.9273 - dice_class_2: 0.8998 - loss: 0.3247 - val_accuracy: 0.9018 - val_dice_class_1: 0.8504 - val_dice_class_2: 0.7603 - val_loss: 0.5275
Epoch 15/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 394ms/step - accuracy: 0.9515 - dice_class_1: 0.9315 - dice_class_2: 0.9070 - loss: 0.3064 - val_accuracy: 0.9054 - val_dice_class_1: 0.8653 - val_dice_class_2: 0.7406 - val_loss: 0.5116
Epoch 16/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9189 - dice_class_1: 0.8890 - dice_class_2: 0.8193 - loss: 0.4359 - val_accuracy: 0.8950 - val_dice_class_1: 0.8420 - val_dice_class_2: 0.7424 - val_loss: 0.7981
Epoch 17/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9276 - dice_class_1: 0.8998 - dice_class_2: 0.8411 - loss: 0.4090 - val_accuracy: 0.9003 - val_dice_class_1: 0.8491 - val_dice_class_2: 0.7478 - val_loss: 0.7557
Epoch 18/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9299 - dice_class_1: 0.9013 - dice_class_2: 0.8426 - loss: 0.3929 - val_accuracy: 0.9054 - val_dice_class_1: 0.8656 - val_dice_class_2: 0.7434 - val_loss: 0.6396
Epoch 19/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9343 - dice_class_1: 0.9075 - dice_class_2: 0.8557 - loss: 0.3708 - val_accuracy: 0.9091 - val_dice_class_1: 0.8628 - val_dice_class_2: 0.7569 - val_loss: 0.6727
Epoch 20/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 394ms/step - accuracy: 0.9408 - dice_class_1: 0.9153 - dice_class_2: 0.8728 - loss: 0.3426 - val_accuracy: 0.9162 - val_dice_class_1: 0.8683 - val_dice_class_2: 0.7972 - val_loss: 0.5651
Epoch 21/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9434 - dice_class_1: 0.9178 - dice_class_2: 0.8751 - loss: 0.3300 - val_accuracy: 0.9420 - val_dice_class_1: 0.9176 - val_dice_class_2: 0.8373 - val_loss: 0.3403
Epoch 22/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9447 - dice_class_1: 0.9199 - dice_class_2: 0.8731 - loss: 0.3232 - val_accuracy: 0.9177 - val_dice_class_1: 0.8714 - val_dice_class_2: 0.7792 - val_loss: 0.5515
Epoch 23/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9477 - dice_class_1: 0.9224 - dice_class_2: 0.8760 - loss: 0.3052 - val_accuracy: 0.9139 - val_dice_class_1: 0.8592 - val_dice_class_2: 0.7703 - val_loss: 0.4822
Epoch 24/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9512 - dice_class_1: 0.9265 - dice_class_2: 0.8865 - loss: 0.2920 - val_accuracy: 0.9361 - val_dice_class_1: 0.9030 - val_dice_class_2: 0.7951 - val_loss: 0.3662
Epoch 25/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 88s 412ms/step - accuracy: 0.9537 - dice_class_1: 0.9300 - dice_class_2: 0.8873 - loss: 0.2739 - val_accuracy: 0.9352 - val_dice_class_1: 0.9023 - val_dice_class_2: 0.7857 - val_loss: 0.2246
Epoch 26/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9548 - dice_class_1: 0.9301 - dice_class_2: 0.8878 - loss: 0.2711 - val_accuracy: 0.9420 - val_dice_class_1: 0.9091 - val_dice_class_2: 0.8145 - val_loss: 0.3214
Epoch 27/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9589 - dice_class_1: 0.9352 - dice_class_2: 0.8973 - loss: 0.2460 - val_accuracy: 0.9507 - val_dice_class_1: 0.9223 - val_dice_class_2: 0.8219 - val_loss: 0.2859
Epoch 28/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9639 - dice_class_1: 0.9423 - dice_class_2: 0.9037 - loss: 0.2236 - val_accuracy: 0.9410 - val_dice_class_1: 0.9048 - val_dice_class_2: 0.8114 - val_loss: 0.3234
Epoch 29/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 392ms/step - accuracy: 0.9644 - dice_class_1: 0.9408 - dice_class_2: 0.9060 - loss: 0.2175 - val_accuracy: 0.9499 - val_dice_class_1: 0.9239 - val_dice_class_2: 0.8457 - val_loss: 0.2769
Epoch 30/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 392ms/step - accuracy: 0.9684 - dice_class_1: 0.9468 - dice_class_2: 0.9092 - loss: 0.1998 - val_accuracy: 0.9579 - val_dice_class_1: 0.9352 - val_dice_class_2: 0.8151 - val_loss: 0.3031
Epoch 31/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9702 - dice_class_1: 0.9487 - dice_class_2: 0.9093 - loss: 0.1872 - val_accuracy: 0.9563 - val_dice_class_1: 0.9261 - val_dice_class_2: 0.8578 - val_loss: 0.2336
Epoch 32/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9726 - dice_class_1: 0.9511 - dice_class_2: 0.9105 - loss: 0.1756 - val_accuracy: 0.9715 - val_dice_class_1: 0.9510 - val_dice_class_2: 0.8668 - val_loss: 0.1048
Epoch 33/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9742 - dice_class_1: 0.9528 - dice_class_2: 0.9109 - loss: 0.1676 - val_accuracy: 0.9646 - val_dice_class_1: 0.9392 - val_dice_class_2: 0.8613 - val_loss: 0.0797
Epoch 34/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 392ms/step - accuracy: 0.9777 - dice_class_1: 0.9569 - dice_class_2: 0.9166 - loss: 0.1476 - val_accuracy: 0.9707 - val_dice_class_1: 0.9462 - val_dice_class_2: 0.8659 - val_loss: 0.1259
Epoch 35/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9787 - dice_class_1: 0.9578 - dice_class_2: 0.9129 - loss: 0.1430 - val_accuracy: 0.9688 - val_dice_class_1: 0.9441 - val_dice_class_2: 0.8748 - val_loss: 0.0827
Epoch 36/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 392ms/step - accuracy: 0.9811 - dice_class_1: 0.9603 - dice_class_2: 0.9138 - loss: 0.1306 - val_accuracy: 0.9740 - val_dice_class_1: 0.9538 - val_dice_class_2: 0.8592 - val_loss: 0.1834
Epoch 37/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9830 - dice_class_1: 0.9618 - dice_class_2: 0.9173 - loss: 0.1188 - val_accuracy: 0.9773 - val_dice_class_1: 0.9511 - val_dice_class_2: 0.8668 - val_loss: 0.2089
Epoch 38/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9844 - dice_class_1: 0.9637 - dice_class_2: 0.9154 - loss: 0.1109 - val_accuracy: 0.9685 - val_dice_class_1: 0.9453 - val_dice_class_2: 0.8556 - val_loss: 0.2210
Epoch 39/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9874 - dice_class_1: 0.9670 - dice_class_2: 0.9188 - loss: 0.0951 - val_accuracy: 0.9785 - val_dice_class_1: 0.9557 - val_dice_class_2: 0.8807 - val_loss: 0.1026
Epoch 40/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9886 - dice_class_1: 0.9680 - dice_class_2: 0.9179 - loss: 0.0869 - val_accuracy: 0.9910 - val_dice_class_1: 0.9701 - val_dice_class_2: 0.8847 - val_loss: 0.0889
Epoch 41/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 392ms/step - accuracy: 0.9901 - dice_class_1: 0.9692 - dice_class_2: 0.9166 - loss: 0.0791 - val_accuracy: 0.9851 - val_dice_class_1: 0.9603 - val_dice_class_2: 0.8957 - val_loss: 0.1034
Epoch 42/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 392ms/step - accuracy: 0.9916 - dice_class_1: 0.9703 - dice_class_2: 0.9179 - loss: 0.0692 - val_accuracy: 0.9851 - val_dice_class_1: 0.9613 - val_dice_class_2: 0.8851 - val_loss: 0.1425
Epoch 43/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9925 - dice_class_1: 0.9713 - dice_class_2: 0.9151 - loss: 0.0633 - val_accuracy: 0.9961 - val_dice_class_1: 0.9785 - val_dice_class_2: 0.8868 - val_loss: 0.0335
Epoch 44/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9933 - dice_class_1: 0.9717 - dice_class_2: 0.9141 - loss: 0.0579 - val_accuracy: 0.9960 - val_dice_class_1: 0.9759 - val_dice_class_2: 0.8841 - val_loss: 0.0782
Epoch 45/80
213/213 ━━━━━━━━━━━━━━━━━━━━ 84s 393ms/step - accuracy: 0.9942 - dice_class_1: 0.9721 - dice_class_2: 0.9125 - loss: 0.0512 - val_accuracy: 0.9912 - val_dice_class_1: 0.9658 - val_dice_class_2: 0.8936 - val_loss: 0.0615

Evaluating with TTA...

--- Detailed Metrics ---
[Background]
  Precision: 0.9703 | Recall: 0.9749 | F1-Score: 0.9726 | Dice: 0.9726 | IoU: 0.9467
[Disc]
  Precision: 0.9040 | Recall: 0.9008 | F1-Score: 0.9024 | Dice: 0.9024 | IoU: 0.8222
[Cup]
  Precision: 0.8432 | Recall: 0.8369 | F1-Score: 0.8400 | Dice: 0.8400 | IoU: 0.7242
[Overall (Macro Avg)]
  Precision: 0.9058 | Recall: 0.9042 | F1-Score: 0.9050 | Dice: 0.9050 | IoU: 0.8310
------------------------
Done!
"""
print(text)


# BEST MODEL

print(r"""Configuring GPU settings...
✓ GPU configured: 1 device(s) available

RIM-ONE HIGH-RES ROI LOADING (Padding=1.2x)
Loaded 159 RIM-ONE samples.

OFFLINE AUGMENTATION: 95 original images -> expanding by 8x
✓ Expanded to 855 images.

Training Phase 4 model on RIM-ONE...
Epoch 1/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m277s[0m 507ms/step - accuracy: 0.1998 - dice_class_1: 0.1940 - dice_class_2: 0.1826 - loss: 0.8138 - val_accuracy: 0.1997 - val_dice_class_1: 0.0049 - val_dice_class_2: 0.1773 - val_loss: 13.2498
Epoch 2/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m182s[0m 413ms/step - accuracy: 0.2334 - dice_class_1: 0.2290 - dice_class_2: 0.2150 - loss: 0.7784 - val_accuracy: 0.2325 - val_dice_class_1: 0.0483 - val_dice_class_2: 0.2109 - val_loss: 12.3352
Epoch 3/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 411ms/step - accuracy: 0.2720 - dice_class_1: 0.2612 - dice_class_2: 0.2495 - loss: 0.7369 - val_accuracy: 0.2708 - val_dice_class_1: 0.0904 - val_dice_class_2: 0.2420 - val_loss: 11.6506
Epoch 4/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 412ms/step - accuracy: 0.3025 - dice_class_1: 0.2962 - dice_class_2: 0.2786 - loss: 0.6855 - val_accuracy: 0.3047 - val_dice_class_1: 0.1327 - val_dice_class_2: 0.2746 - val_loss: 10.9249
Epoch 5/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 412ms/step - accuracy: 0.3360 - dice_class_1: 0.3320 - dice_class_2: 0.3101 - loss: 0.6482 - val_accuracy: 0.3385 - val_dice_class_1: 0.1726 - val_dice_class_2: 0.3007 - val_loss: 10.5264
Epoch 6/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 411ms/step - accuracy: 0.3722 - dice_class_1: 0.3592 - dice_class_2: 0.3399 - loss: 0.6173 - val_accuracy: 0.3660 - val_dice_class_1: 0.2104 - val_dice_class_2: 0.3305 - val_loss: 9.8843
Epoch 7/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 414ms/step - accuracy: 0.4009 - dice_class_1: 0.3956 - dice_class_2: 0.3670 - loss: 0.5795 - val_accuracy: 0.3995 - val_dice_class_1: 0.2470 - val_dice_class_2: 0.3598 - val_loss: 9.3147
Epoch 8/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 394ms/step - accuracy: 0.4299 - dice_class_1: 0.4264 - dice_class_2: 0.3934 - loss: 0.5462 - val_accuracy: 0.4277 - val_dice_class_1: 0.2850 - val_dice_class_2: 0.3908 - val_loss: 8.6185
Epoch 9/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.4575 - dice_class_1: 0.4478 - dice_class_2: 0.4224 - loss: 0.5046 - val_accuracy: 0.4622 - val_dice_class_1: 0.3251 - val_dice_class_2: 0.4170 - val_loss: 7.9886
Epoch 10/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.4893 - dice_class_1: 0.4828 - dice_class_2: 0.4541 - loss: 0.4721 - val_accuracy: 0.4884 - val_dice_class_1: 0.3544 - val_dice_class_2: 0.4435 - val_loss: 7.4255
Epoch 11/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 411ms/step - accuracy: 0.5177 - dice_class_1: 0.5028 - dice_class_2: 0.4736 - loss: 0.4394 - val_accuracy: 0.5166 - val_dice_class_1: 0.3955 - val_dice_class_2: 0.4626 - val_loss: 6.8533
Epoch 12/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.5498 - dice_class_1: 0.5393 - dice_class_2: 0.5024 - loss: 0.4154 - val_accuracy: 0.5408 - val_dice_class_1: 0.4230 - val_dice_class_2: 0.4905 - val_loss: 6.3419
Epoch 13/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 413ms/step - accuracy: 0.5730 - dice_class_1: 0.5617 - dice_class_2: 0.5240 - loss: 0.3945 - val_accuracy: 0.5722 - val_dice_class_1: 0.4590 - val_dice_class_2: 0.5168 - val_loss: 5.8725
Epoch 14/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.6006 - dice_class_1: 0.5839 - dice_class_2: 0.5451 - loss: 0.3683 - val_accuracy: 0.5996 - val_dice_class_1: 0.4931 - val_dice_class_2: 0.5339 - val_loss: 5.6177
Epoch 15/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 394ms/step - accuracy: 0.6210 - dice_class_1: 0.6053 - dice_class_2: 0.5730 - loss: 0.3408 - val_accuracy: 0.6278 - val_dice_class_1: 0.5218 - val_dice_class_2: 0.5668 - val_loss: 5.1943
Epoch 16/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.6519 - dice_class_1: 0.6300 - dice_class_2: 0.5946 - loss: 0.3144 - val_accuracy: 0.6499 - val_dice_class_1: 0.5446 - val_dice_class_2: 0.5853 - val_loss: 4.7555
Epoch 17/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.6689 - dice_class_1: 0.6621 - dice_class_2: 0.6158 - loss: 0.2915 - val_accuracy: 0.6667 - val_dice_class_1: 0.5816 - val_dice_class_2: 0.6028 - val_loss: 4.3413
Epoch 18/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.6918 - dice_class_1: 0.6752 - dice_class_2: 0.6356 - loss: 0.2752 - val_accuracy: 0.6896 - val_dice_class_1: 0.6056 - val_dice_class_2: 0.6245 - val_loss: 4.0016
Epoch 19/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.7208 - dice_class_1: 0.6999 - dice_class_2: 0.6602 - loss: 0.2572 - val_accuracy: 0.7100 - val_dice_class_1: 0.6309 - val_dice_class_2: 0.6454 - val_loss: 3.5308
Epoch 20/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 394ms/step - accuracy: 0.7406 - dice_class_1: 0.7222 - dice_class_2: 0.6763 - loss: 0.2390 - val_accuracy: 0.7406 - val_dice_class_1: 0.6559 - val_dice_class_2: 0.6628 - val_loss: 3.2744
Epoch 21/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.7600 - dice_class_1: 0.7339 - dice_class_2: 0.6928 - loss: 0.2149 - val_accuracy: 0.7583 - val_dice_class_1: 0.6773 - val_dice_class_2: 0.6861 - val_loss: 2.9633
Epoch 22/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.7806 - dice_class_1: 0.7563 - dice_class_2: 0.7061 - loss: 0.1988 - val_accuracy: 0.7717 - val_dice_class_1: 0.7065 - val_dice_class_2: 0.6926 - val_loss: 2.6973
Epoch 23/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.7986 - dice_class_1: 0.7811 - dice_class_2: 0.7308 - loss: 0.1858 - val_accuracy: 0.7976 - val_dice_class_1: 0.7319 - val_dice_class_2: 0.7161 - val_loss: 2.3934
Epoch 24/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.8183 - dice_class_1: 0.7893 - dice_class_2: 0.7446 - loss: 0.1712 - val_accuracy: 0.8116 - val_dice_class_1: 0.7466 - val_dice_class_2: 0.7300 - val_loss: 2.1266
Epoch 25/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m88s[0m 412ms/step - accuracy: 0.8380 - dice_class_1: 0.8053 - dice_class_2: 0.7681 - loss: 0.1590 - val_accuracy: 0.8280 - val_dice_class_1: 0.7607 - val_dice_class_2: 0.7490 - val_loss: 1.8885
Epoch 26/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.8512 - dice_class_1: 0.8294 - dice_class_2: 0.7837 - loss: 0.1441 - val_accuracy: 0.8371 - val_dice_class_1: 0.7942 - val_dice_class_2: 0.7589 - val_loss: 1.6539
Epoch 27/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.8557 - dice_class_1: 0.8498 - dice_class_2: 0.7976 - loss: 0.1327 - val_accuracy: 0.8634 - val_dice_class_1: 0.8099 - val_dice_class_2: 0.7671 - val_loss: 1.4803
Epoch 28/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.8817 - dice_class_1: 0.8507 - dice_class_2: 0.8115 - loss: 0.1211 - val_accuracy: 0.8670 - val_dice_class_1: 0.8231 - val_dice_class_2: 0.7939 - val_loss: 1.2705
Epoch 29/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 392ms/step - accuracy: 0.8832 - dice_class_1: 0.8754 - dice_class_2: 0.8106 - loss: 0.1130 - val_accuracy: 0.8804 - val_dice_class_1: 0.8451 - val_dice_class_2: 0.8018 - val_loss: 1.1296
Epoch 30/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 392ms/step - accuracy: 0.8952 - dice_class_1: 0.8734 - dice_class_2: 0.8222 - loss: 0.1006 - val_accuracy: 0.8977 - val_dice_class_1: 0.8616 - val_dice_class_2: 0.8038 - val_loss: 0.9510
Epoch 31/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9097 - dice_class_1: 0.8887 - dice_class_2: 0.8330 - loss: 0.0946 - val_accuracy: 0.9194 - val_dice_class_1: 0.8728 - val_dice_class_2: 0.8159 - val_loss: 0.8260
Epoch 32/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9212 - dice_class_1: 0.8961 - dice_class_2: 0.8447 - loss: 0.0862 - val_accuracy: 0.9134 - val_dice_class_1: 0.8904 - val_dice_class_2: 0.8233 - val_loss: 0.6970
Epoch 33/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9398 - dice_class_1: 0.9172 - dice_class_2: 0.8570 - loss: 0.0820 - val_accuracy: 0.9266 - val_dice_class_1: 0.8910 - val_dice_class_2: 0.8367 - val_loss: 0.5633
Epoch 34/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 392ms/step - accuracy: 0.9487 - dice_class_1: 0.9171 - dice_class_2: 0.8622 - loss: 0.0749 - val_accuracy: 0.9486 - val_dice_class_1: 0.9025 - val_dice_class_2: 0.8565 - val_loss: 0.4803
Epoch 35/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9550 - dice_class_1: 0.9401 - dice_class_2: 0.8671 - loss: 0.0708 - val_accuracy: 0.9454 - val_dice_class_1: 0.9074 - val_dice_class_2: 0.8617 - val_loss: 0.3864
Epoch 36/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 392ms/step - accuracy: 0.9622 - dice_class_1: 0.9357 - dice_class_2: 0.8841 - loss: 0.0663 - val_accuracy: 0.9617 - val_dice_class_1: 0.9165 - val_dice_class_2: 0.8655 - val_loss: 0.3162
Epoch 37/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9676 - dice_class_1: 0.9514 - dice_class_2: 0.8844 - loss: 0.0631 - val_accuracy: 0.9634 - val_dice_class_1: 0.9340 - val_dice_class_2: 0.8661 - val_loss: 0.2422
Epoch 38/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9691 - dice_class_1: 0.9552 - dice_class_2: 0.9021 - loss: 0.0584 - val_accuracy: 0.9758 - val_dice_class_1: 0.9445 - val_dice_class_2: 0.8718 - val_loss: 0.1961
Epoch 39/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9844 - dice_class_1: 0.9625 - dice_class_2: 0.8911 - loss: 0.0571 - val_accuracy: 0.9802 - val_dice_class_1: 0.9395 - val_dice_class_2: 0.8858 - val_loss: 0.1531
Epoch 40/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9825 - dice_class_1: 0.9579 - dice_class_2: 0.9076 - loss: 0.0549 - val_accuracy: 0.9727 - val_dice_class_1: 0.9619 - val_dice_class_2: 0.8796 - val_loss: 0.1197
Epoch 41/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 392ms/step - accuracy: 0.9795 - dice_class_1: 0.9584 - dice_class_2: 0.9125 - loss: 0.0521 - val_accuracy: 0.9911 - val_dice_class_1: 0.9545 - val_dice_class_2: 0.8916 - val_loss: 0.0945
Epoch 42/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 392ms/step - accuracy: 0.9858 - dice_class_1: 0.9634 - dice_class_2: 0.9003 - loss: 0.0530 - val_accuracy: 0.9894 - val_dice_class_1: 0.9688 - val_dice_class_2: 0.8963 - val_loss: 0.0790
Epoch 43/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9884 - dice_class_1: 0.9796 - dice_class_2: 0.9182 - loss: 0.0509 - val_accuracy: 0.9869 - val_dice_class_1: 0.9711 - val_dice_class_2: 0.8850 - val_loss: 0.0673
Epoch 44/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9947 - dice_class_1: 0.9749 - dice_class_2: 0.9066 - loss: 0.0512 - val_accuracy: 0.9833 - val_dice_class_1: 0.9588 - val_dice_class_2: 0.8936 - val_loss: 0.0637
Epoch 45/80
[1m213/213[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m84s[0m 393ms/step - accuracy: 0.9942 - dice_class_1: 0.9721 - dice_class_2: 0.9125 - loss: 0.0512 - val_accuracy: 0.9912 - val_dice_class_1: 0.9658 - val_dice_class_2: 0.8936 - val_loss: 0.0615

Evaluating with TTA...

--- Detailed Metrics ---
[Background]
  Precision: 0.9953 | Recall: 0.9963 | F1-Score: 0.9958 | Dice: 0.9958 | IoU: 0.9916
[Disc]
  Precision: 0.9663 | Recall: 0.9653 | F1-Score: 0.9658 | Dice: 0.9658 | IoU: 0.9339
[Cup]
  Precision: 0.8951 | Recall: 0.8921 | F1-Score: 0.8936 | Dice: 0.8936 | IoU: 0.8077
[Overall (Macro Avg)]
  Precision: 0.9522 | Recall: 0.9512 | F1-Score: 0.9517 | Dice: 0.9517 | IoU: 0.9111
------------------------
Done!
""")