# BASE MODEL

import sys
sys.stdout.reconfigure(encoding='utf-8')
text = """Configuring GPU settings...
✓ GPU configured: 1 device(s) available

DRISHTI HIGH-RES ROI LOADING (Padding=1.2x)
Loaded 101 Drishti samples.

OFFLINE AUGMENTATION: 60 original images -> expanding by 8x
✓ Expanded to 540 images.

Training Phase 4 model on Drishti...
Epoch 1/80
2026-06-08 15:37:35.359207: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-06-08 15:37:35.595515: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-06-08 15:37:46.642613: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
2026-06-08 15:37:46.879088: E external/local_xla/xla/stream_executor/cuda/cuda_timer.cc:86] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.
135/135 ━━━━━━━━━━━━━━━━━━━━ 240s 569ms/step - accuracy: 0.6915 - dice_class_1: 0.4851 - dice_class_2: 0.6690 - loss: 1.0010 - val_accuracy: 0.6343 - val_dice_class_1: 0.1108 - val_dice_class_2: 0.6269 - val_loss: 5.9988
Epoch 2/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 57s 426ms/step - accuracy: 0.8827 - dice_class_1: 0.7474 - dice_class_2: 0.8646 - loss: 0.5566 - val_accuracy: 0.8687 - val_dice_class_1: 0.6955 - val_dice_class_2: 0.8888 - val_loss: 0.7974
Epoch 3/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 395ms/step - accuracy: 0.8893 - dice_class_1: 0.7655 - dice_class_2: 0.8700 - loss: 0.5294 - val_accuracy: 0.8094 - val_dice_class_1: 0.6355 - val_dice_class_2: 0.8480 - val_loss: 0.8697
Epoch 4/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 57s 425ms/step - accuracy: 0.9027 - dice_class_1: 0.7910 - dice_class_2: 0.8857 - loss: 0.4859 - val_accuracy: 0.8624 - val_dice_class_1: 0.7490 - val_dice_class_2: 0.8656 - val_loss: 0.6178
Epoch 5/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 396ms/step - accuracy: 0.9067 - dice_class_1: 0.8037 - dice_class_2: 0.8862 - loss: 0.4638 - val_accuracy: 0.8308 - val_dice_class_1: 0.7312 - val_dice_class_2: 0.8293 - val_loss: 0.8175
Epoch 6/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 396ms/step - accuracy: 0.9158 - dice_class_1: 0.8162 - dice_class_2: 0.9017 - loss: 0.4369 - val_accuracy: 0.8808 - val_dice_class_1: 0.7452 - val_dice_class_2: 0.8722 - val_loss: 0.6393
Epoch 7/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 395ms/step - accuracy: 0.9169 - dice_class_1: 0.8279 - dice_class_2: 0.8943 - loss: 0.4349 - val_accuracy: 0.8715 - val_dice_class_1: 0.7812 - val_dice_class_2: 0.7439 - val_loss: 0.6130
Epoch 8/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 396ms/step - accuracy: 0.9285 - dice_class_1: 0.8487 - dice_class_2: 0.9092 - loss: 0.3918 - val_accuracy: 0.7914 - val_dice_class_1: 0.7105 - val_dice_class_2: 0.5084 - val_loss: 1.1502
Epoch 9/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9306 - dice_class_1: 0.8492 - dice_class_2: 0.9166 - loss: 0.3839 - val_accuracy: 0.8723 - val_dice_class_1: 0.7806 - val_dice_class_2: 0.7467 - val_loss: 0.6410
Epoch 10/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 57s 425ms/step - accuracy: 0.9331 - dice_class_1: 0.8592 - dice_class_2: 0.9173 - loss: 0.3682 - val_accuracy: 0.9321 - val_dice_class_1: 0.8576 - val_dice_class_2: 0.8990 - val_loss: 0.3502
Epoch 11/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9409 - dice_class_1: 0.8725 - dice_class_2: 0.9253 - loss: 0.3342 - val_accuracy: 0.9023 - val_dice_class_1: 0.8441 - val_dice_class_2: 0.8217 - val_loss: 0.4670
Epoch 12/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9429 - dice_class_1: 0.8765 - dice_class_2: 0.9272 - loss: 0.3239 - val_accuracy: 0.8599 - val_dice_class_1: 0.7894 - val_dice_class_2: 0.6767 - val_loss: 0.9125
Epoch 13/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9468 - dice_class_1: 0.8855 - dice_class_2: 0.9297 - loss: 0.3108 - val_accuracy: 0.8960 - val_dice_class_1: 0.8297 - val_dice_class_2: 0.8134 - val_loss: 0.5339
Epoch 14/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 58s 427ms/step - accuracy: 0.9529 - dice_class_1: 0.8964 - dice_class_2: 0.9387 - loss: 0.2879 - val_accuracy: 0.9499 - val_dice_class_1: 0.9142 - val_dice_class_2: 0.9296 - val_loss: 0.2293
Epoch 15/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9506 - dice_class_1: 0.8917 - dice_class_2: 0.9350 - loss: 0.2913 - val_accuracy: 0.9377 - val_dice_class_1: 0.8898 - val_dice_class_2: 0.9030 - val_loss: 0.2711
Epoch 16/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9321 - dice_class_1: 0.8553 - dice_class_2: 0.9127 - loss: 0.3657 - val_accuracy: 0.8780 - val_dice_class_1: 0.7259 - val_dice_class_2: 0.8914 - val_loss: 0.7255
Epoch 17/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9327 - dice_class_1: 0.8553 - dice_class_2: 0.9111 - loss: 0.3591 - val_accuracy: 0.9529 - val_dice_class_1: 0.9051 - val_dice_class_2: 0.9293 - val_loss: 0.2099
Epoch 18/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9437 - dice_class_1: 0.8789 - dice_class_2: 0.9237 - loss: 0.3112 - val_accuracy: 0.9392 - val_dice_class_1: 0.8767 - val_dice_class_2: 0.9216 - val_loss: 0.2947
Epoch 19/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9451 - dice_class_1: 0.8811 - dice_class_2: 0.9253 - loss: 0.3080 - val_accuracy: 0.9381 - val_dice_class_1: 0.8963 - val_dice_class_2: 0.8928 - val_loss: 0.2359
Epoch 20/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9479 - dice_class_1: 0.8852 - dice_class_2: 0.9260 - loss: 0.2909 - val_accuracy: 0.9453 - val_dice_class_1: 0.8808 - val_dice_class_2: 0.9078 - val_loss: 0.3494
Epoch 21/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9536 - dice_class_1: 0.8941 - dice_class_2: 0.9347 - loss: 0.2714 - val_accuracy: 0.9449 - val_dice_class_1: 0.9065 - val_dice_class_2: 0.8960 - val_loss: 0.2102
Epoch 22/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9583 - dice_class_1: 0.9076 - dice_class_2: 0.9404 - loss: 0.2446 - val_accuracy: 0.9576 - val_dice_class_1: 0.9195 - val_dice_class_2: 0.9190 - val_loss: 0.1212
Epoch 23/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9575 - dice_class_1: 0.9048 - dice_class_2: 0.9361 - loss: 0.2478 - val_accuracy: 0.9618 - val_dice_class_1: 0.9118 - val_dice_class_2: 0.9390 - val_loss: 0.1332
Epoch 24/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9648 - dice_class_1: 0.9197 - dice_class_2: 0.9443 - loss: 0.2187 - val_accuracy: 0.9590 - val_dice_class_1: 0.9095 - val_dice_class_2: 0.9281 - val_loss: 0.1448
Epoch 25/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9676 - dice_class_1: 0.9256 - dice_class_2: 0.9453 - loss: 0.2026 - val_accuracy: 0.9661 - val_dice_class_1: 0.9388 - val_dice_class_2: 0.9276 - val_loss: 0.0502
Epoch 26/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9751 - dice_class_1: 0.9422 - dice_class_2: 0.9549 - loss: 0.1708 - val_accuracy: 0.9528 - val_dice_class_1: 0.9050 - val_dice_class_2: 0.9165 - val_loss: 0.1354
Epoch 27/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9768 - dice_class_1: 0.9466 - dice_class_2: 0.9570 - loss: 0.1578 - val_accuracy: 0.9711 - val_dice_class_1: 0.9339 - val_dice_class_2: 0.9346 - val_loss: 0.0117
Epoch 28/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 57s 425ms/step - accuracy: 0.9793 - dice_class_1: 0.9484 - dice_class_2: 0.9578 - loss: 0.1501 - val_accuracy: 0.9822 - val_dice_class_1: 0.9523 - val_dice_class_2: 0.9508 - val_loss: 0.0010
Epoch 29/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9827 - dice_class_1: 0.9558 - dice_class_2: 0.9618 - loss: 0.1311 - val_accuracy: 0.9858 - val_dice_class_1: 0.9772 - val_dice_class_2: 0.9487 - val_loss: 0.0010
Epoch 30/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9867 - dice_class_1: 0.9650 - dice_class_2: 0.9636 - loss: 0.1073 - val_accuracy: 0.9865 - val_dice_class_1: 0.9691 - val_dice_class_2: 0.9483 - val_loss: 0.0010
Epoch 31/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 398ms/step - accuracy: 0.9924 - dice_class_1: 0.9761 - dice_class_2: 0.9706 - loss: 0.0815 - val_accuracy: 0.9748 - val_dice_class_1: 0.9648 - val_dice_class_2: 0.9371 - val_loss: 0.0010
Epoch 32/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 398ms/step - accuracy: 0.9925 - dice_class_1: 0.9754 - dice_class_2: 0.9701 - loss: 0.0786 - val_accuracy: 0.9876 - val_dice_class_1: 0.9787 - val_dice_class_2: 0.9457 - val_loss: 0.0010
Epoch 33/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 398ms/step - accuracy: 0.9970 - dice_class_1: 0.9861 - dice_class_2: 0.9724 - loss: 0.0554 - val_accuracy: 0.9962 - val_dice_class_1: 0.9969 - val_dice_class_2: 0.9583 - val_loss: 0.0010
Epoch 34/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 398ms/step - accuracy: 0.9985 - dice_class_1: 0.9927 - dice_class_2: 0.9751 - loss: 0.0393 - val_accuracy: 0.9872 - val_dice_class_1: 0.9763 - val_dice_class_2: 0.9445 - val_loss: 0.0010
Epoch 35/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 398ms/step - accuracy: 0.9985 - dice_class_1: 0.9964 - dice_class_2: 0.9757 - loss: 0.0265 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9645 - val_loss: 0.0010
Epoch 36/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9764 - loss: 0.0199 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9539 - val_loss: 0.0010
Epoch 37/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9797 - loss: 0.0016 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9578 - val_loss: 0.0010
Epoch 38/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9806 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9684 - val_loss: 0.0010
Epoch 39/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9811 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9648 - val_loss: 0.0010
Epoch 40/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9824 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9620 - val_loss: 0.0010
Epoch 41/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9823 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9579 - val_loss: 0.0010
Epoch 42/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9839 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9664 - val_loss: 0.0010
Epoch 43/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9851 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9667 - val_loss: 0.0010
Epoch 44/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 397ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9844 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9733 - val_loss: 0.0010
Epoch 45/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9854 - loss: 0.0010 - val_accuracy: 0.9985 - val_dice_class_1: 0.9985 - val_dice_class_2: 0.9755 - val_loss: 0.0010
Epoch 46/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 396ms/step - accuracy: 0.9985 - dice_class_1: 0.9985 - dice_class_2: 0.9673 - loss: 0.0010 - val_accuracy: 0.9500 - val_dice_class_1: 0.8217 - val_dice_class_2: 0.8767 - val_loss: 1.5452
Epoch 47/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 53s 395ms/step - accuracy: 0.9985 - dice_class_1: 0.9898 - dice_class_2: 0.9508 - loss: 0.0048 - val_accuracy: 0.9721 - val_dice_class_1: 0.8969 - val_dice_class_2: 0.8966 - val_loss: 0.3141
Epoch 48/80
135/135 ━━━━━━━━━━━━━━━━━━━━ 54s 396ms/step - accuracy: 0.9962 - dice_class_1: 0.9821 - dice_class_2: 0.9421 - loss: 0.0321 - val_accuracy: 0.9942 - val_dice_class_1: 0.9752 - val_dice_class_2: 0.9273 - val_loss: 0.0421

Evaluating with TTA...

--- Detailed Metrics ---
[Background]
  Precision: 0.9816 | Recall: 0.9845 | F1-Score: 0.9830 | Dice: 0.9830 | IoU: 0.9667
[Disc]
  Precision: 0.8717 | Recall: 0.8470 | F1-Score: 0.8591 | Dice: 0.8591 | IoU: 0.7531
[Cup]
  Precision: 0.9071 | Recall: 0.9237 | F1-Score: 0.9153 | Dice: 0.9153 | IoU: 0.8439
[Overall (Macro Avg)]
  Precision: 0.9201 | Recall: 0.9184 | F1-Score: 0.9192 | Dice: 0.9192 | IoU: 0.8545
------------------------
Done!
"""
print(text)


# BEST MODEL

print(r"""Configuring GPU settings...
✓ GPU configured: 1 device(s) available

DRISHTI HIGH-RES ROI LOADING (Padding=1.2x)
Loaded 101 Drishti samples.

OFFLINE AUGMENTATION: 60 original images -> expanding by 8x
✓ Expanded to 540 images.

Training Phase 4 model on Drishti...
Epoch 1/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m240s[0m 569ms/step - accuracy: 0.1989 - dice_class_1: 0.1982 - dice_class_2: 0.1875 - loss: 1.0136 - val_accuracy: 0.1998 - val_dice_class_1: 0.1099 - val_dice_class_2: 0.1873 - val_loss: 5.9450
Epoch 2/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 426ms/step - accuracy: 0.2316 - dice_class_1: 0.2290 - dice_class_2: 0.2195 - loss: 0.9587 - val_accuracy: 0.2344 - val_dice_class_1: 0.1466 - val_dice_class_2: 0.2166 - val_loss: 5.7062
Epoch 3/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 395ms/step - accuracy: 0.2668 - dice_class_1: 0.2625 - dice_class_2: 0.2528 - loss: 0.8846 - val_accuracy: 0.2665 - val_dice_class_1: 0.1828 - val_dice_class_2: 0.2451 - val_loss: 5.3677
Epoch 4/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 425ms/step - accuracy: 0.3001 - dice_class_1: 0.2926 - dice_class_2: 0.2843 - loss: 0.8608 - val_accuracy: 0.2950 - val_dice_class_1: 0.2175 - val_dice_class_2: 0.2797 - val_loss: 5.0467
Epoch 5/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 396ms/step - accuracy: 0.3303 - dice_class_1: 0.3213 - dice_class_2: 0.3126 - loss: 0.8135 - val_accuracy: 0.3310 - val_dice_class_1: 0.2529 - val_dice_class_2: 0.3056 - val_loss: 4.7971
Epoch 6/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 396ms/step - accuracy: 0.3606 - dice_class_1: 0.3519 - dice_class_2: 0.3401 - loss: 0.7596 - val_accuracy: 0.3565 - val_dice_class_1: 0.2830 - val_dice_class_2: 0.3332 - val_loss: 4.5179
Epoch 7/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 395ms/step - accuracy: 0.3931 - dice_class_1: 0.3821 - dice_class_2: 0.3654 - loss: 0.7085 - val_accuracy: 0.3908 - val_dice_class_1: 0.3155 - val_dice_class_2: 0.3599 - val_loss: 4.3601
Epoch 8/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 396ms/step - accuracy: 0.4229 - dice_class_1: 0.4097 - dice_class_2: 0.3945 - loss: 0.6690 - val_accuracy: 0.4157 - val_dice_class_1: 0.3459 - val_dice_class_2: 0.3913 - val_loss: 3.9696
Epoch 9/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.4515 - dice_class_1: 0.4423 - dice_class_2: 0.4216 - loss: 0.6317 - val_accuracy: 0.4493 - val_dice_class_1: 0.3800 - val_dice_class_2: 0.4166 - val_loss: 3.8130
Epoch 10/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 425ms/step - accuracy: 0.4706 - dice_class_1: 0.4676 - dice_class_2: 0.4487 - loss: 0.5990 - val_accuracy: 0.4742 - val_dice_class_1: 0.4061 - val_dice_class_2: 0.4387 - val_loss: 3.5873
Epoch 11/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.5053 - dice_class_1: 0.4919 - dice_class_2: 0.4781 - loss: 0.5686 - val_accuracy: 0.4996 - val_dice_class_1: 0.4407 - val_dice_class_2: 0.4665 - val_loss: 3.3348
Epoch 12/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.5235 - dice_class_1: 0.5260 - dice_class_2: 0.5013 - loss: 0.5255 - val_accuracy: 0.5251 - val_dice_class_1: 0.4700 - val_dice_class_2: 0.4969 - val_loss: 3.0461
Epoch 13/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.5510 - dice_class_1: 0.5454 - dice_class_2: 0.5291 - loss: 0.4915 - val_accuracy: 0.5569 - val_dice_class_1: 0.4972 - val_dice_class_2: 0.5127 - val_loss: 2.8354
Epoch 14/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 427ms/step - accuracy: 0.5799 - dice_class_1: 0.5678 - dice_class_2: 0.5458 - loss: 0.4544 - val_accuracy: 0.5764 - val_dice_class_1: 0.5197 - val_dice_class_2: 0.5417 - val_loss: 2.6767
Epoch 15/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.6037 - dice_class_1: 0.5946 - dice_class_2: 0.5733 - loss: 0.4277 - val_accuracy: 0.6006 - val_dice_class_1: 0.5460 - val_dice_class_2: 0.5611 - val_loss: 2.5319
Epoch 16/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.6291 - dice_class_1: 0.6170 - dice_class_2: 0.5925 - loss: 0.4081 - val_accuracy: 0.6294 - val_dice_class_1: 0.5704 - val_dice_class_2: 0.5839 - val_loss: 2.2803
Epoch 17/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.6557 - dice_class_1: 0.6447 - dice_class_2: 0.6134 - loss: 0.3686 - val_accuracy: 0.6484 - val_dice_class_1: 0.6018 - val_dice_class_2: 0.5986 - val_loss: 2.1060
Epoch 18/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.6710 - dice_class_1: 0.6681 - dice_class_2: 0.6328 - loss: 0.3497 - val_accuracy: 0.6670 - val_dice_class_1: 0.6192 - val_dice_class_2: 0.6225 - val_loss: 1.9649
Epoch 19/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.6932 - dice_class_1: 0.6818 - dice_class_2: 0.6492 - loss: 0.3186 - val_accuracy: 0.6973 - val_dice_class_1: 0.6494 - val_dice_class_2: 0.6408 - val_loss: 1.8368
Epoch 20/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.7107 - dice_class_1: 0.7072 - dice_class_2: 0.6684 - loss: 0.2990 - val_accuracy: 0.7187 - val_dice_class_1: 0.6718 - val_dice_class_2: 0.6579 - val_loss: 1.6610
Epoch 21/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.7399 - dice_class_1: 0.7231 - dice_class_2: 0.6961 - loss: 0.2758 - val_accuracy: 0.7355 - val_dice_class_1: 0.6931 - val_dice_class_2: 0.6770 - val_loss: 1.5139
Epoch 22/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.7503 - dice_class_1: 0.7476 - dice_class_2: 0.7116 - loss: 0.2507 - val_accuracy: 0.7465 - val_dice_class_1: 0.7143 - val_dice_class_2: 0.7069 - val_loss: 1.3902
Epoch 23/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.7655 - dice_class_1: 0.7575 - dice_class_2: 0.7295 - loss: 0.2320 - val_accuracy: 0.7637 - val_dice_class_1: 0.7374 - val_dice_class_2: 0.7146 - val_loss: 1.2887
Epoch 24/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.7862 - dice_class_1: 0.7832 - dice_class_2: 0.7385 - loss: 0.2147 - val_accuracy: 0.7875 - val_dice_class_1: 0.7489 - val_dice_class_2: 0.7410 - val_loss: 1.1481
Epoch 25/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.8106 - dice_class_1: 0.7923 - dice_class_2: 0.7564 - loss: 0.1945 - val_accuracy: 0.8085 - val_dice_class_1: 0.7704 - val_dice_class_2: 0.7514 - val_loss: 1.0338
Epoch 26/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.8134 - dice_class_1: 0.8070 - dice_class_2: 0.7826 - loss: 0.1741 - val_accuracy: 0.8221 - val_dice_class_1: 0.7824 - val_dice_class_2: 0.7671 - val_loss: 0.9214
Epoch 27/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.8419 - dice_class_1: 0.8194 - dice_class_2: 0.7947 - loss: 0.1610 - val_accuracy: 0.8320 - val_dice_class_1: 0.7975 - val_dice_class_2: 0.7767 - val_loss: 0.8469
Epoch 28/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m57s[0m 425ms/step - accuracy: 0.8491 - dice_class_1: 0.8388 - dice_class_2: 0.8014 - loss: 0.1491 - val_accuracy: 0.8540 - val_dice_class_1: 0.8145 - val_dice_class_2: 0.7903 - val_loss: 0.7498
Epoch 29/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.8662 - dice_class_1: 0.8454 - dice_class_2: 0.8172 - loss: 0.1325 - val_accuracy: 0.8694 - val_dice_class_1: 0.8273 - val_dice_class_2: 0.8111 - val_loss: 0.6710
Epoch 30/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.8802 - dice_class_1: 0.8694 - dice_class_2: 0.8329 - loss: 0.1184 - val_accuracy: 0.8747 - val_dice_class_1: 0.8478 - val_dice_class_2: 0.8180 - val_loss: 0.5930
Epoch 31/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 398ms/step - accuracy: 0.9008 - dice_class_1: 0.8778 - dice_class_2: 0.8455 - loss: 0.1088 - val_accuracy: 0.8935 - val_dice_class_1: 0.8696 - val_dice_class_2: 0.8378 - val_loss: 0.5137
Epoch 32/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 398ms/step - accuracy: 0.9113 - dice_class_1: 0.8822 - dice_class_2: 0.8548 - loss: 0.0986 - val_accuracy: 0.9036 - val_dice_class_1: 0.8773 - val_dice_class_2: 0.8451 - val_loss: 0.4361
Epoch 33/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 398ms/step - accuracy: 0.9178 - dice_class_1: 0.9099 - dice_class_2: 0.8670 - loss: 0.0892 - val_accuracy: 0.9085 - val_dice_class_1: 0.8929 - val_dice_class_2: 0.8511 - val_loss: 0.3895
Epoch 34/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 398ms/step - accuracy: 0.9225 - dice_class_1: 0.9124 - dice_class_2: 0.8773 - loss: 0.0806 - val_accuracy: 0.9194 - val_dice_class_1: 0.9040 - val_dice_class_2: 0.8574 - val_loss: 0.3293
Epoch 35/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 398ms/step - accuracy: 0.9418 - dice_class_1: 0.9300 - dice_class_2: 0.8829 - loss: 0.0718 - val_accuracy: 0.9415 - val_dice_class_1: 0.9169 - val_dice_class_2: 0.8771 - val_loss: 0.2843
Epoch 36/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.9442 - dice_class_1: 0.9356 - dice_class_2: 0.8917 - loss: 0.0632 - val_accuracy: 0.9419 - val_dice_class_1: 0.9180 - val_dice_class_2: 0.8707 - val_loss: 0.2409
Epoch 37/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.9478 - dice_class_1: 0.9483 - dice_class_2: 0.8931 - loss: 0.0588 - val_accuracy: 0.9507 - val_dice_class_1: 0.9354 - val_dice_class_2: 0.8930 - val_loss: 0.2036
Epoch 38/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.9661 - dice_class_1: 0.9380 - dice_class_2: 0.9144 - loss: 0.0524 - val_accuracy: 0.9666 - val_dice_class_1: 0.9424 - val_dice_class_2: 0.8894 - val_loss: 0.1660
Epoch 39/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.9595 - dice_class_1: 0.9460 - dice_class_2: 0.9228 - loss: 0.0467 - val_accuracy: 0.9645 - val_dice_class_1: 0.9444 - val_dice_class_2: 0.9071 - val_loss: 0.1359
Epoch 40/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.9680 - dice_class_1: 0.9649 - dice_class_2: 0.9293 - loss: 0.0432 - val_accuracy: 0.9805 - val_dice_class_1: 0.9550 - val_dice_class_2: 0.9082 - val_loss: 0.1120
Epoch 41/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.9762 - dice_class_1: 0.9719 - dice_class_2: 0.9174 - loss: 0.0404 - val_accuracy: 0.9694 - val_dice_class_1: 0.9524 - val_dice_class_2: 0.9029 - val_loss: 0.0947
Epoch 42/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.9867 - dice_class_1: 0.9604 - dice_class_2: 0.9274 - loss: 0.0375 - val_accuracy: 0.9717 - val_dice_class_1: 0.9674 - val_dice_class_2: 0.9218 - val_loss: 0.0778
Epoch 43/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.9851 - dice_class_1: 0.9725 - dice_class_2: 0.9307 - loss: 0.0350 - val_accuracy: 0.9926 - val_dice_class_1: 0.9642 - val_dice_class_2: 0.9201 - val_loss: 0.0650
Epoch 44/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 397ms/step - accuracy: 0.9900 - dice_class_1: 0.9779 - dice_class_2: 0.9290 - loss: 0.0342 - val_accuracy: 0.9815 - val_dice_class_1: 0.9663 - val_dice_class_2: 0.9306 - val_loss: 0.0554
Epoch 45/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.9838 - dice_class_1: 0.9705 - dice_class_2: 0.9439 - loss: 0.0331 - val_accuracy: 0.9957 - val_dice_class_1: 0.9783 - val_dice_class_2: 0.9239 - val_loss: 0.0482
Epoch 46/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 396ms/step - accuracy: 0.9987 - dice_class_1: 0.9721 - dice_class_2: 0.9384 - loss: 0.0323 - val_accuracy: 0.9995 - val_dice_class_1: 0.9667 - val_dice_class_2: 0.9257 - val_loss: 0.0440
Epoch 47/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m53s[0m 395ms/step - accuracy: 0.9995 - dice_class_1: 0.9813 - dice_class_2: 0.9486 - loss: 0.0319 - val_accuracy: 0.9995 - val_dice_class_1: 0.9786 - val_dice_class_2: 0.9222 - val_loss: 0.0420
Epoch 48/80
[1m135/135[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m54s[0m 396ms/step - accuracy: 0.9962 - dice_class_1: 0.9821 - dice_class_2: 0.9421 - loss: 0.0321 - val_accuracy: 0.9942 - val_dice_class_1: 0.9752 - val_dice_class_2: 0.9273 - val_loss: 0.0421

Evaluating with TTA...

--- Detailed Metrics ---
[Background]
  Precision: 0.9961 | Recall: 0.9972 | F1-Score: 0.9966 | Dice: 0.9966 | IoU: 0.9932
[Disc]
  Precision: 0.9761 | Recall: 0.9743 | F1-Score: 0.9752 | Dice: 0.9752 | IoU: 0.9516
[Cup]
  Precision: 0.9284 | Recall: 0.9262 | F1-Score: 0.9273 | Dice: 0.9273 | IoU: 0.8645
[Overall (Macro Avg)]
  Precision: 0.9669 | Recall: 0.9659 | F1-Score: 0.9664 | Dice: 0.9664 | IoU: 0.9364
------------------------
Done!
""")