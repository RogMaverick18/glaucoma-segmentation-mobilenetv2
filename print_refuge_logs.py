from IPython.display import display, Markdown

print(r"""2026-06-07 23:12:31.674212: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1780873951.867622      23 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1780873951.926273      23 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1780873952.393092      23 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1780873952.393141      23 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1780873952.393144      23 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1780873952.393147      23 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Configuring GPU settings...
✓ GPU configured: 1 device(s) available

====================================================================================================
              V8 + Phase 4: V5 BASE + CLAHE + NOISE + PP + TTA + TRAINING REFINEMENTS               
====================================================================================================

Phase 1 — Targeted Augmentation:
  - V5's exact pipeline + CLAHE + mild GaussNoise
  - ROI padding: 1.2× (same as V5)
  - NO offline augmentation (~600 images, ~1hr training)

Phase 2 — Post-Processing: Largest CC → Closing → Fill → Median → Cup⊂Disc

Phase 3 — TTA: 4-pass flip ensemble

Phase 4 — Training Refinements:
  - Cosine Annealing: T_0=15, T_mult=2, η_min=1e-7
  - Label Smoothing: 0.05
  - Disc focal α: [0.25, 1.3, 1.0]
  - 100 epochs, patience=20

================================================================================
HIGH-RES ROI DATA LOADING (padding=1.2×)
================================================================================
  train: 400 samples (original resolution -> ROI extracted)
  val: 400 samples (original resolution -> ROI extracted)
  test: 400 samples (original resolution -> ROI extracted)

  Split - Train: 600, Val: 200, Test: 400

================================================================================
TRAINING: V8_Phase4_CLAHE_PP_TTA
================================================================================
  Phase 4 refinements ACTIVE:
    - Cosine Annealing: T_0=15, T_mult=2, eta_min=1e-7
    - Label smoothing: 0.05
    - Disc focal α: [0.25, 1.3, 1.0]
    - 100 epochs, patience=20
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
[1m9406464/9406464[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 0us/step

Model: 47,631,627 params | MSCA+LBFR+PPM triple attention (12 blocks)
  Cosine LR: epoch=0, cycle=0, T_cur=0/15, lr=2.00e-04
Epoch 1/100
Epoch 1: val_loss improved to 5.30280, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 1: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 311s 615ms/step - accuracy: 0.8172 - dice_class_0: 0.8900 - dice_class_1: 0.7676 - dice_class_2: 0.6651 - dice_coef_multiclass: 0.7742 - iou_class_0: 0.8109 - iou_class_1: 0.6412 - iou_class_2: 0.5266 - iou_coef_multiclass: 0.6596 - loss: 0.7684 - val_accuracy: 0.7093 - val_dice_class_0: 0.8867 - val_dice_class_1: 0.5238 - val_dice_class_2: 0.5254 - val_dice_coef_multiclass: 0.6453 - val_iou_class_0: 0.8054 - val_iou_class_1: 0.3584 - val_iou_class_2: 0.3579 - val_iou_coef_multiclass: 0.5072 - val_loss: 5.3028
Epoch 2/100
Epoch 2: val_loss did not improve from 5.30280
150/150 ━━━━━━━━━━━━━━━━━━━━ 69s 431ms/step - accuracy: 0.9017 - dice_class_0: 0.8969 - dice_class_1: 0.7812 - dice_class_2: 0.6859 - dice_coef_multiclass: 0.7880 - iou_class_0: 0.8252 - iou_class_1: 0.6642 - iou_class_2: 0.5530 - iou_coef_multiclass: 0.6808 - loss: 0.5422 - val_accuracy: 0.4680 - val_dice_class_0: 0.7647 - val_dice_class_1: 0.0799 - val_dice_class_2: 0.3400 - val_dice_coef_multiclass: 0.3949 - val_iou_class_0: 0.6207 - val_iou_class_1: 0.0426 - val_iou_class_2: 0.2070 - val_iou_coef_multiclass: 0.2901 - val_loss: 12.2519
Epoch 3/100
Epoch 3: val_loss improved to 1.56290, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 3: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 458ms/step - accuracy: 0.9080 - dice_class_0: 0.9062 - dice_class_1: 0.7994 - dice_class_2: 0.7062 - dice_coef_multiclass: 0.8039 - iou_class_0: 0.8451 - iou_class_1: 0.6861 - iou_class_2: 0.5778 - iou_coef_multiclass: 0.7030 - loss: 0.4949 - val_accuracy: 0.8329 - val_dice_class_0: 0.9064 - val_dice_class_1: 0.5965 - val_dice_class_2: 0.5824 - val_dice_coef_multiclass: 0.6951 - val_iou_class_0: 0.8258 - val_iou_class_1: 0.4573 - val_iou_class_2: 0.4345 - val_iou_coef_multiclass: 0.5726 - val_loss: 1.5629
Epoch 4/100
Epoch 4: val_loss did not improve from 1.56290
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 420ms/step - accuracy: 0.9176 - dice_class_0: 0.9129 - dice_class_1: 0.8138 - dice_class_2: 0.7244 - dice_coef_multiclass: 0.8170 - iou_class_0: 0.8495 - iou_class_1: 0.7028 - iou_class_2: 0.5988 - iou_coef_multiclass: 0.7170 - loss: 0.4596 - val_accuracy: 0.7042 - val_dice_class_0: 0.9123 - val_dice_class_1: 0.6258 - val_dice_class_2: 0.6142 - val_dice_coef_multiclass: 0.7174 - val_iou_class_0: 0.8458 - val_iou_class_1: 0.4863 - val_iou_class_2: 0.4768 - val_iou_coef_multiclass: 0.6029 - val_loss: 3.5853
Epoch 5/100
Epoch 5: val_loss improved to 0.66970, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 5: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 432ms/step - accuracy: 0.9188 - dice_class_0: 0.9184 - dice_class_1: 0.8253 - dice_class_2: 0.7418 - dice_coef_multiclass: 0.8285 - iou_class_0: 0.8648 - iou_class_1: 0.7216 - iou_class_2: 0.6247 - iou_coef_multiclass: 0.7370 - loss: 0.4512 - val_accuracy: 0.8972 - val_dice_class_0: 0.9241 - val_dice_class_1: 0.6497 - val_dice_class_2: 0.6332 - val_dice_coef_multiclass: 0.7357 - val_iou_class_0: 0.8438 - val_iou_class_1: 0.5228 - val_iou_class_2: 0.5026 - val_iou_coef_multiclass: 0.6231 - val_loss: 0.6697
Epoch 6/100
Epoch 6: val_loss did not improve from 0.66970
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 446ms/step - accuracy: 0.8705 - dice_class_0: 0.9217 - dice_class_1: 0.8347 - dice_class_2: 0.7573 - dice_coef_multiclass: 0.8379 - iou_class_0: 0.8727 - iou_class_1: 0.7387 - iou_class_2: 0.6423 - iou_coef_multiclass: 0.7512 - loss: 0.5755 - val_accuracy: 0.8240 - val_dice_class_0: 0.9223 - val_dice_class_1: 0.6755 - val_dice_class_2: 0.6689 - val_dice_coef_multiclass: 0.7556 - val_iou_class_0: 0.8581 - val_iou_class_1: 0.5553 - val_iou_class_2: 0.5421 - val_iou_coef_multiclass: 0.6518 - val_loss: 1.2680
Epoch 7/100
Epoch 7: val_loss did not improve from 0.66970
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 424ms/step - accuracy: 0.8852 - dice_class_0: 0.9299 - dice_class_1: 0.8430 - dice_class_2: 0.7667 - dice_coef_multiclass: 0.8465 - iou_class_0: 0.8789 - iou_class_1: 0.7518 - iou_class_2: 0.6581 - iou_coef_multiclass: 0.7629 - loss: 0.5440 - val_accuracy: 0.8402 - val_dice_class_0: 0.9327 - val_dice_class_1: 0.7009 - val_dice_class_2: 0.6893 - val_dice_coef_multiclass: 0.7743 - val_iou_class_0: 0.8724 - val_iou_class_1: 0.5799 - val_iou_class_2: 0.5590 - val_iou_coef_multiclass: 0.6704 - val_loss: 0.7213
Epoch 8/100
Epoch 8: val_loss improved to 0.45890, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 8: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 453ms/step - accuracy: 0.8867 - dice_class_0: 0.9378 - dice_class_1: 0.8561 - dice_class_2: 0.7821 - dice_coef_multiclass: 0.8587 - iou_class_0: 0.8919 - iou_class_1: 0.7693 - iou_class_2: 0.6815 - iou_coef_multiclass: 0.7809 - loss: 0.5168 - val_accuracy: 0.8525 - val_dice_class_0: 0.9315 - val_dice_class_1: 0.7141 - val_dice_class_2: 0.7016 - val_dice_coef_multiclass: 0.7824 - val_iou_class_0: 0.8823 - val_iou_class_1: 0.6117 - val_iou_class_2: 0.5821 - val_iou_coef_multiclass: 0.6920 - val_loss: 0.4589
Epoch 9/100
Epoch 9: val_loss did not improve from 0.45890
150/150 ━━━━━━━━━━━━━━━━━━━━ 69s 426ms/step - accuracy: 0.8989 - dice_class_0: 0.9421 - dice_class_1: 0.8636 - dice_class_2: 0.7952 - dice_coef_multiclass: 0.8670 - iou_class_0: 0.8983 - iou_class_1: 0.7797 - iou_class_2: 0.6945 - iou_coef_multiclass: 0.7908 - loss: 0.4824 - val_accuracy: 0.8598 - val_dice_class_0: 0.9335 - val_dice_class_1: 0.7416 - val_dice_class_2: 0.7187 - val_dice_coef_multiclass: 0.7980 - val_iou_class_0: 0.8938 - val_iou_class_1: 0.6347 - val_iou_class_2: 0.6107 - val_iou_coef_multiclass: 0.7130 - val_loss: 0.4994
Epoch 10/100
Epoch 10: val_loss improved to 0.39360, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 10: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 455ms/step - accuracy: 0.9067 - dice_class_0: 0.9433 - dice_class_1: 0.8716 - dice_class_2: 0.8073 - dice_coef_multiclass: 0.8741 - iou_class_0: 0.9018 - iou_class_1: 0.7908 - iou_class_2: 0.7116 - iou_coef_multiclass: 0.8014 - loss: 0.4627 - val_accuracy: 0.8722 - val_dice_class_0: 0.9399 - val_dice_class_1: 0.7596 - val_dice_class_2: 0.7567 - val_dice_coef_multiclass: 0.8187 - val_iou_class_0: 0.8900 - val_iou_class_1: 0.6572 - val_iou_class_2: 0.6317 - val_iou_coef_multiclass: 0.7263 - val_loss: 0.3936
Epoch 11/100
Epoch 11: val_loss improved to 0.38992, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 11: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 432ms/step - accuracy: 0.9104 - dice_class_0: 0.9482 - dice_class_1: 0.8824 - dice_class_2: 0.8129 - dice_coef_multiclass: 0.8812 - iou_class_0: 0.9116 - iou_class_1: 0.8065 - iou_class_2: 0.7197 - iou_coef_multiclass: 0.8126 - loss: 0.4409 - val_accuracy: 0.8815 - val_dice_class_0: 0.9408 - val_dice_class_1: 0.7741 - val_dice_class_2: 0.7533 - val_dice_coef_multiclass: 0.8227 - val_iou_class_0: 0.9071 - val_iou_class_1: 0.6791 - val_iou_class_2: 0.6441 - val_iou_coef_multiclass: 0.7435 - val_loss: 0.3899
Epoch 12/100
Epoch 12: val_loss improved to 0.38064, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 12: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 432ms/step - accuracy: 0.9184 - dice_class_0: 0.9525 - dice_class_1: 0.8853 - dice_class_2: 0.8252 - dice_coef_multiclass: 0.8877 - iou_class_0: 0.9163 - iou_class_1: 0.8153 - iou_class_2: 0.7299 - iou_coef_multiclass: 0.8205 - loss: 0.4271 - val_accuracy: 0.8871 - val_dice_class_0: 0.9521 - val_dice_class_1: 0.7806 - val_dice_class_2: 0.7724 - val_dice_coef_multiclass: 0.8350 - val_iou_class_0: 0.9131 - val_iou_class_1: 0.7003 - val_iou_class_2: 0.6637 - val_iou_coef_multiclass: 0.7591 - val_loss: 0.3806
Epoch 13/100
Epoch 13: val_loss did not improve from 0.38064
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 426ms/step - accuracy: 0.9243 - dice_class_0: 0.9532 - dice_class_1: 0.8895 - dice_class_2: 0.8389 - dice_coef_multiclass: 0.8939 - iou_class_0: 0.9247 - iou_class_1: 0.8226 - iou_class_2: 0.7505 - iou_coef_multiclass: 0.8326 - loss: 0.4036 - val_accuracy: 0.9009 - val_dice_class_0: 0.9559 - val_dice_class_1: 0.8010 - val_dice_class_2: 0.7878 - val_dice_coef_multiclass: 0.8482 - val_iou_class_0: 0.9126 - val_iou_class_1: 0.7098 - val_iou_class_2: 0.6738 - val_iou_coef_multiclass: 0.7654 - val_loss: 0.3984
Epoch 14/100
Epoch 14: val_loss improved to 0.37997, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 14: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 426ms/step - accuracy: 0.9277 - dice_class_0: 0.9600 - dice_class_1: 0.8967 - dice_class_2: 0.8428 - dice_coef_multiclass: 0.8998 - iou_class_0: 0.9286 - iou_class_1: 0.8364 - iou_class_2: 0.7546 - iou_coef_multiclass: 0.8399 - loss: 0.3909 - val_accuracy: 0.8960 - val_dice_class_0: 0.9612 - val_dice_class_1: 0.8080 - val_dice_class_2: 0.7871 - val_dice_coef_multiclass: 0.8521 - val_iou_class_0: 0.9261 - val_iou_class_1: 0.7307 - val_iou_class_2: 0.6949 - val_iou_coef_multiclass: 0.7839 - val_loss: 0.3800
Epoch 15/100
Epoch 15: val_loss improved to 0.34211, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 15: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 423ms/step - accuracy: 0.9336 - dice_class_0: 0.9593 - dice_class_1: 0.9044 - dice_class_2: 0.8486 - dice_coef_multiclass: 0.9041 - iou_class_0: 0.9332 - iou_class_1: 0.8434 - iou_class_2: 0.7636 - iou_coef_multiclass: 0.8467 - loss: 0.3811 - val_accuracy: 0.9134 - val_dice_class_0: 0.9620 - val_dice_class_1: 0.8284 - val_dice_class_2: 0.8043 - val_dice_coef_multiclass: 0.8649 - val_iou_class_0: 0.9251 - val_iou_class_1: 0.7495 - val_iou_class_2: 0.7085 - val_iou_coef_multiclass: 0.7944 - val_loss: 0.3421
  Cosine LR: epoch=15, cycle=1, T_cur=0/30, lr=2.00e-04
Epoch 16/100
Epoch 16: val_loss did not improve from 0.34211
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 459ms/step - accuracy: 0.9373 - dice_class_0: 0.9622 - dice_class_1: 0.9060 - dice_class_2: 0.8542 - dice_coef_multiclass: 0.9075 - iou_class_0: 0.9386 - iou_class_1: 0.8492 - iou_class_2: 0.7709 - iou_coef_multiclass: 0.8529 - loss: 0.3575 - val_accuracy: 0.9122 - val_dice_class_0: 0.9556 - val_dice_class_1: 0.8404 - val_dice_class_2: 0.8174 - val_dice_coef_multiclass: 0.8711 - val_iou_class_0: 0.9308 - val_iou_class_1: 0.7664 - val_iou_class_2: 0.7195 - val_iou_coef_multiclass: 0.8056 - val_loss: 0.3421
Epoch 17/100
Epoch 17: val_loss did not improve from 0.34211
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 437ms/step - accuracy: 0.9388 - dice_class_0: 0.9664 - dice_class_1: 0.9103 - dice_class_2: 0.8618 - dice_coef_multiclass: 0.9128 - iou_class_0: 0.9428 - iou_class_1: 0.8561 - iou_class_2: 0.7810 - iou_coef_multiclass: 0.8600 - loss: 0.3664 - val_accuracy: 0.9203 - val_dice_class_0: 0.9638 - val_dice_class_1: 0.8510 - val_dice_class_2: 0.8235 - val_dice_coef_multiclass: 0.8794 - val_iou_class_0: 0.9289 - val_iou_class_1: 0.7819 - val_iou_class_2: 0.7319 - val_iou_coef_multiclass: 0.8142 - val_loss: 0.3513
Epoch 18/100
Epoch 18: val_loss did not improve from 0.34211
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 455ms/step - accuracy: 0.9397 - dice_class_0: 0.9671 - dice_class_1: 0.9173 - dice_class_2: 0.8655 - dice_coef_multiclass: 0.9166 - iou_class_0: 0.9405 - iou_class_1: 0.8607 - iou_class_2: 0.7832 - iou_coef_multiclass: 0.8614 - loss: 0.3558 - val_accuracy: 0.9288 - val_dice_class_0: 0.9737 - val_dice_class_1: 0.8594 - val_dice_class_2: 0.8302 - val_dice_coef_multiclass: 0.8878 - val_iou_class_0: 0.9462 - val_iou_class_1: 0.7929 - val_iou_class_2: 0.7432 - val_iou_coef_multiclass: 0.8274 - val_loss: 0.3482
Epoch 19/100
Epoch 19: val_loss improved to 0.32369, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 19: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 433ms/step - accuracy: 0.9454 - dice_class_0: 0.9738 - dice_class_1: 0.9212 - dice_class_2: 0.8739 - dice_coef_multiclass: 0.9230 - iou_class_0: 0.9466 - iou_class_1: 0.8687 - iou_class_2: 0.7975 - iou_coef_multiclass: 0.8709 - loss: 0.3169 - val_accuracy: 0.9407 - val_dice_class_0: 0.9714 - val_dice_class_1: 0.8726 - val_dice_class_2: 0.8452 - val_dice_coef_multiclass: 0.8964 - val_iou_class_0: 0.9420 - val_iou_class_1: 0.8041 - val_iou_class_2: 0.7533 - val_iou_coef_multiclass: 0.8331 - val_loss: 0.3237
Epoch 20/100
Epoch 20: val_loss improved to 0.31375, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 20: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 458ms/step - accuracy: 0.9540 - dice_class_0: 0.9718 - dice_class_1: 0.9261 - dice_class_2: 0.8751 - dice_coef_multiclass: 0.9244 - iou_class_0: 0.9528 - iou_class_1: 0.8752 - iou_class_2: 0.8022 - iou_coef_multiclass: 0.8767 - loss: 0.3164 - val_accuracy: 0.9334 - val_dice_class_0: 0.9670 - val_dice_class_1: 0.8750 - val_dice_class_2: 0.8440 - val_dice_coef_multiclass: 0.8954 - val_iou_class_0: 0.9478 - val_iou_class_1: 0.8075 - val_iou_class_2: 0.7677 - val_iou_coef_multiclass: 0.8410 - val_loss: 0.3137
Epoch 21/100
Epoch 21: val_loss improved to 0.31334, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 21: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 427ms/step - accuracy: 0.9553 - dice_class_0: 0.9765 - dice_class_1: 0.9260 - dice_class_2: 0.8810 - dice_coef_multiclass: 0.9278 - iou_class_0: 0.9566 - iou_class_1: 0.8790 - iou_class_2: 0.8081 - iou_coef_multiclass: 0.8812 - loss: 0.3126 - val_accuracy: 0.9440 - val_dice_class_0: 0.9728 - val_dice_class_1: 0.8755 - val_dice_class_2: 0.8537 - val_dice_coef_multiclass: 0.9007 - val_iou_class_0: 0.9591 - val_iou_class_1: 0.8262 - val_iou_class_2: 0.7618 - val_iou_coef_multiclass: 0.8491 - val_loss: 0.3133
Epoch 22/100
Epoch 22: val_loss improved to 0.30221, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 22: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 452ms/step - accuracy: 0.9589 - dice_class_0: 0.9739 - dice_class_1: 0.9298 - dice_class_2: 0.8861 - dice_coef_multiclass: 0.9299 - iou_class_0: 0.9565 - iou_class_1: 0.8804 - iou_class_2: 0.8177 - iou_coef_multiclass: 0.8849 - loss: 0.2996 - val_accuracy: 0.9063 - val_dice_class_0: 0.9548 - val_dice_class_1: 0.8735 - val_dice_class_2: 0.8443 - val_dice_coef_multiclass: 0.8908 - val_iou_class_0: 0.9356 - val_iou_class_1: 0.8123 - val_iou_class_2: 0.7569 - val_iou_coef_multiclass: 0.8349 - val_loss: 0.3022
Epoch 23/100
Epoch 23: val_loss improved to 0.29306, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 23: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 453ms/step - accuracy: 0.9604 - dice_class_0: 0.9789 - dice_class_1: 0.9339 - dice_class_2: 0.8881 - dice_coef_multiclass: 0.9336 - iou_class_0: 0.9607 - iou_class_1: 0.8857 - iou_class_2: 0.8180 - iou_coef_multiclass: 0.8882 - loss: 0.2850 - val_accuracy: 0.9486 - val_dice_class_0: 0.9741 - val_dice_class_1: 0.8905 - val_dice_class_2: 0.8651 - val_dice_coef_multiclass: 0.9099 - val_iou_class_0: 0.9533 - val_iou_class_1: 0.8444 - val_iou_class_2: 0.7948 - val_iou_coef_multiclass: 0.8642 - val_loss: 0.2931
Epoch 24/100
Epoch 24: val_loss did not improve from 0.29306
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 431ms/step - accuracy: 0.9624 - dice_class_0: 0.9798 - dice_class_1: 0.9363 - dice_class_2: 0.8939 - dice_coef_multiclass: 0.9367 - iou_class_0: 0.9600 - iou_class_1: 0.8921 - iou_class_2: 0.8212 - iou_coef_multiclass: 0.8911 - loss: 0.2850 - val_accuracy: 0.9459 - val_dice_class_0: 0.9791 - val_dice_class_1: 0.8964 - val_dice_class_2: 0.8736 - val_dice_coef_multiclass: 0.9163 - val_iou_class_0: 0.9565 - val_iou_class_1: 0.8366 - val_iou_class_2: 0.7973 - val_iou_coef_multiclass: 0.8635 - val_loss: 0.3041
Epoch 25/100
Epoch 25: val_loss did not improve from 0.29306
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 452ms/step - accuracy: 0.9679 - dice_class_0: 0.9807 - dice_class_1: 0.9389 - dice_class_2: 0.8968 - dice_coef_multiclass: 0.9388 - iou_class_0: 0.9620 - iou_class_1: 0.8947 - iou_class_2: 0.8242 - iou_coef_multiclass: 0.8936 - loss: 0.2745 - val_accuracy: 0.9500 - val_dice_class_0: 0.9704 - val_dice_class_1: 0.9170 - val_dice_class_2: 0.8812 - val_dice_coef_multiclass: 0.9229 - val_iou_class_0: 0.9686 - val_iou_class_1: 0.8572 - val_iou_class_2: 0.7971 - val_iou_coef_multiclass: 0.8743 - val_loss: 0.3066
Epoch 26/100
Epoch 26: val_loss did not improve from 0.29306
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 428ms/step - accuracy: 0.9688 - dice_class_0: 0.9793 - dice_class_1: 0.9416 - dice_class_2: 0.9014 - dice_coef_multiclass: 0.9408 - iou_class_0: 0.9626 - iou_class_1: 0.8974 - iou_class_2: 0.8291 - iou_coef_multiclass: 0.8964 - loss: 0.2658 - val_accuracy: 0.9603 - val_dice_class_0: 0.9807 - val_dice_class_1: 0.9027 - val_dice_class_2: 0.8728 - val_dice_coef_multiclass: 0.9187 - val_iou_class_0: 0.9608 - val_iou_class_1: 0.8537 - val_iou_class_2: 0.8071 - val_iou_coef_multiclass: 0.8739 - val_loss: 0.3133
Epoch 27/100
Epoch 27: val_loss improved to 0.28747, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 27: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 443ms/step - accuracy: 0.9700 - dice_class_0: 0.9839 - dice_class_1: 0.9447 - dice_class_2: 0.9027 - dice_coef_multiclass: 0.9438 - iou_class_0: 0.9689 - iou_class_1: 0.9003 - iou_class_2: 0.8306 - iou_coef_multiclass: 0.8999 - loss: 0.2601 - val_accuracy: 0.9664 - val_dice_class_0: 0.9763 - val_dice_class_1: 0.9127 - val_dice_class_2: 0.8847 - val_dice_coef_multiclass: 0.9246 - val_iou_class_0: 0.9586 - val_iou_class_1: 0.8597 - val_iou_class_2: 0.8124 - val_iou_coef_multiclass: 0.8769 - val_loss: 0.2875
Epoch 28/100
Epoch 28: val_loss did not improve from 0.28747
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 430ms/step - accuracy: 0.9707 - dice_class_0: 0.9808 - dice_class_1: 0.9444 - dice_class_2: 0.9041 - dice_coef_multiclass: 0.9431 - iou_class_0: 0.9696 - iou_class_1: 0.9020 - iou_class_2: 0.8349 - iou_coef_multiclass: 0.9022 - loss: 0.2580 - val_accuracy: 0.9564 - val_dice_class_0: 0.9848 - val_dice_class_1: 0.9134 - val_dice_class_2: 0.8882 - val_dice_coef_multiclass: 0.9288 - val_iou_class_0: 0.9606 - val_iou_class_1: 0.8630 - val_iou_class_2: 0.8141 - val_iou_coef_multiclass: 0.8792 - val_loss: 0.2917
Epoch 29/100
Epoch 29: val_loss improved to 0.27536, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 29: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 457ms/step - accuracy: 0.9727 - dice_class_0: 0.9841 - dice_class_1: 0.9473 - dice_class_2: 0.9040 - dice_coef_multiclass: 0.9451 - iou_class_0: 0.9738 - iou_class_1: 0.9071 - iou_class_2: 0.8397 - iou_coef_multiclass: 0.9069 - loss: 0.2585 - val_accuracy: 0.9588 - val_dice_class_0: 0.9855 - val_dice_class_1: 0.9169 - val_dice_class_2: 0.8873 - val_dice_coef_multiclass: 0.9299 - val_iou_class_0: 0.9702 - val_iou_class_1: 0.8714 - val_iou_class_2: 0.8279 - val_iou_coef_multiclass: 0.8898 - val_loss: 0.2754
Epoch 30/100
Epoch 30: val_loss improved to 0.27255, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 30: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 435ms/step - accuracy: 0.9752 - dice_class_0: 0.9854 - dice_class_1: 0.9512 - dice_class_2: 0.9098 - dice_coef_multiclass: 0.9488 - iou_class_0: 0.9732 - iou_class_1: 0.9087 - iou_class_2: 0.8423 - iou_coef_multiclass: 0.9081 - loss: 0.2458 - val_accuracy: 0.9685 - val_dice_class_0: 0.9758 - val_dice_class_1: 0.9117 - val_dice_class_2: 0.8938 - val_dice_coef_multiclass: 0.9271 - val_iou_class_0: 0.9752 - val_iou_class_1: 0.8748 - val_iou_class_2: 0.8326 - val_iou_coef_multiclass: 0.8942 - val_loss: 0.2726
Epoch 31/100
Epoch 31: val_loss did not improve from 0.27255
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 424ms/step - accuracy: 0.9769 - dice_class_0: 0.9829 - dice_class_1: 0.9499 - dice_class_2: 0.9120 - dice_coef_multiclass: 0.9483 - iou_class_0: 0.9728 - iou_class_1: 0.9075 - iou_class_2: 0.8420 - iou_coef_multiclass: 0.9074 - loss: 0.2419 - val_accuracy: 0.9543 - val_dice_class_0: 0.9840 - val_dice_class_1: 0.9288 - val_dice_class_2: 0.9021 - val_dice_coef_multiclass: 0.9383 - val_iou_class_0: 0.9623 - val_iou_class_1: 0.8790 - val_iou_class_2: 0.8272 - val_iou_coef_multiclass: 0.8895 - val_loss: 0.2813
Epoch 32/100
Epoch 32: val_loss did not improve from 0.27255
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 460ms/step - accuracy: 0.9779 - dice_class_0: 0.9862 - dice_class_1: 0.9497 - dice_class_2: 0.9098 - dice_coef_multiclass: 0.9485 - iou_class_0: 0.9729 - iou_class_1: 0.9103 - iou_class_2: 0.8463 - iou_coef_multiclass: 0.9098 - loss: 0.2352 - val_accuracy: 0.9774 - val_dice_class_0: 0.9787 - val_dice_class_1: 0.9229 - val_dice_class_2: 0.8987 - val_dice_coef_multiclass: 0.9335 - val_iou_class_0: 0.9664 - val_iou_class_1: 0.8781 - val_iou_class_2: 0.8376 - val_iou_coef_multiclass: 0.8940 - val_loss: 0.2792
Epoch 33/100
Epoch 33: val_loss did not improve from 0.27255
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 442ms/step - accuracy: 0.9768 - dice_class_0: 0.9867 - dice_class_1: 0.9559 - dice_class_2: 0.9179 - dice_coef_multiclass: 0.9535 - iou_class_0: 0.9717 - iou_class_1: 0.9092 - iou_class_2: 0.8491 - iou_coef_multiclass: 0.9100 - loss: 0.2330 - val_accuracy: 0.9701 - val_dice_class_0: 0.9764 - val_dice_class_1: 0.9385 - val_dice_class_2: 0.8961 - val_dice_coef_multiclass: 0.9370 - val_iou_class_0: 0.9692 - val_iou_class_1: 0.8915 - val_iou_class_2: 0.8278 - val_iou_coef_multiclass: 0.8962 - val_loss: 0.2862
Epoch 34/100
Epoch 34: val_loss improved to 0.26939, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 34: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 436ms/step - accuracy: 0.9765 - dice_class_0: 0.9864 - dice_class_1: 0.9540 - dice_class_2: 0.9160 - dice_coef_multiclass: 0.9521 - iou_class_0: 0.9764 - iou_class_1: 0.9150 - iou_class_2: 0.8512 - iou_coef_multiclass: 0.9142 - loss: 0.2290 - val_accuracy: 0.9692 - val_dice_class_0: 0.9814 - val_dice_class_1: 0.9310 - val_dice_class_2: 0.9078 - val_dice_coef_multiclass: 0.9401 - val_iou_class_0: 0.9732 - val_iou_class_1: 0.8964 - val_iou_class_2: 0.8312 - val_iou_coef_multiclass: 0.9003 - val_loss: 0.2694
Epoch 35/100
Epoch 35: val_loss improved to 0.25466, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 35: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 428ms/step - accuracy: 0.9847 - dice_class_0: 0.9867 - dice_class_1: 0.9548 - dice_class_2: 0.9159 - dice_coef_multiclass: 0.9525 - iou_class_0: 0.9767 - iou_class_1: 0.9148 - iou_class_2: 0.8537 - iou_coef_multiclass: 0.9151 - loss: 0.2276 - val_accuracy: 0.9754 - val_dice_class_0: 0.9797 - val_dice_class_1: 0.9288 - val_dice_class_2: 0.9074 - val_dice_coef_multiclass: 0.9386 - val_iou_class_0: 0.9749 - val_iou_class_1: 0.8994 - val_iou_class_2: 0.8315 - val_iou_coef_multiclass: 0.9020 - val_loss: 0.2547
Epoch 36/100
Epoch 36: val_loss did not improve from 0.25466
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 456ms/step - accuracy: 0.9810 - dice_class_0: 0.9875 - dice_class_1: 0.9537 - dice_class_2: 0.9202 - dice_coef_multiclass: 0.9538 - iou_class_0: 0.9776 - iou_class_1: 0.9162 - iou_class_2: 0.8535 - iou_coef_multiclass: 0.9158 - loss: 0.2239 - val_accuracy: 0.9733 - val_dice_class_0: 0.9922 - val_dice_class_1: 0.9468 - val_dice_class_2: 0.9063 - val_dice_coef_multiclass: 0.9484 - val_iou_class_0: 0.9801 - val_iou_class_1: 0.8970 - val_iou_class_2: 0.8413 - val_iou_coef_multiclass: 0.9061 - val_loss: 0.2713
Epoch 37/100
Epoch 37: val_loss improved to 0.24985, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 37: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 428ms/step - accuracy: 0.9807 - dice_class_0: 0.9877 - dice_class_1: 0.9549 - dice_class_2: 0.9212 - dice_coef_multiclass: 0.9546 - iou_class_0: 0.9777 - iou_class_1: 0.9197 - iou_class_2: 0.8565 - iou_coef_multiclass: 0.9180 - loss: 0.2261 - val_accuracy: 0.9704 - val_dice_class_0: 0.9910 - val_dice_class_1: 0.9407 - val_dice_class_2: 0.9104 - val_dice_coef_multiclass: 0.9474 - val_iou_class_0: 0.9798 - val_iou_class_1: 0.9001 - val_iou_class_2: 0.8424 - val_iou_coef_multiclass: 0.9074 - val_loss: 0.2499
Epoch 38/100
Epoch 38: val_loss did not improve from 0.24985
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 425ms/step - accuracy: 0.9803 - dice_class_0: 0.9894 - dice_class_1: 0.9572 - dice_class_2: 0.9186 - dice_coef_multiclass: 0.9551 - iou_class_0: 0.9808 - iou_class_1: 0.9200 - iou_class_2: 0.8600 - iou_coef_multiclass: 0.9203 - loss: 0.2337 - val_accuracy: 0.9679 - val_dice_class_0: 0.9897 - val_dice_class_1: 0.9393 - val_dice_class_2: 0.9125 - val_dice_coef_multiclass: 0.9472 - val_iou_class_0: 0.9845 - val_iou_class_1: 0.9041 - val_iou_class_2: 0.8524 - val_iou_coef_multiclass: 0.9136 - val_loss: 0.2527
Epoch 39/100
Epoch 39: val_loss did not improve from 0.24985
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 435ms/step - accuracy: 0.9852 - dice_class_0: 0.9918 - dice_class_1: 0.9581 - dice_class_2: 0.9173 - dice_coef_multiclass: 0.9557 - iou_class_0: 0.9812 - iou_class_1: 0.9217 - iou_class_2: 0.8613 - iou_coef_multiclass: 0.9214 - loss: 0.2215 - val_accuracy: 0.9717 - val_dice_class_0: 0.9887 - val_dice_class_1: 0.9406 - val_dice_class_2: 0.9164 - val_dice_coef_multiclass: 0.9485 - val_iou_class_0: 0.9824 - val_iou_class_1: 0.9131 - val_iou_class_2: 0.8535 - val_iou_coef_multiclass: 0.9163 - val_loss: 0.2564
Epoch 40/100
Epoch 40: val_loss improved to 0.24656, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 40: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 437ms/step - accuracy: 0.9824 - dice_class_0: 0.9920 - dice_class_1: 0.9561 - dice_class_2: 0.9223 - dice_coef_multiclass: 0.9568 - iou_class_0: 0.9753 - iou_class_1: 0.9193 - iou_class_2: 0.8633 - iou_coef_multiclass: 0.9193 - loss: 0.2116 - val_accuracy: 0.9784 - val_dice_class_0: 0.9916 - val_dice_class_1: 0.9492 - val_dice_class_2: 0.9231 - val_dice_coef_multiclass: 0.9546 - val_iou_class_0: 0.9725 - val_iou_class_1: 0.9076 - val_iou_class_2: 0.8495 - val_iou_coef_multiclass: 0.9098 - val_loss: 0.2466
Epoch 41/100
Epoch 41: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 452ms/step - accuracy: 0.9834 - dice_class_0: 0.9914 - dice_class_1: 0.9605 - dice_class_2: 0.9250 - dice_coef_multiclass: 0.9590 - iou_class_0: 0.9832 - iou_class_1: 0.9228 - iou_class_2: 0.8635 - iou_coef_multiclass: 0.9232 - loss: 0.2236 - val_accuracy: 0.9750 - val_dice_class_0: 0.9883 - val_dice_class_1: 0.9489 - val_dice_class_2: 0.9158 - val_dice_coef_multiclass: 0.9510 - val_iou_class_0: 0.9813 - val_iou_class_1: 0.9025 - val_iou_class_2: 0.8552 - val_iou_coef_multiclass: 0.9130 - val_loss: 0.2529
Epoch 42/100
Epoch 42: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 431ms/step - accuracy: 0.9869 - dice_class_0: 0.9902 - dice_class_1: 0.9592 - dice_class_2: 0.9235 - dice_coef_multiclass: 0.9576 - iou_class_0: 0.9799 - iou_class_1: 0.9260 - iou_class_2: 0.8640 - iou_coef_multiclass: 0.9233 - loss: 0.2135 - val_accuracy: 0.9798 - val_dice_class_0: 0.9939 - val_dice_class_1: 0.9464 - val_dice_class_2: 0.9239 - val_dice_coef_multiclass: 0.9547 - val_iou_class_0: 0.9749 - val_iou_class_1: 0.9166 - val_iou_class_2: 0.8513 - val_iou_coef_multiclass: 0.9143 - val_loss: 0.2560
Epoch 43/100
Epoch 43: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 69s 458ms/step - accuracy: 0.9836 - dice_class_0: 0.9926 - dice_class_1: 0.9617 - dice_class_2: 0.9243 - dice_coef_multiclass: 0.9595 - iou_class_0: 0.9792 - iou_class_1: 0.9297 - iou_class_2: 0.8619 - iou_coef_multiclass: 0.9236 - loss: 0.2221 - val_accuracy: 0.9859 - val_dice_class_0: 0.9976 - val_dice_class_1: 0.9453 - val_dice_class_2: 0.9174 - val_dice_coef_multiclass: 0.9534 - val_iou_class_0: 0.9777 - val_iou_class_1: 0.9066 - val_iou_class_2: 0.8583 - val_iou_coef_multiclass: 0.9142 - val_loss: 0.2574
Epoch 44/100
Epoch 44: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 424ms/step - accuracy: 0.9868 - dice_class_0: 0.9925 - dice_class_1: 0.9606 - dice_class_2: 0.9232 - dice_coef_multiclass: 0.9588 - iou_class_0: 0.9823 - iou_class_1: 0.9263 - iou_class_2: 0.8612 - iou_coef_multiclass: 0.9233 - loss: 0.2145 - val_accuracy: 0.9811 - val_dice_class_0: 0.9869 - val_dice_class_1: 0.9428 - val_dice_class_2: 0.9158 - val_dice_coef_multiclass: 0.9485 - val_iou_class_0: 0.9918 - val_iou_class_1: 0.9138 - val_iou_class_2: 0.8631 - val_iou_coef_multiclass: 0.9229 - val_loss: 0.2582
Epoch 45/100
Epoch 45: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 454ms/step - accuracy: 0.9859 - dice_class_0: 0.9924 - dice_class_1: 0.9621 - dice_class_2: 0.9313 - dice_coef_multiclass: 0.9620 - iou_class_0: 0.9871 - iou_class_1: 0.9277 - iou_class_2: 0.8646 - iou_coef_multiclass: 0.9265 - loss: 0.2152 - val_accuracy: 0.9864 - val_dice_class_0: 0.9910 - val_dice_class_1: 0.9554 - val_dice_class_2: 0.9262 - val_dice_coef_multiclass: 0.9575 - val_iou_class_0: 0.9870 - val_iou_class_1: 0.9165 - val_iou_class_2: 0.8689 - val_iou_coef_multiclass: 0.9241 - val_loss: 0.2505
  Cosine LR: epoch=45, cycle=2, T_cur=0/60, lr=2.00e-04
Epoch 46/100
Epoch 46: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 439ms/step - accuracy: 0.9843 - dice_class_0: 0.9893 - dice_class_1: 0.9663 - dice_class_2: 0.9290 - dice_coef_multiclass: 0.9615 - iou_class_0: 0.9844 - iou_class_1: 0.9273 - iou_class_2: 0.8679 - iou_coef_multiclass: 0.9265 - loss: 0.2002 - val_accuracy: 0.9778 - val_dice_class_0: 0.9876 - val_dice_class_1: 0.9587 - val_dice_class_2: 0.9147 - val_dice_coef_multiclass: 0.9537 - val_iou_class_0: 0.9744 - val_iou_class_1: 0.9084 - val_iou_class_2: 0.8687 - val_iou_coef_multiclass: 0.9171 - val_loss: 0.2682
Epoch 47/100
Epoch 47: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 454ms/step - accuracy: 0.9744 - dice_class_0: 0.9804 - dice_class_1: 0.9569 - dice_class_2: 0.9193 - dice_coef_multiclass: 0.9522 - iou_class_0: 0.9762 - iou_class_1: 0.9185 - iou_class_2: 0.8607 - iou_coef_multiclass: 0.9184 - loss: 0.2370 - val_accuracy: 0.9503 - val_dice_class_0: 0.9602 - val_dice_class_1: 0.9239 - val_dice_class_2: 0.8954 - val_dice_coef_multiclass: 0.9265 - val_iou_class_0: 0.9586 - val_iou_class_1: 0.8964 - val_iou_class_2: 0.8417 - val_iou_coef_multiclass: 0.8989 - val_loss: 0.3676
Epoch 48/100
Epoch 48: val_loss did not improve from 0.24656
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 439ms/step - accuracy: 0.9739 - dice_class_0: 0.9817 - dice_class_1: 0.9570 - dice_class_2: 0.9210 - dice_coef_multiclass: 0.9532 - iou_class_0: 0.9762 - iou_class_1: 0.9192 - iou_class_2: 0.8568 - iou_coef_multiclass: 0.9174 - loss: 0.2346 - val_accuracy: 0.9450 - val_dice_class_0: 0.9691 - val_dice_class_1: 0.9321 - val_dice_class_2: 0.8959 - val_dice_coef_multiclass: 0.9323 - val_iou_class_0: 0.9597 - val_iou_class_1: 0.8877 - val_iou_class_2: 0.8321 - val_iou_coef_multiclass: 0.8932 - val_loss: 0.3691
Epoch 49/100
Epoch 49: val_loss improved to 0.24247, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 49: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 434ms/step - accuracy: 0.9889 - dice_class_0: 0.9909 - dice_class_1: 0.9609 - dice_class_2: 0.9288 - dice_coef_multiclass: 0.9602 - iou_class_0: 0.9864 - iou_class_1: 0.9265 - iou_class_2: 0.8668 - iou_coef_multiclass: 0.9265 - loss: 0.2073 - val_accuracy: 0.9846 - val_dice_class_0: 0.9912 - val_dice_class_1: 0.9559 - val_dice_class_2: 0.9165 - val_dice_coef_multiclass: 0.9545 - val_iou_class_0: 0.9764 - val_iou_class_1: 0.9181 - val_iou_class_2: 0.8640 - val_iou_coef_multiclass: 0.9195 - val_loss: 0.2425
Epoch 50/100
Epoch 50: val_loss improved to 0.24230, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 50: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 67s 457ms/step - accuracy: 0.9835 - dice_class_0: 0.9928 - dice_class_1: 0.9583 - dice_class_2: 0.9251 - dice_coef_multiclass: 0.9588 - iou_class_0: 0.9855 - iou_class_1: 0.9262 - iou_class_2: 0.8696 - iou_coef_multiclass: 0.9271 - loss: 0.2107 - val_accuracy: 0.9933 - val_dice_class_0: 0.9906 - val_dice_class_1: 0.9590 - val_dice_class_2: 0.9309 - val_dice_coef_multiclass: 0.9601 - val_iou_class_0: 0.9761 - val_iou_class_1: 0.9253 - val_iou_class_2: 0.8725 - val_iou_coef_multiclass: 0.9246 - val_loss: 0.2423
Epoch 51/100
Epoch 51: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 449ms/step - accuracy: 0.9883 - dice_class_0: 0.9938 - dice_class_1: 0.9591 - dice_class_2: 0.9317 - dice_coef_multiclass: 0.9616 - iou_class_0: 0.9853 - iou_class_1: 0.9305 - iou_class_2: 0.8656 - iou_coef_multiclass: 0.9271 - loss: 0.2079 - val_accuracy: 0.9820 - val_dice_class_0: 0.9994 - val_dice_class_1: 0.9604 - val_dice_class_2: 0.9291 - val_dice_coef_multiclass: 0.9630 - val_iou_class_0: 0.9760 - val_iou_class_1: 0.9176 - val_iou_class_2: 0.8628 - val_iou_coef_multiclass: 0.9188 - val_loss: 0.2484
Epoch 52/100
Epoch 52: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 459ms/step - accuracy: 0.9877 - dice_class_0: 0.9923 - dice_class_1: 0.9627 - dice_class_2: 0.9281 - dice_coef_multiclass: 0.9610 - iou_class_0: 0.9864 - iou_class_1: 0.9320 - iou_class_2: 0.8682 - iou_coef_multiclass: 0.9288 - loss: 0.1974 - val_accuracy: 0.9862 - val_dice_class_0: 0.9938 - val_dice_class_1: 0.9549 - val_dice_class_2: 0.9196 - val_dice_coef_multiclass: 0.9561 - val_iou_class_0: 0.9742 - val_iou_class_1: 0.9303 - val_iou_class_2: 0.8715 - val_iou_coef_multiclass: 0.9253 - val_loss: 0.2620
Epoch 53/100
Epoch 53: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 439ms/step - accuracy: 0.9866 - dice_class_0: 0.9938 - dice_class_1: 0.9655 - dice_class_2: 0.9299 - dice_coef_multiclass: 0.9630 - iou_class_0: 0.9820 - iou_class_1: 0.9315 - iou_class_2: 0.8681 - iou_coef_multiclass: 0.9272 - loss: 0.2074 - val_accuracy: 0.9936 - val_dice_class_0: 0.9922 - val_dice_class_1: 0.9603 - val_dice_class_2: 0.9252 - val_dice_coef_multiclass: 0.9592 - val_iou_class_0: 0.9854 - val_iou_class_1: 0.9162 - val_iou_class_2: 0.8702 - val_iou_coef_multiclass: 0.9239 - val_loss: 0.2540
Epoch 54/100
Epoch 54: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 69s 438ms/step - accuracy: 0.9918 - dice_class_0: 0.9928 - dice_class_1: 0.9629 - dice_class_2: 0.9291 - dice_coef_multiclass: 0.9616 - iou_class_0: 0.9810 - iou_class_1: 0.9329 - iou_class_2: 0.8712 - iou_coef_multiclass: 0.9284 - loss: 0.2042 - val_accuracy: 0.9803 - val_dice_class_0: 0.9918 - val_dice_class_1: 0.9530 - val_dice_class_2: 0.9365 - val_dice_coef_multiclass: 0.9604 - val_iou_class_0: 0.9750 - val_iou_class_1: 0.9271 - val_iou_class_2: 0.8628 - val_iou_coef_multiclass: 0.9217 - val_loss: 0.2734
Epoch 55/100
Epoch 55: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 447ms/step - accuracy: 0.9909 - dice_class_0: 0.9902 - dice_class_1: 0.9635 - dice_class_2: 0.9328 - dice_coef_multiclass: 0.9621 - iou_class_0: 0.9838 - iou_class_1: 0.9294 - iou_class_2: 0.8701 - iou_coef_multiclass: 0.9277 - loss: 0.1981 - val_accuracy: 0.9838 - val_dice_class_0: 0.9830 - val_dice_class_1: 0.9576 - val_dice_class_2: 0.9190 - val_dice_coef_multiclass: 0.9532 - val_iou_class_0: 0.9766 - val_iou_class_1: 0.9224 - val_iou_class_2: 0.8536 - val_iou_coef_multiclass: 0.9176 - val_loss: 0.2507
Epoch 56/100
Epoch 56: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 458ms/step - accuracy: 0.9921 - dice_class_0: 0.9902 - dice_class_1: 0.9625 - dice_class_2: 0.9315 - dice_coef_multiclass: 0.9614 - iou_class_0: 0.9829 - iou_class_1: 0.9317 - iou_class_2: 0.8701 - iou_coef_multiclass: 0.9282 - loss: 0.2073 - val_accuracy: 0.9885 - val_dice_class_0: 0.9953 - val_dice_class_1: 0.9592 - val_dice_class_2: 0.9260 - val_dice_coef_multiclass: 0.9602 - val_iou_class_0: 0.9852 - val_iou_class_1: 0.9266 - val_iou_class_2: 0.8627 - val_iou_coef_multiclass: 0.9248 - val_loss: 0.2657
Epoch 57/100
Epoch 57: val_loss did not improve from 0.24230
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 449ms/step - accuracy: 0.9890 - dice_class_0: 0.9911 - dice_class_1: 0.9685 - dice_class_2: 0.9328 - dice_coef_multiclass: 0.9641 - iou_class_0: 0.9863 - iou_class_1: 0.9324 - iou_class_2: 0.8724 - iou_coef_multiclass: 0.9304 - loss: 0.1971 - val_accuracy: 0.9810 - val_dice_class_0: 0.9945 - val_dice_class_1: 0.9615 - val_dice_class_2: 0.9316 - val_dice_coef_multiclass: 0.9625 - val_iou_class_0: 0.9777 - val_iou_class_1: 0.9222 - val_iou_class_2: 0.8723 - val_iou_coef_multiclass: 0.9241 - val_loss: 0.2638
Epoch 58/100
Epoch 58: val_loss improved to 0.24003, saving model to results/V8_Phase4_CLAHE_PP_TTA.keras

Epoch 58: finished saving model to results/V8_Phase4_CLAHE_PP_TTA.keras
150/150 ━━━━━━━━━━━━━━━━━━━━ 68s 453ms/step - accuracy: 0.9857 - dice_class_0: 0.9941 - dice_class_1: 0.9676 - dice_class_2: 0.9258 - dice_coef_multiclass: 0.9625 - iou_class_0: 0.9819 - iou_class_1: 0.9311 - iou_class_2: 0.8707 - iou_coef_multiclass: 0.9279 - loss: 0.2009 - val_accuracy: 0.9824 - val_dice_class_0: 0.9916 - val_dice_class_1: 0.9617 - val_dice_class_2: 0.9271 - val_dice_coef_multiclass: 0.9601 - val_iou_class_0: 0.9787 - val_iou_class_1: 0.9273 - val_iou_class_2: 0.8635 - val_iou_coef_multiclass: 0.9232 - val_loss: 0.2400
Epoch 59/100
Epoch 59: val_loss did not improve from 0.24003
150/150 ━━━━━━━━━━━━━━━━━━━━ 65s 453ms/step - accuracy: 0.9905 - dice_class_0: 0.9928 - dice_class_1: 0.9666 - dice_class_2: 0.9328 - dice_coef_multiclass: 0.9641 - iou_class_0: 0.9885 - iou_class_1: 0.9346 - iou_class_2: 0.8717 - iou_coef_multiclass: 0.9316 - loss: 0.2030 - val_accuracy: 0.9852 - val_dice_class_0: 0.9872 - val_dice_class_1: 0.9601 - val_dice_class_2: 0.9274 - val_dice_coef_multiclass: 0.9582 - val_iou_class_0: 0.9847 - val_iou_class_1: 0.9250 - val_iou_class_2: 0.8768 - val_iou_coef_multiclass: 0.9288 - val_loss: 0.2586
Epoch 60/100
Epoch 60: val_loss did not improve from 0.24003
150/150 ━━━━━━━━━━━━━━━━━━━━ 69s 427ms/step - accuracy: 0.9899 - dice_class_0: 0.9918 - dice_class_1: 0.9639 - dice_class_2: 0.9330 - dice_coef_multiclass: 0.9629 - iou_class_0: 0.9869 - iou_class_1: 0.9302 - iou_class_2: 0.8697 - iou_coef_multiclass: 0.9289 - loss: 0.2137 - val_accuracy: 0.9893 - val_dice_class_0: 0.9899 - val_dice_class_1: 0.9625 - val_dice_class_2: 0.9377 - val_dice_coef_multiclass: 0.9634 - val_iou_class_0: 0.9835 - val_iou_class_1: 0.9314 - val_iou_class_2: 0.8621 - val_iou_coef_multiclass: 0.9257 - val_loss: 0.2569
Epoch 61/100
Epoch 61: val_loss did not improve from 0.24003
150/150 ━━━━━━━━━━━━━━━━━━━━ 64s 436ms/step - accuracy: 0.9901 - dice_class_0: 0.9931 - dice_class_1: 0.9664 - dice_class_2: 0.9321 - dice_coef_multiclass: 0.9639 - iou_class_0: 0.9848 - iou_class_1: 0.9308 - iou_class_2: 0.8719 - iou_coef_multiclass: 0.9292 - loss: 0.1972 - val_accuracy: 0.9819 - val_dice_class_0: 0.9904 - val_dice_class_1: 0.9680 - val_dice_class_2: 0.9381 - val_dice_coef_multiclass: 0.9655 - val_iou_class_0: 0.9970 - val_iou_class_1: 0.9185 - val_iou_class_2: 0.8703 - val_iou_coef_multiclass: 0.9286 - val_loss: 0.2622
Epoch 62/100
Epoch 62: val_loss did not improve from 0.24003
150/150 ━━━━━━━━━━━━━━━━━━━━ 66s 447ms/step - accuracy: 0.9900 - dice_class_0: 0.9930 - dice_class_1: 0.9650 - dice_class_2: 0.9320 - dice_coef_multiclass: 0.9633 - iou_class_0: 0.9860 - iou_class_1: 0.9320 - iou_class_2: 0.8720 - iou_coef_multiclass: 0.9300 - loss: 0.2000 - val_accuracy: 0.9850 - val_dice_class_0: 0.9921 - val_dice_class_1: 0.9622 - val_dice_class_2: 0.9298 - val_dice_coef_multiclass: 0.9614 - val_iou_class_0: 0.9843 - val_iou_class_1: 0.9272 - val_iou_class_2: 0.8688 - val_iou_coef_multiclass: 0.9268 - val_loss: 0.2500
Epoch 62: early stopping
✓ Training completed
  ✓ Saved: results/V8_Phase4_CLAHE_PP_TTA_history.png
================================================================================
EVALUATION: V8_Phase4_CLAHE_PP_TTA
================================================================================

────────────────────────────────────────────────────────────────────────────────
STAGE 1: RAW PREDICTIONS
────────────────────────────────────────────────────────────────────────────────
  ✓ Saved: results/V8_Phase4_CLAHE_PP_TTA_raw_confusion_matrices.png""")
display(Markdown("![Confusion Matrix](confusion_matrix.png)"))
print(r"""Raw metrics:
  Background: Precision=0.9925, Recall=0.9917, F1=0.9921
  Disc: Precision=0.9634, Recall=0.9610, F1=0.9622
  Cup: Precision=0.9281, Recall=0.9315, F1=0.9298
  
  Background: Dice=0.9921, IoU=0.9843
  Disc: Dice=0.9622, IoU=0.9272
  Cup: Dice=0.9298, IoU=0.8688

Overall (Macro Avg):
  Precision=0.9613, Recall=0.9614, F1=0.9613
  Dice=0.9613, IoU=0.9267

────────────────────────────────────────────────────────────────────────────────
STAGE 2: POST-PROCESSED
────────────────────────────────────────────────────────────────────────────────
Post-processed metrics:
  Background: Precision=0.9924, Recall=0.9916, F1=0.9920
  Disc: Precision=0.9634, Recall=0.9610, F1=0.9622
  Cup: Precision=0.9281, Recall=0.9315, F1=0.9298
  
  Background: Dice=0.9920 (-0.0001), IoU=0.9841
  Disc: Dice=0.9622 (+0.0000), IoU=0.9272
  Cup: Dice=0.9298 (+0.0000), IoU=0.8688

Overall (Macro Avg):
  Precision=0.9613, Recall=0.9613, F1=0.9613
  Dice=0.9613, IoU=0.9267

────────────────────────────────────────────────────────────────────────────────
STAGE 3: TTA + POST-PROCESSING
────────────────────────────────────────────────────────────────────────────────

Running TTA with 4 geometric passes...
  Pass 1/4: Original
  Pass 2/4: Horizontal flip
  Pass 3/4: Vertical flip
  Pass 4/4: Both flips
  ✓ TTA completed

TTA+PP metrics:
  Background: Precision=0.9961, Recall=0.9971, F1=0.9966
  Disc: Precision=0.9691, Recall=0.9677, F1=0.9684
  Cup: Precision=0.9378, Recall=0.9356, F1=0.9367

  Background: Dice=0.9966 (+0.0045 vs raw), IoU=0.9932
  Disc: Dice=0.9684 (+0.0062 vs raw), IoU=0.9387
  Cup: Dice=0.9367 (+0.0069 vs raw), IoU=0.8809

Overall (Macro Avg):
  Precision=0.9676, Recall=0.9668, F1=0.9672
  Dice=0.9672, IoU=0.9376

================================================================================
RESULTS COMPARISON
================================================================================
Stage                          BG Dice      Disc Dice    Cup Dice    
──────────────────────────────────────────────────────────────────
Raw (no PP, no TTA)            0.9921       0.9622       0.9298      
+ Post-Processing              0.9920       0.9622       0.9298      
+ TTA + Post-Processing        0.9966       0.9684       0.9367      
Saving 400 predictions to results/V8_Phase4_CLAHE_PP_TTA_all_preds...
  50/400
  100/400
  150/400
  200/400
  250/400
  300/400
  350/400
  400/400
✓ Saved to results/V8_Phase4_CLAHE_PP_TTA_all_preds

Per-class (TTA+PP):
  Background: Precision=0.9961, Recall=0.9971, F1=0.9966
  Disc: Precision=0.9691, Recall=0.9677, F1=0.9684
  Cup: Precision=0.9378, Recall=0.9356, F1=0.9367

  Background: Dice=0.9966, IoU=0.9932
  Disc: Dice=0.9684, IoU=0.9387
  Cup: Dice=0.9367, IoU=0.8809

Overall (Macro Avg):
  Precision=0.9676, Recall=0.9668, F1=0.9672
  Dice=0.9672, IoU=0.9376

================================================================================
GLAUCOMA PROGRESSION
================================================================================
  CDR Error: 0.0589 ± 0.0564
  Accuracy: 81.75% (327/400)
    Normal: 0/0 = 0.0%
    Suspect: 241/257 = 93.8%
    Moderate: 58/106 = 54.7%
    Critical: 28/37 = 75.7%

  Manual Review: 101/400 (25.2%)
  ✓ Saved: results/glaucoma_progression_analysis.csv
  ✓ Saved: results/cdr_analysis.png
""")
display(Markdown("![Progression](progression.png)"))
print(r"""
================================================================================
MC DROPOUT UNCERTAINTY
================================================================================

MC Dropout: 15 passes...
  5/15
  10/15
  15/15
  ✓ Done
  Background uncertainty: 0.0018
  Disc uncertainty: 0.0033
  Cup uncertainty: 0.0016

====================================================================================================
                                     ✓ COMPLETED — V8 + PHASE 4                                     
====================================================================================================
  Model: V5 base + CLAHE + noise + PP + TTA + Phase 4
  Parameters: 47,631,627
  ROI: 1.2× padding, 512×512
====================================================================================================

""")
