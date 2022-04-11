matriculation number: G2103278G
CodaLab username: Ian729

Submitted files:
1. Report:report.pdf
2. checkpoint files submitted to OneDrive, addresses included in report.pdf
3. plot_curve.ipynb is for plotting the curves in the report
4. baseline.py is the config file for mIoU 69-73 models(DeepLabV3+)
5. upernet_swin_tiny_patch4_window7_512x512_160k_pretrain_224x224_1K.py is the config file for mIoU 75 model(SWIN-T)
6. upernet_swin_base_512x512_160k.py is the config file for mIoU 77 model(SWIN-B)
7. use python run_baseline.py to run baseline model
8. use python run_swin.py to run SWIN-T model
9. use python run_swin_base.py to run SWIN-B model
10.__base__ is the folder similar to the mmsegmentation repo base models, it is used by SWIN models to configure
11.test_mask_7757.zip is the test_mask that got 77.57 on coda

Prerequisites for SWIN models:
1. folder structure different from school GPU environment, might need to change a few paths

Prerequisites for SWIN-T:
1. Download swin_tiny_patch4_window7_224.pth from official website
2. python tools/model_converters/swin2mmseg.py

Prerequisites for SWIN-B:
1. Download swin_base_patch4_window7_224.pth
2. python tools/model_converters/swin2mmseg.py

References to the third-party libraries:
None
