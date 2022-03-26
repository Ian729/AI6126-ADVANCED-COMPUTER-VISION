## FAQ on Project 1
FAQ:  
Q1: I notice that there are train/val/test splits of the provided dataset. Which ones can be used for training and which ones cannot?  
A1: You can use the train and val splits to train your model, but training on the test split, e.g., pseudo labeling self-training, is strictly prohibited.  
Q2: I notice that the original CelebA-Mask dataset contains 30K images. Are we allowed to train on it?  
A2: No, and please also avoid downloading dataset from the official CelebA-Mask repo. You should only use the link in the handout to download data.  
Q3: I'm new to MMSegmentation and I'm confused about the config file naming and the meaning of each config field. Where can I find related tutorials?  
A3: https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html  
Q4: Related to Q3. I'm worried about mistakenly loading weights pre-trained on other datasets. How can I make sure I initialize my model with only ImageNet pre-trained weights?  
A4: See this example config: https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html#an-example-of-pspnet. Look at line 4, `pretrained='open-mmlab://resnet50_v1c'`, and this field indicates which weights you are using for initialization. Other valid options can be found at https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json (You should not download weights from the MMSegmentation model zoo because they are trained on segmentation datasets, such as Cityscapes). Besides, make sure the `load_from` and `resume_from` are None.  
Q5: The config file of MMSegmentation has hierarchical inheritance specified with `_base_`. How can I unroll the inheritance and inspect the full config?  
A5: Try `python tools/print_config.py [config_path]`.  
Q6: I saved the predictions in png images. But they look almost pure black images when opened with a picture browser. Is this as expected?  
A6: Yes, the saved images should be 512x512 and have integer values in [0, 18].  
Q7: I fail to run the provided `baseline.py` after installing MMSegmentation. What could be the problem?  
A1: MMSegmentation doesn't support the CelebA-Mask dataset out of the box. You need to write a customized data loader for it. Please refer to https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/voc.py for an example.  
Q8: I notice that, in the CelebA-Mask dataset, some necklace pixels are mistakenly annotated as skin or background. Is this normal?  
A2: Yes, after double checking with the original CelebA-Mask dataset, we confirm that some samples are mislabeled. You don't need to relabel the data. The provided baseline doesn’t particularly deal with the necklace problem and gets around 8.7 IoU for the necklace class.  
Q9: Are we allowed to use models pre-trained on the ImageNet-21K?  
A3: No, ImageNet-21K is a much larger dataset than the commonly used ImageNet (a.k.a. ImageNet-1K).  
Q10: I plan to do this project on Google Colab. How should I get started?  
A4: Related to Q1, since you at least need to implement the data loader of CelebA-Mask, you can no longer `git clone` from the original MMSegmentation repo. Instead, you should fork the MMSegmentation, modify the code, push the modifications to your fork, and `git clone` from the forked repo.  
Q11: How can I generate the predictions for submission?  
A5: Please refer to https://github.com/open-mmlab/mmsegmentation/blob/master/demo/image_demo.py#L29. Instead of feeding the `result` into `show_result_pyplot`, you should save it as a png file.  