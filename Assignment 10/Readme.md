1. CIFAR10 Model using Resnet18 network architetcure
2. Data augmentation using albumentations library - applied cutout data augmentation
3. LRFinder to get the lr range estimate for model training
4. ReduceLrOnPlateu is applied to reduce LR when model get stuck in Plateu region
5. GradCAM for 25 misclassified images
