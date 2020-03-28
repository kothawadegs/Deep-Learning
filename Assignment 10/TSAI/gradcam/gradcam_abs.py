import torch
import matplotlib.pyplot as plt
import numpy as np

import gradcam.gradcam_utils as grad_utils
import gradcam.gradcam as gradcam


class GradCamAbs():
      def __init__(self, device, config, means, stds):
          self.device = device
          self.config = config
          self.means = means
          self.stds = stds
          self.model = config["arch"]

          self.gcam = gradcam.GradCAMpp.from_config(**config)

      def apply(self, normalized_img, target_class_id):
          ''' 
          1. Use unsqueeze to convert image to a batch of size 1
          2. pass normalized test input for Grad CAM processing..
          3. get a GradCAM saliency map on the class index..

          return: orig img, heatmap image and supper imposed images 
          '''
          mask, _ = self.gcam(normalized_img.unsqueeze(0), class_idx=target_class_id)

          # Un-Mormalize the inp image
          unnormalizedImg = self.UnNormalize(normalized_img.cpu())

          # apply mask on the original image and get the superimposed images(cam_result) and heatmap
          heatmap, cam_result = grad_utils.visualize_cam(mask, unnormalizedImg)

          return unnormalizedImg, heatmap, cam_result

      def applyOnImages(self, dataloader, num_of_images=5):
          gradcam_images = []
          pred_results = []

          data, target = iter(dataloader).next()
          data, target = data.to(self.device), target.to(self.device)
          for index, label in enumerate(target[:num_of_images]):

            # do class prediction and save results
            output = self.model(data[index].unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
            pred_class_id = pred[0].item()
            target_class_id = label.item()
            pred_results.append(dict(pred=pred_class_id, target=target_class_id))  

            # do GradCAM processing and save results
            origImg, heatmapImg, cam_result = self.apply(data[index],target_class_id)
            gradcam_images.append([origImg, heatmapImg, cam_result])

          return gradcam_images, pred_results
          
      def applyOnMisclassifiedImages(self, dataloader, num_of_images=5):
          gradcam_images = []
          pred_results = []
          misclassified_cnt = 0

          for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
            pred_marker = pred.eq(target.view_as(pred))   
            wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
            for idx in wrong_idx:
              index = idx[0].item()

              pred_class_id = pred[index][0].item()
              target_class_id = target[index].item()
              pred_results.append(dict(pred=pred_class_id, target=target_class_id))

              # do GradCAM processing and save results
              origImg, heatmapImg, cam_result = self.apply(data[index],target_class_id)
              gradcam_images.append([origImg, heatmapImg, cam_result])

              misclassified_cnt += 1
              if(misclassified_cnt==num_of_images):
                break

            if(misclassified_cnt==num_of_images):
              break
          return gradcam_images, pred_results

      def UnNormalize(self, tensor):
          """
          Args:
              tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
          Returns:
              Tensor: Normalized image.
          """
          for t, m, s in zip(tensor, self.means, self.stds):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
          
          return tensor
      
      def plot_results(self, images_set, predictions, classesname, save_filename="gradcam_details"):
          cnt=0
          nrow = len(images_set)
          fig = plt.figure(figsize=(5,48)) # (10,12)
          for index, imgs in enumerate(images_set):
              # original image
              title = "T:{}, P:{}".format(classesname[predictions[index]["target"]], classesname[predictions[index]["pred"]])
              ax = fig.add_subplot(nrow, 3, cnt+1, xticks=[], yticks=[])
              ax.set_title(title)
              #ax.axis('off')
              self.imshow(imgs[0])
              cnt += 1

              # heatmap
              ax = fig.add_subplot(nrow, 3, cnt+1, xticks=[], yticks=[])
              #ax.axis('off')
              self.imshow(imgs[1])
              cnt += 1

              # superimposed image
              ax = fig.add_subplot(nrow, 3, cnt+1, xticks=[], yticks=[])
              #ax.axis('off')
              self.imshow(imgs[2])
              cnt += 1

          fig.savefig("./images/{}.png".format(save_filename))
          return

      def plot_cam_images(self, images_set, predictions, classesname, save_filename="gradcam_result"):
          cnt=0
          fig = plt.figure(figsize=(10,10)) #(12,12)
          nrow = len(images_set)/5
          for index, imgs in enumerate(images_set):
              title = "T:{}, P:{}".format(classesname[predictions[index]["target"]], classesname[predictions[index]["pred"]])
              #print(title)
              ax = fig.add_subplot(nrow, 5, cnt+1, xticks=[], yticks=[])
              #ax.axis('off')
              ax.set_title(title)
              self.imshow(imgs[2])

              cnt += 1
              if(cnt>=5*nrow):
                 break

          fig.savefig("./images/{}.png".format(save_filename))
          return

      # functions to show an image
      def imshow(self, img):
          #img = img / 2 + 0.5     # # first convert back to [0,1] range from [-1,1] range
          npimg = img.numpy()
          # convert from CHW to HWC
          # from 3x32x32 to 32x32x3
          plt.imshow(np.transpose(npimg, (1, 2, 0)))