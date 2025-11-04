from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt

# ✅ 正确路径
# base_path = Path("/home/gincy/音乐/Mamba-UNet/model/ACDC/VIM_140_labeled_3/mambaunet_predictions")
# base_path = Path("/home/gincy/音乐/Mamba-UNet/model/ACDC/DynamicMask_Mamba_140/mambaunet_dynamicmask_predictions")
base_path = Path("./model/ACDC/VIM_140_labeled_3/mambaunet_predictions")

# 文件名（例如 patient001_frame01_img.nii.gz）
index =7


img_name = "patient001_frame01_img.nii.gz"
gt_name = "patient001_frame01_gt.nii.gz"
pred_name = "patient001_frame01_pred.nii.gz"

# img_name = f"patient00{index}_frame0{index}_img.nii.gz"
# gt_name = f"patient00{index}_frame0{index}_gt.nii.gz"
# pred_name = f"patient00{index}_frame0{index}_pred.nii.gz"


# 拼接路径
img_path = base_path / img_name
gt_path = base_path / gt_name
pred_path = base_path / pred_name

print("Image path:", img_path)
print("GT path:", gt_path)
print("Pred path:", pred_path)

# ✅ 加载 nii 文件
img = nib.load(img_path).get_fdata()
gt = nib.load(gt_path).get_fdata()
pred = nib.load(pred_path).get_fdata()

# 可视化
slice_idx = img.shape[2] // 2
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img[:, :, slice_idx], cmap='gray')
plt.title('Image')

plt.subplot(1, 3, 2)
plt.imshow(img[:, :, slice_idx], cmap='gray')
plt.imshow(gt[:, :, slice_idx], alpha=0.4, cmap='Greens')
plt.title('Ground Truth')
         
plt.subplot(1, 3, 3)
plt.imshow(img[:, :, slice_idx], cmap='gray')
plt.imshow(pred[:, :, slice_idx], alpha=0.4, cmap='Reds')
plt.title('Prediction')

plt.tight_layout()
# plt.savefig("visual_result_mambaunet_dynamicmask.png")
# print("✅ 图像已保存：visual_result_mambaunet_dynamicmask.png")
plt.savefig("visual_result_mambaunet.png")
print("✅ 图像已保存：visual_result_mambaunet.png")

#/home/gincy/音乐/MN/model/ACDC/VIM_140_labeled_3/mambaunet_predictions/patient001_frame01_gt.nii.gz