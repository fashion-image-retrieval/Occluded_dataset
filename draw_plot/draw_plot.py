import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

x = [5, 10, 20]

model= 'hyp_deit' # 'unicom' or 'hyp_vit' or 'hyp_dino' or 'hyp_deit'
is_zero_shot = False # True if zero-shot, False if fine-tuned
is_black = False # True if black occluded, False if color occluded
orig_finetuned = False # True if finetuned on original inshop dataset, False if finetuned on 5 center color occluded inshop dataset

################################# ORIGINAL FINETUNED ################################
################################# UNICOM ############################################
##### UNICOM Zero-shot Black Object Occluded #####
if model == 'unicom' and is_zero_shot == True and is_black == True and orig_finetuned == True:
    orig_zs_unicom = [75.86, 75.86, 75.86]
    zs_top_y = [60.87, 50.4, 43.56]
    zs_center_y = [55.2, 42.19, 26.13]
    zs_bottom_y = [69.33, 62.81, 54.54]
    zs_random_y = [66.32, 61.73, 55.01]

    ax.plot(x, orig_zs_unicom, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('UNICOM Zero-Shot Black Object per ratio and location')

##### UNICOM Fine-tuned Black Object Occluded #####
elif model == 'unicom' and is_zero_shot == False and is_black == True and orig_finetuned == True:
    orig_ft_unicom = [95.56, 95.56, 95.56]
    ft_top_y = [92.29, 86.66, 76.42]
    ft_center_y = [91.56, 85.31, 71.7]
    ft_bottom_y = [94.34, 92.64, 89.04]
    ft_random_y = [93.96, 92.29, 87.27]

    ax.plot(x, orig_ft_unicom, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('UNICOM Fine-tuned Black Object per ratio and location')

##### UNICOM Zero-shot Color Object Occluded #####
elif model == 'unicom' and is_zero_shot == True and is_black == False and orig_finetuned == True:
    orig_zs_unicom = [75.86, 75.86, 75.86]
    zs_top_y = [55.65, 28.87, 8.51]
    zs_center_y = [44.66, 21.65, 6.59]
    zs_bottom_y = [61.42, 37.22, 12.06]
    zs_random_y = [58.4, 44.26, 16.2]

    ax.plot(x, orig_zs_unicom, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('UNICOM Zero-Shot Color Object per ratio and location')

##### UNICOM Fine-tuned Color Object Occluded #####
elif model == 'unicom' and is_zero_shot == False and is_black == False and orig_finetuned == True:
    orig_ft_unicom = [95.56, 95.56, 95.56]
    ft_top_y = [92.43, 83.55, 64.01]
    ft_center_y = [91.43, 83.58, 59.11]
    ft_bottom_y = [94.20, 91.74, 80.63]
    ft_random_y = [93.58, 91.45, 79.93]

    ax.plot(x, orig_ft_unicom, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('UNICOM Fine-tuned Color Object Occluded per ratio and location')

################################# Hyp_ViT ############################################
##### Hyp_ViT Zero-shot Black Object Occluded #####
if model == 'hyp_vit' and is_zero_shot == True and is_black == True and orig_finetuned == True:
    orig_zs_hyp_vit = [43.19, 43.19, 43.19]
    zs_top_y = [17.98, 9.09, 4.37]
    zs_center_y = [14.76, 6.57, 2.73]
    zs_bottom_y = [19.52, 9.28, 4.82]
    zs_random_y = [18.49, 9.2, 4.46]

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # fig.subplots_adjust(hspace = 0.1)

    ax.plot(x, orig_zs_hyp_vit, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    # ax1.set_ylim(40, 44)
    # ax2.set_ylim(0, 25)

    plt.title('Hyp_ViT Zero-Shot Black Object per ratio and location')

##### Hyp_ViT Fine-tuned Black Object Occluded #####
elif model == 'hyp_vit' and is_zero_shot == False and is_black == True and orig_finetuned == True:
    orig_ft_hyp_vit = [92.63, 92.63, 92.63]
    ft_top_y = [86.64, 78.85, 65.61]
    ft_center_y = [82.61, 72.89, 55.65]
    ft_bottom_y = [89.3, 86.43, 79.53]
    ft_random_y = [88.28, 83.29, 71.97]

    ax.plot(x, orig_ft_hyp_vit, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('Hyp_ViT Fine-tuned Black Object per ratio and location')

##### Hyp_ViT Zero-shot Color Object Occluded #####
elif model == 'hyp_vit' and is_zero_shot == True and is_black == False and orig_finetuned == True:
    orig_zs_hyp_vit = [43.19, 43.19, 43.19]
    zs_top_y = [6.87, 2.07, 1.32]
    zs_center_y = [4.43, 1.63, 1.11]
    zs_bottom_y = [8.22, 2.22, 1.31]
    zs_random_y = [7.2, 3.02, 0.96]

    ax.plot(x, orig_zs_hyp_vit, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('Hyp_ViT Zero-Shot Color Object per ratio and location')

##### Hyp_ViT Fine-tuned Color Object Occluded #####
elif model == 'hyp_vit' and is_zero_shot == False and is_black == False and orig_finetuned == True:
    orig_ft_hyp_vit = [92.63, 92.63, 92.63]
    ft_top_y = [80.82, 58.55, 22.96]
    ft_center_y = [70.39, 41.97, 13.82]
    ft_bottom_y = [86.24, 72.86, 34.55]
    ft_random_y = [82.16, 71.61, 32.16]

    ax.plot(x, orig_ft_hyp_vit, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('Hyp_ViT Fine-tuned Color Object per ratio and location')

################################# Hyp_Dino ############################################
##### Hyp_Dino Zero-shot Black Object Occluded #####
if model == 'hyp_dino' and is_zero_shot == True and is_black == True and orig_finetuned == True:
    orig_zs_hyp_dino = [46.09, 46.09, 46.09]
    zs_top_y = [30.23, 19.49, 11.13]
    zs_center_y = [19.96, 10.96, 5.14]
    zs_bottom_y = [26.09, 13.86, 8.26]
    zs_random_y = [27.12, 16.73, 10.5]

    ax.plot(x, orig_zs_hyp_dino, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('Hyp_Dino Zero-Shot Black Object per ratio and location')

##### Hyp_Dino Fine-tuned Black Object Occluded #####
elif model == 'hyp_dino' and is_zero_shot == False and is_black == True and orig_finetuned == True:
    orig_ft_hyp_dino = [91.79, 91.79, 91.79]
    ft_top_y = [85.14, 77.09, 64.43]
    ft_center_y = [80.12, 69.24, 54.13]
    ft_bottom_y = [88.79, 85.36, 76.92]
    ft_random_y = [85.91, 80.69, 69.05]

    ax.plot(x, orig_ft_hyp_dino, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('Hyp_Dino Fine-tuned Black Object per ratio and location')

##### Hyp_Dino Zero-shot Color Object Occluded #####
elif model == 'hyp_dino' and is_zero_shot == True and is_black == False and orig_finetuned == True:
    orig_zs_hyp_dino = [46.09, 46.09, 46.09]
    zs_top_y = [15, 3.35, 1.51]
    zs_center_y = [3.91, 1.53, 1.29]
    zs_bottom_y = [12.86, 3.47, 1.41]
    zs_random_y = [9.95, 3.64, 1.14]

    ax.plot(x, orig_zs_hyp_dino, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('Hyp_Dino Zero-Shot Color Object per ratio and location')

##### Hyp_Dino Fine-tuned Color Object Occluded #####
elif model == 'hyp_dino' and is_zero_shot == False and is_black == False and orig_finetuned == True:
    orig_ft_hyp_dino = [91.79, 91.79, 91.79]
    ft_top_y = [77.49, 50.06, 14.56]
    ft_center_y = [58.13, 26.6, 7.08]
    ft_bottom_y = [83.3, 61.9, 18.33]
    ft_random_y = [76.29, 59.09, 18.45]

    ax.plot(x, orig_ft_hyp_dino, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('Hyp_ViT Fine-tuned Color Object per ratio and location')

################################# Hyp_DeiT ############################################
##### Hyp_DeiT Zero-shot Black Object Occluded #####
if model == 'hyp_deit' and is_zero_shot == True and is_black == True and orig_finetuned == True:
    orig_zs_hyp_deit = [37.95, 37.95, 37.95]
    zs_top_y = [16.07, 8.05, 3.89]
    zs_center_y = [12.89, 6.57, 2.89]
    zs_bottom_y = [14.31, 7.31, 3.97]
    zs_random_y = [11.87, 6.1, 3.55]

    ax.plot(x, orig_zs_hyp_deit, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('Hyp_DeiT Zero-Shot Black Object per ratio and location')

##### Hyp_DeiT Fine-tuned Black Object Occluded #####
elif model == 'hyp_deit' and is_zero_shot == False and is_black == True and orig_finetuned == True:
    orig_ft_hyp_deit = [91.12, 91.12, 91.12]
    ft_top_y = [83.85, 76.23, 63.9]
    ft_center_y = [80.17, 70.05, 54.69]
    ft_bottom_y = [88.16, 84.89, 78.08]
    ft_random_y = [86.4, 81.37, 70.8]

    ax.plot(x, orig_ft_hyp_deit, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('Hyp_DeiT Fine-tuned Black Object per ratio and location')

##### Hyp_DeiT Zero-shot Color Object Occluded #####
elif model == 'hyp_deit' and is_zero_shot == True and is_black == False and orig_finetuned == True:
    # orig_zs_hyp_deit = [37.95, 37.95, 37.95] # too much difference so erased it out
    zs_top_y = [5.93, 2.06, 1.29]
    zs_center_y = [4.47, 1.66, 1.16]
    zs_bottom_y = [5.95, 2.29, 1.24]
    zs_random_y = [5.38, 2.43, 0.88]

    # ax.plot(x, orig_zs_hyp_deit, "o--", label='original dataset', color='black')
    ax.plot(x, zs_top_y, "o--", label='zero-shot top', color='green')
    ax.plot(x, zs_center_y, "o--", label='zero-shot center', color='hotpink')
    ax.plot(x, zs_bottom_y, "o--", label='zero-shot bottom', color='CornflowerBlue')
    ax.plot(x, zs_random_y, "o--", label='zero-shot random', color='#AA393F')

    plt.title('Hyp_DeiT Zero-Shot Color Object per ratio and location')

##### Hyp_DeiT Fine-tuned Color Object Occluded #####
elif model == 'hyp_deit' and is_zero_shot == False and is_black == False and orig_finetuned == True:
    orig_ft_hyp_deit = [91.12, 91.12, 91.12]
    ft_top_y = [79.5, 56.54, 21.01]
    ft_center_y = [69.54, 41.05, 13.25]
    ft_bottom_y = [85.26, 73.54, 33.57]
    ft_random_y = [81.74, 71.33, 33]

    ax.plot(x, orig_ft_hyp_deit, "o-", label='original dataset', color='black')
    ax.plot(x, ft_top_y, "o-", label='finetune top', color='green')
    ax.plot(x, ft_center_y, "o-", label='finetune center', color='hotpink')
    ax.plot(x, ft_bottom_y, "o-", label='finetune bottom', color='CornflowerBlue')
    ax.plot(x, ft_random_y, "o-", label='finetune random', color='#AA393F')

    plt.title('Hyp_DeiT Fine-tuned Color Object per ratio and location')

################################# 5 CENTER COLOR OCCLUDED FINETUNED ################################
################################# Hyp_ViT ############################################
##### Hyp_ViT 5 center occluded finetuned Black Object Occluded #####
if model == 'hyp_vit' and is_zero_shot == False and is_black == True and orig_finetuned == False:
    orig_finetuned_hyp_vit = [92.63, 92.63, 92.63]
    ratio5_center_finetuned_hyp_vit = [90.05, 90.05, 90.05]
    orig_finetuned_top_y = [86.64, 78.85, 65.51]
    orig_finetuned_center_y = [82.61, 72.89, 55.65]
    orig_finetuned_bottom_y = [89.3, 86.43, 79.53]
    orig_finetuned_random_y = [88.28, 83.29, 71.97]

    ratio5_center_finetuned_top_y = [88.81, 84.64, 76.32]
    ratio5_center_finetuned_center_y = [88.77, 84.14, 72.74]
    ratio5_center_finetuned_bottom_y = [91.11, 89.51, 85.71]
    ratio5_center_finetuned_random_y = [90.5, 89.02, 82.87]

    ax.plot(x, orig_finetuned_hyp_vit, "o--", label='original finetuned', color='black')
    ax.plot(x, ratio5_center_finetuned_hyp_vit, "o-", label='5 center occluded finetuned', color='red')
    ax.plot(x, orig_finetuned_top_y, "o--", label='orig finetuned top', color='green')
    ax.plot(x, orig_finetuned_center_y, "o--", label='orig finetuned center', color='hotpink')
    ax.plot(x, orig_finetuned_bottom_y, "o--", label='orig finetuned bottom', color='CornflowerBlue')
    ax.plot(x, orig_finetuned_random_y, "o--", label='orig finetuned random', color='#AA393F')
    ax.plot(x, ratio5_center_finetuned_top_y, "o-", label='5 center finetuned top', color='green')
    ax.plot(x, ratio5_center_finetuned_center_y, "o-", label='5 center finetuned center', color='hotpink')
    ax.plot(x, ratio5_center_finetuned_bottom_y, "o-", label='5 center finetuned bottom', color='CornflowerBlue')
    ax.plot(x, ratio5_center_finetuned_random_y, "o-", label='5 center finetuned random', color='#AA393F')

    plt.title('Hyp_ViT Original & 5 Center Finetuned Black Object')

##### Hyp_ViT 5 center occluded finetuned Color Object Occluded #####
elif model == 'hyp_vit' and is_zero_shot == False and is_black == False and orig_finetuned == False:
    orig_finetuned_hyp_vit = [92.63, 92.63, 92.63]
    ratio5_center_finetuned_hyp_vit = [90.05, 90.05, 90.05]
    orig_finetuned_top_y = [80.82, 58.55, 22.96]
    orig_finetuned_center_y = [70.39, 41.97, 13.82]
    orig_finetuned_bottom_y = [86.24, 72.86, 34.55]
    orig_finetuned_random_y = [82.16, 71.61, 32.16]

    ratio5_center_finetuned_top_y = [89.5, 84.82, 73.46]
    ratio5_center_finetuned_center_y = [90, 86.59, 75.28]
    ratio5_center_finetuned_bottom_y = [91.2, 89.41, 83.39]
    ratio5_center_finetuned_random_y = [90.8, 90, 82.08]

    ax.plot(x, orig_finetuned_hyp_vit, "o--", label='original finetuned', color='black')
    ax.plot(x, ratio5_center_finetuned_hyp_vit, "o-", label='5 center occluded finetuned', color='red')
    ax.plot(x, orig_finetuned_top_y, "o--", label='orig finetuned top', color='green')
    ax.plot(x, orig_finetuned_center_y, "o--", label='orig finetuned center', color='hotpink')
    ax.plot(x, orig_finetuned_bottom_y, "o--", label='orig finetuned bottom', color='CornflowerBlue')
    ax.plot(x, orig_finetuned_random_y, "o--", label='orig finetuned random', color='#AA393F')
    ax.plot(x, ratio5_center_finetuned_top_y, "o-", label='5 center finetuned top', color='green')
    ax.plot(x, ratio5_center_finetuned_center_y, "o-", label='5 center finetuned center', color='hotpink')
    ax.plot(x, ratio5_center_finetuned_bottom_y, "o-", label='5 center finetuned bottom', color='CornflowerBlue')
    ax.plot(x, ratio5_center_finetuned_random_y, "o-", label='5 center finetuned random', color='#AA393F')

    plt.title('Hyp_ViT Original & 5 Center Finetuned Color Object')

################################# Hyp_Dino ############################################
##### Hyp_Dino 5 center occluded finetuned Black Object Occluded #####
if model == 'hyp_dino' and is_zero_shot == False and is_black == True and orig_finetuned == False:
    orig_finetuned_hyp_dino = [91.79, 91.79, 91.79]
    ratio5_center_finetuned_hyp_dino = [89.25, 89.25, 89.25]
    orig_finetuned_top_y = [85.14, 77.09, 64.43]
    orig_finetuned_center_y = [80.12, 69.24, 54.13]
    orig_finetuned_bottom_y = [88.79, 85.36, 76.92]
    orig_finetuned_random_y = [85.91, 80.69, 69.05]

    ratio5_center_finetuned_top_y = [87.79, 83.44, 76.04]
    ratio5_center_finetuned_center_y = [87.99, 82.92, 72.24]
    ratio5_center_finetuned_bottom_y = [90.29, 88.67, 85.18]
    ratio5_center_finetuned_random_y = [89.63, 87.9, 81.45]

    ax.plot(x, orig_finetuned_hyp_dino, "o--", label='original finetuned', color='black')
    ax.plot(x, ratio5_center_finetuned_hyp_dino, "o-", label='5 center occluded finetuned', color='red')
    ax.plot(x, orig_finetuned_top_y, "o--", label='orig finetuned top', color='green')
    ax.plot(x, orig_finetuned_center_y, "o--", label='orig finetuned center', color='hotpink')
    ax.plot(x, orig_finetuned_bottom_y, "o--", label='orig finetuned bottom', color='CornflowerBlue')
    ax.plot(x, orig_finetuned_random_y, "o--", label='orig finetuned random', color='#AA393F')
    ax.plot(x, ratio5_center_finetuned_top_y, "o-", label='5 center finetuned top', color='green')
    ax.plot(x, ratio5_center_finetuned_center_y, "o-", label='5 center finetuned center', color='hotpink')
    ax.plot(x, ratio5_center_finetuned_bottom_y, "o-", label='5 center finetuned bottom', color='CornflowerBlue')
    ax.plot(x, ratio5_center_finetuned_random_y, "o-", label='5 center finetuned random', color='#AA393F')

    plt.title('Hyp_Dino Original & 5 Center Finetuned Black Object')

##### Hyp_Dino 5 center occluded finetuned Color Object Occluded #####
elif model == 'hyp_dino' and is_zero_shot == False and is_black == False and orig_finetuned == False:
    orig_finetuned_hyp_dino = [91.79, 91.79, 91.79]
    ratio5_center_finetuned_hyp_dino = [89.25, 89.25, 89.25]
    orig_finetuned_top_y = [77.49, 50.06, 14.56]
    orig_finetuned_center_y = [58.13, 26.6, 7.08]
    orig_finetuned_bottom_y = [83.3, 61.9, 18.33]
    orig_finetuned_random_y = [76.29, 59.09, 18.45]

    ratio5_center_finetuned_top_y = [88.9, 83.9, 71.7]
    ratio5_center_finetuned_center_y = [88.85, 85.17, 70.87]
    ratio5_center_finetuned_bottom_y = [90.42, 88.48, 80.71]
    ratio5_center_finetuned_random_y = [90.34, 88.99, 79.91]

    ax.plot(x, orig_finetuned_hyp_dino, "o--", label='original finetuned', color='black')
    ax.plot(x, ratio5_center_finetuned_hyp_dino, "o-", label='5 center occluded finetuned', color='red')
    ax.plot(x, orig_finetuned_top_y, "o--", label='orig finetuned top', color='green')
    ax.plot(x, orig_finetuned_center_y, "o--", label='orig finetuned center', color='hotpink')
    ax.plot(x, orig_finetuned_bottom_y, "o--", label='orig finetuned bottom', color='CornflowerBlue')
    ax.plot(x, orig_finetuned_random_y, "o--", label='orig finetuned random', color='#AA393F')
    ax.plot(x, ratio5_center_finetuned_top_y, "o-", label='5 center finetuned top', color='green')
    ax.plot(x, ratio5_center_finetuned_center_y, "o-", label='5 center finetuned center', color='hotpink')
    ax.plot(x, ratio5_center_finetuned_bottom_y, "o-", label='5 center finetuned bottom', color='CornflowerBlue')
    ax.plot(x, ratio5_center_finetuned_random_y, "o-", label='5 center finetuned random', color='#AA393F')

    plt.title('Hyp_Dino Original & 5 Center Finetuned Color Object')

################################# Hyp_DeiT ############################################
##### Hyp_DeiT 5 center occluded finetuned Black Object Occluded #####
if model == 'hyp_deit' and is_zero_shot == False and is_black == True and orig_finetuned == False:
    orig_finetuned_hyp_deit = [91.12, 91.12, 91.12]
    ratio5_center_finetuned_hyp_deit = [88.39, 88.39, 88.39]
    orig_finetuned_top_y = [83.85, 76.23, 63.9]
    orig_finetuned_center_y = [80.17, 70.05, 54.69]
    orig_finetuned_bottom_y = [88.16, 84.89, 78.08]
    orig_finetuned_random_y = [86.4, 81.37, 70.8]

    ratio5_center_finetuned_top_y = [87, 82.66, 74.19]
    ratio5_center_finetuned_center_y = [87.14, 82.4, 71.24]
    ratio5_center_finetuned_bottom_y = [88.98, 87.38, 83.29]
    ratio5_center_finetuned_random_y = [88.74, 87.14, 81.17]

    ax.plot(x, orig_finetuned_hyp_deit, "o--", label='original finetuned', color='black')
    ax.plot(x, ratio5_center_finetuned_hyp_deit, "o-", label='5 center occluded finetuned', color='red')
    ax.plot(x, orig_finetuned_top_y, "o--", label='orig finetuned top', color='green')
    ax.plot(x, orig_finetuned_center_y, "o--", label='orig finetuned center', color='hotpink')
    ax.plot(x, orig_finetuned_bottom_y, "o--", label='orig finetuned bottom', color='CornflowerBlue')
    ax.plot(x, orig_finetuned_random_y, "o--", label='orig finetuned random', color='#AA393F')
    ax.plot(x, ratio5_center_finetuned_top_y, "o-", label='5 center finetuned top', color='green')
    ax.plot(x, ratio5_center_finetuned_center_y, "o-", label='5 center finetuned center', color='hotpink')
    ax.plot(x, ratio5_center_finetuned_bottom_y, "o-", label='5 center finetuned bottom', color='CornflowerBlue')
    ax.plot(x, ratio5_center_finetuned_random_y, "o-", label='5 center finetuned random', color='#AA393F')

    plt.title('Hyp_DeiT Original & 5 Center Finetuned Black Object')

##### Hyp_DeiT 5 center occluded finetuned Color Object Occluded #####
elif model == 'hyp_deit' and is_zero_shot == False and is_black == False and orig_finetuned == False:
    orig_finetuned_hyp_deit = [91.12, 91.12, 91.12]
    ratio5_center_finetuned_hyp_deit = [88.39, 88.39, 88.39]
    orig_finetuned_top_y = [79.5, 56.54, 21.01]
    orig_finetuned_center_y = [69.54, 41.05, 13.25]
    orig_finetuned_bottom_y = [85.26, 73.54, 33.57]
    orig_finetuned_random_y = [81.74, 71.33, 33]

    ratio5_center_finetuned_top_y = [87.62, 82.94, 71.51]
    ratio5_center_finetuned_center_y = [88, 84.3, 70.59]
    ratio5_center_finetuned_bottom_y = [89.15, 87.47, 79.58]
    ratio5_center_finetuned_random_y = [89.24, 88.19, 79.03]

    ax.plot(x, orig_finetuned_hyp_deit, "o--", label='original finetuned', color='black')
    ax.plot(x, ratio5_center_finetuned_hyp_deit, "o-", label='5 center occluded finetuned', color='red')
    ax.plot(x, orig_finetuned_top_y, "o--", label='orig finetuned top', color='green')
    ax.plot(x, orig_finetuned_center_y, "o--", label='orig finetuned center', color='hotpink')
    ax.plot(x, orig_finetuned_bottom_y, "o--", label='orig finetuned bottom', color='CornflowerBlue')
    ax.plot(x, orig_finetuned_random_y, "o--", label='orig finetuned random', color='#AA393F')
    ax.plot(x, ratio5_center_finetuned_top_y, "o-", label='5 center finetuned top', color='green')
    ax.plot(x, ratio5_center_finetuned_center_y, "o-", label='5 center finetuned center', color='hotpink')
    ax.plot(x, ratio5_center_finetuned_bottom_y, "o-", label='5 center finetuned bottom', color='CornflowerBlue')
    ax.plot(x, ratio5_center_finetuned_random_y, "o-", label='5 center finetuned random', color='#AA393F')

    plt.title('Hyp_DeiT Original & 5 Center Finetuned Color Object')

plt.xticks([5,10,20])
plt.xlabel('Occlusion Ratio')
plt.ylabel('Recall@1')

# Shrink current axis by 20%
box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.6, box.height]) # for original finetuned
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height]) # for 5 center color occluded finetuned
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

##### save orig finetuned UNICOM #####
if model == 'unicom' and is_zero_shot == True and is_black == True and orig_finetuned == True: # unicom zeroshot black
    plt.savefig('./unicom_zero_shot_black_object_ratio_location.png')
elif model == 'unicom' and is_zero_shot == False and is_black == True and orig_finetuned == True: # unicom finetune black
    plt.savefig('./unicom_finetuned_black_object_ratio_location.png')
elif model == 'unicom' and is_zero_shot == True and is_black == False and orig_finetuned == True: # unicom zeroshot color
    plt.savefig('./unicom_zero_shot_color_object_ratio_location.png')
elif model == 'unicom' and is_zero_shot == False and is_black == False and orig_finetuned == True: # unicom finetune color
    plt.savefig('./unicom_finetuned_color_object_ratio_location.png')

##### save orig finetuned Hyp_ViT #####
if model == 'hyp_vit' and is_zero_shot == True and is_black == True and orig_finetuned == True: # hyp_vit zeroshot black
    plt.savefig('./hyp_vit_zero_shot_black_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == False and is_black == True and orig_finetuned == True: # hyp_vit finetune black
    plt.savefig('./hyp_vit_finetuned_black_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == True and is_black == False and orig_finetuned == True: # hyp_vit zeroshot color
    plt.savefig('./hyp_vit_zero_shot_color_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == False and is_black == False and orig_finetuned == True: # hyp_vit finetune color
    plt.savefig('./hyp_vit_finetuned_color_object_ratio_location.png')

##### save orig finetuned Hyp_Dino #####
if model == 'hyp_dino' and is_zero_shot == True and is_black == True and orig_finetuned == True: # hyp_dino zeroshot black
    plt.savefig('./hyp_dino_zero_shot_black_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == False and is_black == True and orig_finetuned == True: # hyp_dino finetune black
    plt.savefig('./hyp_dino_finetuned_black_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == True and is_black == False and orig_finetuned == True: # hyp_dino zeroshot color
    plt.savefig('./hyp_dino_zero_shot_color_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == False and is_black == False and orig_finetuned == True: # hyp_dino finetune color
    plt.savefig('./hyp_dino_finetuned_color_object_ratio_location.png')

##### save orig finetuned Hyp_DeiT #####
if model == 'hyp_deit' and is_zero_shot == True and is_black == True and orig_finetuned == True: # hyp_deit zeroshot black
    plt.savefig('./hyp_deit_zero_shot_black_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == False and is_black == True and orig_finetuned == True: # hyp_deit finetune black
    plt.savefig('./hyp_deit_finetuned_black_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == True and is_black == False and orig_finetuned == True: # hyp_deit zeroshot color
    plt.savefig('./hyp_deit_zero_shot_color_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == False and is_black == False and orig_finetuned == True: # hyp_deit finetune color
    plt.savefig('./hyp_deit_finetuned_color_object_ratio_location.png')

##### save 5 center color occluded finetuned Hyp_ViT #####
if model == 'hyp_vit' and is_zero_shot == False and is_black == True and orig_finetuned == False: # hyp_vit finetune black
    plt.savefig('./hyp_vit_5center_finetuned_black_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == False and is_black == False and orig_finetuned == False: # hyp_vit finetune color
    plt.savefig('./hyp_vit_5center_finetuned_color_object_ratio_location.png')

##### save 5 center color occluded finetuned Hyp_Dino #####
if model == 'hyp_dino' and is_zero_shot == False and is_black == True and orig_finetuned == False: # hyp_dino finetune black
    plt.savefig('./hyp_dino_5center_finetuned_black_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == False and is_black == False and orig_finetuned == False: # hyp_dino finetune color
    plt.savefig('./hyp_dino_5center_finetuned_color_object_ratio_location.png')

##### save 5 center color occluded finetuned Hyp_DeiT #####
if model == 'hyp_deit' and is_zero_shot == False and is_black == True and orig_finetuned == False: # hyp_deit finetune black
    plt.savefig('./hyp_deit_5center_finetuned_black_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == False and is_black == False and orig_finetuned == False: # hyp_deit finetune color
    plt.savefig('./hyp_deit_5center_finetuned_color_object_ratio_location.png')

plt.show()