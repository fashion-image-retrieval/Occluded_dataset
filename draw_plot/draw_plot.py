import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

x = [5, 10, 20]

model= 'hyp_deit' # 'unicom' or 'hyp_vit' or 'hyp_dino' or 'hyp_deit'
is_zero_shot = False # True if zero-shot, False if fine-tuned
is_black = False # True if black occluded, False if color occluded

################################# UNICOM ############################################
##### UNICOM Zero-shot Black Object Occluded #####
if model == 'unicom' and is_zero_shot == True and is_black == True:
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
elif model == 'unicom' and is_zero_shot == False and is_black == True:
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
elif model == 'unicom' and is_zero_shot == True and is_black == False:
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
elif model == 'unicom' and is_zero_shot == False and is_black == False:
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
if model == 'hyp_vit' and is_zero_shot == True and is_black == True:
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
elif model == 'hyp_vit' and is_zero_shot == False and is_black == True:
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
elif model == 'hyp_vit' and is_zero_shot == True and is_black == False:
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
elif model == 'hyp_vit' and is_zero_shot == False and is_black == False:
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
if model == 'hyp_dino' and is_zero_shot == True and is_black == True:
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
elif model == 'hyp_dino' and is_zero_shot == False and is_black == True:
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
elif model == 'hyp_dino' and is_zero_shot == True and is_black == False:
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
elif model == 'hyp_dino' and is_zero_shot == False and is_black == False:
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
if model == 'hyp_deit' and is_zero_shot == True and is_black == True:
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
elif model == 'hyp_deit' and is_zero_shot == False and is_black == True:
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
elif model == 'hyp_deit' and is_zero_shot == True and is_black == False:
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
elif model == 'hyp_deit' and is_zero_shot == False and is_black == False:
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

plt.xticks([5,10,20])
plt.xlabel('Occlusion Ratio')
plt.ylabel('Recall@1')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

##### save UNICOM #####
if model == 'unicom' and is_zero_shot == True and is_black == True: # unicom zeroshot black
    plt.savefig('./unicom_zero_shot_black_object_ratio_location.png')
elif model == 'unicom' and is_zero_shot == False and is_black == True: # unicom finetune black
    plt.savefig('./unicom_finetuned_black_object_ratio_location.png')
elif model == 'unicom' and is_zero_shot == True and is_black == False: # unicom zeroshot color
    plt.savefig('./unicom_zero_shot_color_object_ratio_location.png')
elif model == 'unicom' and is_zero_shot == False and is_black == False: # unicom finetune color
    plt.savefig('./unicom_finetuned_color_object_ratio_location.png')

##### save Hyp_ViT #####
if model == 'hyp_vit' and is_zero_shot == True and is_black == True: # hyp_vit zeroshot black
    plt.savefig('./hyp_vit_zero_shot_black_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == False and is_black == True: # hyp_vit finetune black
    plt.savefig('./hyp_vit_finetuned_black_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == True and is_black == False: # hyp_vit zeroshot color
    plt.savefig('./hyp_vit_zero_shot_color_object_ratio_location.png')
elif model == 'hyp_vit' and is_zero_shot == False and is_black == False: # hyp_vit finetune color
    plt.savefig('./hyp_vit_finetuned_color_object_ratio_location.png')

##### save Hyp_Dino #####
if model == 'hyp_dino' and is_zero_shot == True and is_black == True: # hyp_dino zeroshot black
    plt.savefig('./hyp_dino_zero_shot_black_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == False and is_black == True: # hyp_dino finetune black
    plt.savefig('./hyp_dino_finetuned_black_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == True and is_black == False: # hyp_dino zeroshot color
    plt.savefig('./hyp_dino_zero_shot_color_object_ratio_location.png')
elif model == 'hyp_dino' and is_zero_shot == False and is_black == False: # hyp_dino finetune color
    plt.savefig('./hyp_dino_finetuned_color_object_ratio_location.png')

##### save Hyp_DeiT #####
if model == 'hyp_deit' and is_zero_shot == True and is_black == True: # hyp_deit zeroshot black
    plt.savefig('./hyp_deit_zero_shot_black_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == False and is_black == True: # hyp_deit finetune black
    plt.savefig('./hyp_deit_finetuned_black_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == True and is_black == False: # hyp_deit zeroshot color
    plt.savefig('./hyp_deit_zero_shot_color_object_ratio_location.png')
elif model == 'hyp_deit' and is_zero_shot == False and is_black == False: # hyp_deit finetune color
    plt.savefig('./hyp_deit_finetuned_color_object_ratio_location.png')

plt.show()