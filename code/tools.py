import re


def model_index(models, filenames):
    models = {
        'AttnGAN_normal':0, 'DALLE2_normal':1, 'glide_normal':2, 
        'midjourney_lowstep':3, 'midjourney_normal':3, 
        'sd1.5_highcorr':4, 'sd1.5_lowcorr':4, 'sd1.5_lowstep':4, 'sd1.5_normal':4, 
        'xl2.2_normal':5
    }
    matched_indices = []

    for filename in filenames:
        # 使用正则表达式匹配并提取文件名中的变化部分
        match = re.match(r"(.+?)_(\d+)\.jpg$", filename) #如midjourney_lowstep_000.jpg
        if match:
            base_name = match.group(1)
            # 使用列表推导式和 startswith 方法进行前缀匹配
            # matched_index = [i for i, prefix in enumerate(models) if base_name == prefix]
            index = models[base_name]
            matched_indices.append(index)
        else:
            matched_indices.append(None)

    return matched_indices


def model_index_after(models, filenames, exclude_index):
    models_all = [
        'AttnGAN_normal', 'DALLE2_normal', 'glide_normal', 
        ['midjourney_lowstep', 'midjourney_normal'], 
        ['sd1.5_highcorr', 'sd1.5_lowcorr', 'sd1.5_lowstep', 'sd1.5_normal'], 
        'xl2.2_normal'
    ]
    models = models_all[:exclude_index] + models_all[exclude_index+1:]
    # 形成编号列表
    model_dict = {}
    counter = 0
    for item in models:
        if isinstance(item, list):  # 检查是否为列表
            for sub_item in item:
                model_dict[sub_item] = counter
        else:
            model_dict[item] = counter
        counter += 1  

    matched_indices = []
    for filename in filenames:
        # 使用正则表达式匹配并提取文件名中的变化部分
        match = re.match(r"(.+?)_(\d+)\.jpg$", filename) #如midjourney_lowstep_000.jpg
        if match:
            base_name = match.group(1)
            # 使用列表推导式和 startswith 方法进行前缀匹配
            # matched_index = [i for i, prefix in enumerate(models) if base_name == prefix]
            index = model_dict[base_name]
            matched_indices.append(index)
        else:
            print(filenames, 'model target error')

    return matched_indices