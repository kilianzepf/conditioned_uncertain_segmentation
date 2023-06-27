"""
Store all dataset/task relevant meta data here for passing them to the training script.
"""


def get_meta(task):

    if task == 'isic3':
        meta = {
            'description': 'ISIC Skin Lesion Dataset (subset with 3 annotations)',
            'channels': 3,
            'all_data_path': '/home/kmze//uncertain_segmentation/data/isic3/isic256_3',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'isic3_style':
        meta = {
            'description': 'ISIC Skin Lesion Dataset with Styles (subset with 3 annotations)',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/isic3/isic256_3_style',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'isic3_style_0':
        meta = {
            'description': 'ISIC Skin Lesion Dataset with only Style 0 ',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/isic3/isic256_3_style',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'isic3_style_1':
        meta = {
            'description': 'ISIC Skin Lesion Dataset with only Style 1 ',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/isic3/isic256_3_style',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'isic3_style_2':
        meta = {
            'description': 'ISIC Skin Lesion Dataset with only Style 2 ',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/isic3/isic256_3_style',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'isic3_style_concat':
        meta = {
            'description': 'ISIC Skin Lesion Dataset with same split as style subsets ',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/isic3/isic256_3_style',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'isic3_dynamic_aug':
        meta = {
            'description': 'ISIC Skin Lesion Dataset with dynamic augmentation ',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/isic3/isic256_3_style',
            'masking_threshold': 0.5,
            'image_size': 256,
            'admissible_size': 340,
            'output_size': 252,
            'directory_name': 'isic3',
            'raters': 3,
            'num_filters': [32, 64, 128, 192],
            # 'lossfunction': define lossfunction here
        }
        return meta

    if task == 'phc':
        meta = {
            'description': 'PHC cell Dataset (5 annotations)',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/phc_data',
            'masking_threshold': 0.5,
            'image_size': 128,
            'admissible_size': 128,
            'output_size': 128,
            'directory_name': 'phc',
            'raters': 6,
            'num_filters': [32, 64, 128, 192],

        }
        return meta

    if task == 'phc_style':
        meta = {
            'description': 'PHC cell Dataset with style labels (5 annotations)',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/phc_data',
            'masking_threshold': 0.5,
            'image_size': 128,
            'admissible_size': 128,
            'output_size': 128,
            'directory_name': 'phc',
            'raters': 6,
            'num_filters': [32, 64, 128, 192],

        }
        return meta

    if task == 'phc_style_0':
        meta = {
            'description': 'PHC cell Dataset only Style 0 (subset with 3 annotations)',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/phc_data',
            'masking_threshold': 0.5,
            'image_size': 128,
            'admissible_size': 128,
            'output_size': 128,
            'directory_name': 'phc',
            'raters': 6,
            'num_filters': [32, 64, 128, 192],

        }
        return meta

    if task == 'phc_style_1':
        meta = {
            'description': 'PHC cell Dataset only Style 1 (subset with 2 annotations)',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/phc_data',
            'masking_threshold': 0.5,
            'image_size': 128,
            'admissible_size': 128,
            'output_size': 128,
            'directory_name': 'phc',
            'raters': 6,
            'num_filters': [32, 64, 128, 192],

        }
        return meta

    if task == 'phc_style_concat':
        meta = {
            'description': 'PHC style cell Dataset with same split as style subsets',
            'channels': 3,
            'all_data_path': '/home/kmze/style_probunet/data/phc_data',
            'masking_threshold': 0.5,
            'image_size': 128,
            'admissible_size': 128,
            'output_size': 128,
            'directory_name': 'phc',
            'raters': 6,
            'num_filters': [32, 64, 128, 192],

        }
        return meta

    # Test comment2
