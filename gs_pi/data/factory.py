from .registry import get_dataset


def create_dataset(name, **kwargs):
    dataset_cls = get_dataset(name=name)
    # 将配置文件中的参数（如 point_cloud_path 等）传给类构造函数
    dataset = dataset_cls(**kwargs)
    return dataset
