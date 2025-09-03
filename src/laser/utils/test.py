def get_tensor_parallel_size():
    """
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 2  # 如果没有GPU，默认返回2
        
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return 2
        
        # 检查第一个GPU的信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
        
        # 如果是3090或者显存小于30GB，设置为2
        if "3090" in gpu_name or gpu_memory < 30:
            return 2
        return 1
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return 2  # 出错时默认返回2
    
print(get_tensor_parallel_size())