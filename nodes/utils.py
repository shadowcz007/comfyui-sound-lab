import os

# 写一个python文件，用来 判断文件夹内命名为 所有chat_tts开头的文件数量（chat_tts_00001），并输出新的编号
def get_new_counter(full_output_folder, filename_prefix):
    # 获取目录中的所有文件
    files = os.listdir(full_output_folder)
    
    # 过滤出以 filename_prefix 开头并且后续部分为数字的文件
    filtered_files = []
    for f in files:
        if f.startswith(filename_prefix):
            # 去掉文件名中的前缀和后缀，只保留中间的数字部分
            base_name = f[len(filename_prefix)+1:]
            number_part = base_name.split('.')[0]  # 假设文件名中只有一个点，即扩展名
            if number_part.isdigit():
                filtered_files.append(int(number_part))

    if not filtered_files:
        return 1

    # 获取最大的编号
    max_number = max(filtered_files)
    
    # 新的编号
    return max_number + 1