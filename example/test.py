# filename = batch_convert_ini_overwrite.py
import configparser
import glob
import os


def is_old_ini_format(filepath: str) -> bool:
    """
    Checks if an INI file is in the old format by looking for the absence of template_path.

    Args:
        filepath: The full path to the INI file.

    Returns:
        True if it's considered old format, False otherwise or if an error occurs.
    """
    if not os.path.exists(filepath):
        return False

    parser = configparser.ConfigParser()
    try:
        # 尝试读取文件，如果失败则认为不是有效的INI或无法处理
        parser.read(filepath, encoding="utf-8")
    except configparser.Error:
        # 如果读取出错，可能不是INI文件或者格式有问题，跳过
        return False

    section = "MatchSettings"
    # 判断是否存在 [MatchSettings] section 且该 section 中没有 template_path 键
    return parser.has_section(section) and not parser.has_option(
        section, "template_path"
    )


def convert_and_overwrite_ini(ini_filepath: str):
    """
    Reads an old INI file, converts it to the new format, and overwrites the original file.

    Args:
        ini_filepath: The full path to the INI file to convert and overwrite.
    """
    # 再次确认文件存在
    if not os.path.exists(ini_filepath):
        print(f"错误: 转换并覆盖失败，找不到INI文件: {ini_filepath}")
        return

    # 读取旧的INI文件
    old_parser = configparser.ConfigParser()
    try:
        old_parser.read(ini_filepath, encoding="utf-8")
    except configparser.Error as e:
        print(f"读取INI文件时出错 {ini_filepath}: {e}")
        return

    # 创建新的INI结构
    new_parser = configparser.ConfigParser()
    section = "MatchSettings"
    # 确保有 MatchSettings section，即使旧文件中没有
    if not new_parser.has_section(section):
        new_parser.add_section(section)

    # 定义新INI文件中的所有键和它们的默认值 (如果旧文件没有这些键)
    # threshold 的值固定为 0.8
    new_settings_template = {
        "template_path": "",  # This will be derived
        "filter_type": "none",
        "match_method": "ccoeff_normed",
        "threshold": "0.8",  # Fixed value as string
        "canny_t1": "100",
        "canny_t2": "200",
        "match_range_x1": "0",
        "match_range_y1": "0",
        "match_range_x2": "1920",
        "match_range_y2": "1080",
        "offset_x": "0",
        "offset_y": "0",
    }

    # 从旧文件中复制存在的键值，并应用新规则或默认值
    # 遍历新模板中的键，确保新文件包含所有这些键
    for key, default_value in new_settings_template.items():
        # 特殊处理 template_path
        if key == "template_path":
            # 根据INI文件的完整路径生成对应的JPG路径
            base_name = os.path.splitext(ini_filepath)[0]
            template_jpg_path = f"{base_name}.jpg"
            new_parser.set(section, key, template_jpg_path)
        # 特殊处理 threshold，固定为 0.8
        elif key == "threshold":
            new_parser.set(section, key, "0.8")
        # 从旧文件复制其他键的值
        elif old_parser.has_section(section) and old_parser.has_option(section, key):
            new_parser.set(section, key, old_parser.get(section, key))
        # 如果旧文件中没有，则使用默认值
        else:
            new_parser.set(section, key, default_value)

    # 保存新的INI文件 (覆盖原文件)
    try:
        # 在写入之前，可以考虑先读取原文件内容，如果一致则不写入，减少不必要的写操作
        # 但为了简单和确保转换，这里直接写入
        with open(ini_filepath, "w", encoding="utf-8") as configfile:
            new_parser.write(configfile)
        print(f"  成功转换并覆盖文件: {ini_filepath}")
    except (IOError, OSError, configparser.Error) as e:
        print(f"  写入INI文件时出错 {ini_filepath}: {e}")


if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本运行目录: {script_dir}")
    print("开始搜索并转换旧格式的INI文件 (将覆盖原文件)...")

    # 使用 glob 查找当前目录及其所有子目录下的所有 .ini 文件
    ini_files = glob.glob(os.path.join(script_dir, "**", "*.ini"), recursive=True)

    if not ini_files:
        print("未找到任何INI文件。")
    else:
        print(f"共找到 {len(ini_files)} 个INI文件，开始检查格式并转换...")
        processed_count = 0
        for ini_file_path in ini_files:
            # 忽略脚本自身
            if os.path.abspath(ini_file_path) == os.path.abspath(__file__):
                continue

            print(f"\n检查文件: {ini_file_path}")
            # 在覆盖模式下，我们只处理确定是旧格式的文件
            if is_old_ini_format(ini_file_path):
                print("  -> 检测到旧格式，准备转换并覆盖...")
                convert_and_overwrite_ini(ini_file_path)
                processed_count += 1
            else:
                print("  -> 检测到新格式或无效格式，跳过。")

        print(f"\n转换完成。共处理并覆盖 {processed_count} 个旧格式INI文件。")
