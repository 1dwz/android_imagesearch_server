# Android 图像搜索服务器

一个基于 FastAPI 的图像匹配 API 服务器，支持通过 INI 文件配置进行模板匹配。它能从 MJPEG 视频流中捕获图像，并允许通过管理器脚本启动和管理多个独立的服务器实例。

## 功能特点

- 支持从 MJPEG 流实时捕获图像
- 支持多种图像处理和匹配方法
- 支持通过 INI 文件配置匹配参数
- 支持多实例管理
- 健壮的错误处理和参数验证
- 完整的日志记录和调试支持

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd android_imagesearch_server
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 启动服务器

1. 单实例模式：
```bash
python main_instance.py --port 8000 --mjpeg_url http://example.com/mjpeg
```

2. 多实例管理模式：
```bash
python manager.py --config multi_config.json
```

### API 端点

#### 1. 图像搜索 (`/search`)

使用查询参数进行模板匹配：

```
GET /search?img=/path/to/template.png&threshold=0.8&filter_type=canny
```

参数：
- `img`: 模板图像的路径（必需）
- `filter_type`: 过滤器类型 ('none' 或 'canny')
- `match_method`: 匹配方法
- `threshold`: 匹配阈值 (0.0-1.0)
- `x1`, `y1`, `x2`, `y2`: 搜索区域坐标
- `offsetx`, `offsety`: 结果坐标的偏移
- `canny_t1`, `canny_t2`: Canny 边缘检测参数

#### 2. INI 配置搜索 (`/search_ini`)

使用 INI 文件进行模板匹配：

```
GET /search_ini?ini_path=/path/to/config.ini
```

参数：
- `ini_path`: INI 配置文件的路径（必需）

#### 3. 健康检查 (`/health`)

检查服务器状态：

```
GET /health
```

### 错误处理

服务器使用以下错误类型进行错误处理：

1. **`ImageSearchError`**
   - 所有错误的基类
   - 用于通用错误处理

2. **`InvalidParameterError`**
   - 当提供的参数无效时抛出
   - HTTP 状态码：400 Bad Request
   - 示例：无效的阈值、坐标超出范围

3. **`TemplateLoadError`**
   - 当模板图像加载失败时抛出
   - HTTP 状态码：404 Not Found
   - 示例：模板文件不存在、文件格式无效

4. **`FilterProcessingError`**
   - 当图像过滤处理失败时抛出
   - HTTP 状态码：500 Internal Server Error
   - 示例：Canny 边缘检测失败

5. **`TemplateMatchError`**
   - 当模板匹配过程失败时抛出
   - HTTP 状态码：500 Internal Server Error
   - 示例：匹配算法失败、内存不足

6. **`MJPEGStreamError`**
   - 当 MJPEG 流操作失败时抛出
   - HTTP 状态码：503 Service Unavailable
   - 示例：流连接失败、帧解码错误

7. **`CacheError`**
   - 当缓存操作失败时抛出
   - HTTP 状态码：500 Internal Server Error
   - 示例：缓存写入失败、缓存已满

错误响应格式：
```json
{
    "error": "错误消息",
    "error_type": "错误类型名称",
    "details": {
        "path": "/api/endpoint",
        "method": "GET",
        "timestamp": 1234567890.123
    }
}
```

### 参数验证

服务器使用 `ParameterValidator` 类进行参数验证：

1. **阈值验证**
   ```python
   threshold = ParameterValidator.validate_threshold(0.8)  # 0.0-1.0
   ```

2. **坐标验证**
   ```python
   x, y = ParameterValidator.validate_coordinates(100, 200, width, height)
   ```

3. **搜索区域验证**
   ```python
   region = ParameterValidator.validate_search_region(x1, y1, x2, y2, width, height)
   ```

4. **Canny 参数验证**
   ```python
   t1, t2 = ParameterValidator.validate_canny_params(100, 200)  # 0-255
   ```

## 配置文件

### 1. 服务器配置 (`server.ini`)

```ini
[Server]
idle_timeout = 300
wakeup_timeout = 15
enable_debug_saving = false
debug_save_dir = debug_images

[MatchDefaults]
filter_type = none
match_method = ccoeff_normed
threshold = 0.8
canny_t1 = 100
canny_t2 = 200
```

### 2. 多实例配置 (`multi_config.json`)

```json
{
    "instances": [
        {
            "port": 8000,
            "mjpeg_url": "http://example1.com/mjpeg"
        },
        {
            "port": 8001,
            "mjpeg_url": "http://example2.com/mjpeg"
        }
    ]
}
```

### 3. 匹配配置 (`.ini`)

```ini
[MatchSettings]
template_path = /path/to/template.png
filter_type = canny
match_method = ccoeff_normed
threshold = 0.8
canny_t1 = 100
canny_t2 = 200
match_range_x1 = 0
match_range_y1 = 0
match_range_x2 = 1920
match_range_y2 = 1080
offset_x = 0
offset_y = 0
```

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码风格

项目遵循 PEP 8 代码风格指南。使用 `black` 进行代码格式化：

```bash
black .
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
