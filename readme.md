# 图像匹配 API 文档 (V6.0 - 多实例管理 - INI驱动 - INI可选滤镜 - 默认原始匹配)

**版本:** 6.0 (在 `modules/constants.py` 中定义)

## 目录

1.  [概述](#1-概述)
2.  [核心工作流程](#2-核心工作流程)
3.  [主要特性](#3-主要特性)
4.  [快速入门：通过 INI 文件进行图像匹配](#4-快速入门通过-ini-文件进行图像匹配)
    *   [步骤 1: 创建模板图像和 INI 配置文件](#步骤-1-创建模板图像和-ini-配置文件)
    *   [步骤 2: 启动服务器实例](#步骤-2-启动服务器实例)
    *   [步骤 3: 发送 API 请求](#步骤-3-发送-api-请求)
    *   [步骤 4: 查看结果](#步骤-4-查看结果)
5.  [INI 配置文件详解](#5-ini-配置文件详解)
    *   [标准 INI 格式与 \`[MatchSettings]\` 部分](#标准-ini-格式与-matchsettings-部分)
    *   [\`[MatchSettings]\` 配置项详解与优先级](#matchsettings-配置项详解与优先级)
6.  [API 端点](#6-api-端点)
    *   [6.1. 公共 API 端点](#61-公共-api-端点)
        *   [`/search_ini` (GET) - 通过 INI 文件执行图像搜索](#search_ini-get---通过-ini-文件执行图像搜索)
        *   [`/search` 或 `/` (GET) - 直接参数化图像搜索 (高级/备用)](#search-或--get---直接参数化图像搜索-高级备用)
        *   [`/health` (GET) - 健康检查](#health-get---健康检查)
    *   [6.2. 内部管理 API 端点 (供管理器使用)](#62-内部管理-api-端点-供管理器使用)
7.  [多实例管理器 (Manager) 使用](#7-多实例管理器-manager-使用)
    *   [7.1. 架构概述](#71-架构概述)
    *   [7.2. 依赖安装](#72-依赖安装)
    *   [7.3. 配置文件](#73-配置文件)
        *   [`multi_config.json` (多实例配置)](#multi_configjson-多实例配置)
        *   [`config/server.ini` (通用服务器配置)](#configserverini-通用服务器配置)
    *   [7.4. Manager 命令行命令](#74-manager-命令行命令)
        *   [`start`](#start)
        *   [`stop`](#stop)
        *   [`status`](#status)
        *   [`update`](#update)
        *   [`create`](#create)
        *   [`delete`](#delete)
    *   [7.5. 示例 Manager 命令](#75-示例-manager-命令)
8.  [通用 API 说明](#8-通用-api-说明)
9.  [响应体详解 (成功匹配)](#9-响应体详解-成功匹配)
10. [错误处理](#10-错误处理)
11. [版本变更](#11-版本变更)

---

## 1. 概述

本服务提供了一个基于 FastAPI 的 HTTP API，用于通过 **INI 配置文件**驱动的图像模板匹配。现在，它支持通过一个 **管理器脚本 (`manager.py`)** 启动和管理**多个独立的服务器实例**。每个实例可以监听不同的端口，并连接到不同的 MJPEG 视频流，从而实现对多个视频源的并行处理。

每个服务器实例从其配置的 MJPEG 视频流中实时捕获图像，并**默认直接在灰度图像上进行匹配** (`filter_type=none`)。用户可以根据需要在 INI 文件中为特定匹配任务选择预处理滤镜（如 Canny 边缘检测）以增强匹配效果，从而在不同场景下平衡匹配速度与准确性。**INI 配置文件必须符合标准INI格式，并且 \`[MatchSettings]\` 部分的配置优先级最高。**

**每个实例的基础 URL:** \`http://<服务器IP>:<实例端口>\` (例如: \`http://192.168.1.3:60003\`, \`http://192.168.1.3:60004\`)

## 2. 核心工作流程

1.  **创建 INI 配置文件:** 定义搜索区域、匹配阈值、坐标偏移等参数。**INI 文件必须包含 \`[MatchSettings]\` 部分，并遵循标准INI格式。\`[MatchSettings]\` 部分中的配置将覆盖服务器默认设置，未定义的配置会被忽略。** 默认情况下，将直接在灰度图上匹配。您可以在 INI 中选择启用 Canny 边缘检测 (\`filter_type=canny\`)并配置其参数，以应对复杂场景或提高特定目标的识别准确率。模板图像将根据 INI 文件名自动派生 (例如, \`my_target.ini\` 的模板为 \`my_target.jpg\`).
2.  **配置多实例设置:** 在 \`multi_config.json\` 文件中定义每个服务器实例要监听的端口和连接的 MJPEG URL。
3.  **使用 Manager 脚本启动服务器实例:** 运行 \`manager.py\` 脚本来启动一个或所有配置的服务器实例。Manager 负责启动独立的子进程来运行每个实例。
4.  **发送 API 请求:** 使用 \`/search_ini\` 端点，并提供目标 INI 文件的绝对路径，请求发送到相应的服务器实例端口。
5.  **获取结果:** API 将返回 JSON 格式的匹配结果，包含是否成功、匹配位置、得分等信息。

## 3. 主要特性

*   **多实例管理:** 通过 \`manager.py\` 脚本集中管理多个独立的服务器实例。
*   **每个实例独立 MJPEG 源:** 每个服务器实例连接并处理一个独立的 MJPEG 视频流。
*   **端口隔离:** 每个实例监听不同的端口，相互隔离。
*   **主端点 \`/search_ini\`:** 通过 INI 文件驱动，简化复杂参数配置。
*   **默认原始匹配:** 服务器默认在灰度图像上进行模板匹配 (\`filter_type=none\`)，追求通用性和效率。
*   **INI 可配置增强滤镜:**
    *   支持在 INI 文件中为每个匹配任务单独指定 \`filter_type\` (\`none\` 或 \`canny\`).
    *   允许在 INI 中配置 Canny 边缘检测的特定参数 (\`canny_t1\`, \`canny_t2\`).
    *   若 INI 中未指定滤镜相关参数，则使用服务器的默认设置。
*   **自动模板派生:** 模板图像路径根据 INI 文件名自动生成 (扩展名替换为 \`.jpg\`)，简化配置。
*   **绝对路径要求:** INI 文件路径和（对于 \`/search\` 端点）模板图像路径必须提供绝对路径。
*   **灵活的参数处理:**
    *   INI 文件中的设置优先于通用服务器配置 (\`config/server.ini\`) 中的默认设置。
    *   空参数值（无论在 INI 中或 URL 查询中）将回退到通用服务器配置的默认值。
    *   未在 API 定义中出现的查询参数将被自动忽略。
*   **辅助端点 \`/search\`:** 提供直接通过 URL 参数进行图像搜索的灵活性，适合动态场景或测试。
*   **内部管理 API:** 每个实例暴露内部 API 端点 (\`/internal/status\`, \`/internal/update_mjpeg_url\`) 供 \`manager.py\` 脚本使用。
*   **MJPEG 流管理:** 支持空闲时自动休眠和请求时自动唤醒 MJPEG 流。
*   **进程管理:** Manager 脚本负责启动、停止和监控子进程。
*   **动态更新 MJPEG URL:** Manager 脚本可以向运行中的实例发送命令，动态更新其连接的 MJPEG URL。
*   **调试图像保存:** 可选保存**原始**MJPEG 帧用于调试目的。如果启用，成功接收到帧时会异步保存图像到指定目录。该目录也可用于保存其他类型的调试图像（例如，带标注的匹配结果帧，如果相关功能被启用）。
*   **缓存机制:**
    *   **模板缓存:** LRU 缓存处理过的模板图像（区分滤镜类型和参数）。存储经过灰度化、滤波等预处理的模板图像，避免重复处理，提高匹配效率。特别是在多次使用同一模板或使用相同滤镜参数时效果显著。
    *   **INI 缓存:** LRU 缓存解析过的 INI 文件参数。存储已解析的 INI 文件内容，避免重复读取和解析文件系统，加快 \`/search_ini\` 请求的处理速度。**注意：** INI 缓存仅存储解析后的参数，不监控 INI 文件本身的变动。如果修改了 INI 文件，需要重启相应的实例或等待缓存过期（如果实现了过期机制）才能加载新配置。

## 4. 快速入门：通过 INI 文件进行图像匹配

### 步骤 1: 创建模板图像和 INI 配置文件

假设您要匹配的图像名为 \`login_button.jpg\`.

1.  **模板图像 (\`login_button.jpg\`):**
    *   确保图像清晰，特征明显，尽量裁剪掉不必要的边缘。
    *   保存路径示例: \`C:\\myapp\\templates\\login_button.jpg\`

2.  **INI 配置文件 (\`login_button.ini\`):**
    *   保存路径示例: \`C:\\myapp\\templates\\login_button.ini\` (与模板图像同名，扩展名不同)

    *   **示例 1 (推荐 - 默认灰度图匹配):**
        ```ini
        [MatchSettings]
        ; template_path 会自动从 login_button.ini 派生为 login_button.jpg
        
        ; filter_type 默认为通用服务器配置 (config/server.ini) 中的 default_filter_type (推荐为 'none')
        ; 或者可以显式指定:
        filter_type = none
        
        ; 匹配阈值至关重要，需要根据实际情况仔细调整 (0.0-1.0)
        threshold = 0.85
        
        ; 可选：定义搜索区域 (不指定则为全屏)
        ; match_range_x1 = 100
        ; match_range_y1 = 200
        ; match_range_x2 = 800
        ; match_range_y2 = 600
        
        ; 可选：匹配算法
        ; match_method = ccoeff_normed
        
        ; 可选：结果坐标偏移
        ; offset_x = 0
        ; offset_y = 0
        ```

    *   **示例 2 (可选增强 - 使用Canny滤镜):**
        *   *当默认的灰度匹配在复杂背景、光照变化或需要更高区分度的场景下效果不佳时，考虑使用 Canny。*
        ```ini
        [MatchSettings]
        filter_type = canny
        canny_t1 = 50  ; Canny 低阈值，根据图像调整
        canny_t2 = 150 ; Canny 高阈值，根据图像调整
        
        threshold = 0.70 ; 使用 Canny 时，阈值可能需要与灰度匹配时不同
        
        ; match_range_x1 = 100
        ; ... 其他参数 ...
        ```

### 步骤 2: 启动服务器实例

首先，确保您的 \`multi_config.json\` 文件中包含了要启动的实例配置。例如：

```json
[
  {
    "port": 60003,
    "mjpeg_url": "http://192.168.1.3:8080/stream.mjpeg"
  }
]
```

然后，使用 \`manager.py\` 脚本启动该实例：

```bash
python manager.py start --port 60003 --server-config config/server.ini
```

您也可以不指定 \`--port\` 来启动 \`multi_config.json\` 中配置的所有实例：

```bash
python manager.py start --server-config config/server.ini
```

### 步骤 3: 发送 API 请求

使用 \`curl\` 或任何 HTTP 客户端，调用 \`/search_ini\` 端点，并提供 \`ini_path\` 参数（确保URL编码），将请求发送到相应实例的端口（例如 60003）。

*   **Windows (路径中的 \`\` 需要编码为 \`%5C\`):**
    ```bash
    curl "http://127.0.0.1:60003/search_ini?ini_path=C%3A%5Cmyapp%5Ctemplates%5Clogin_button.ini"
    ```
*   **Linux/macOS:**
    ```bash
    curl "http://127.0.0.1:60003/search_ini?ini_path=/home/user/myapp/templates/login_button.ini"
    ```

### 步骤 4: 查看结果

如果找到匹配，您将收到类似以下的 JSON 响应 (具体字段见 [响应体详解](#9-响应体详解-成功匹配)):
```json
{
  "found": true,
  "center_x": 450,
  "center_y": 300,
  "template_name": "login_button.jpg",
  "template_path": "C:\\myapp\\templates\\login_button.jpg",
  "score": 0.9234,
  "top_left_x": 482,
  "top_left_y": 335,
  "width": 80,
  "height": 40,
  "top_left_x_with_offset": 482,
  "top_left_y_with_offset": 335,
  "offset_applied_x": 0,
  "offset_applied_y": 0,
  "verify_wait": 0.0,
  "verify_confirmed": false,
  "verify_score": null,
  "recheck_status": "Not performed",
  "recheck_frame_timestamp": null,
  "search_region_x1": 100,
  "search_region_y1": 200,
  "search_region_x2": 800,
  "search_region_y2": 600,
  "search_region_full_search": false,
  "filter_type_used": "none",
  "match_method_used": "ccoeff_normed",
  "frame_timestamp": 1678886420.123,
  "frame_width": 1920,
  "frame_height": 1080,
  "threshold": 0.85,
  "highest_score": 0.9234,
  "error": null
}
```
如果未找到，\`found\` 将为 \`false\`，\`error\` 将包含错误信息，\`highest_score\` 可能包含低于阈值的最高得分，其他坐标字段可能为 null。

## 5. INI 配置文件详解

### 标准 INI 格式与 \`[MatchSettings]\` 部分

当使用 \`/search_ini\` 端点时，服务器要求 INI 文件必须遵循标准的 INI 格式，并且其核心配置必须位于 \`[MatchSettings]\` 部分下。

**服务器只会读取并解析 \`[MatchSettings]\` 部分中已定义的标准配置项。任何在此部分或其他部分中未被服务器明确支持或定义的配置项都将被忽略。**

### \`[MatchSettings]\` 配置项详解与优先级

服务器会解析 \`[MatchSettings]\` 部分下的配置项。**INI 文件中提供的配置项将具有最高优先级，会覆盖通用服务器配置 (\`config/server.ini\`) 中的默认值。** 如果 INI 文件中未提供某个可选参数，则会使用通用服务器配置中设置的相应默认值。

以下是 \`[MatchSettings]\` 部分支持的标准配置项及其说明：

*   \`template_path\` (自动派生):
    *   模板图像的路径。**不需要在 INI 文件中指定。**
    *   服务器会自动将 INI 文件的扩展名替换为 \`.jpg\` 来定位模板图像 (例如, \`config.ini\` -> \`config.jpg\`). 确保模板图像与 INI 文件具有此派生关系（通常在同一目录）。

*   \`filter_type\`** (可选, 字符串):
    *   指定用于此匹配任务的图像预处理滤镜。
    *   可选值:
        *   \`none\`: (**服务器推荐默认值**) 直接在灰度图像上匹配。
        *   \`canny\`: 使用 Canny 边缘检测作为预处理步骤。可能在特定场景下（如光照变化、需要区分细微边缘）提高匹配鲁棒性。
    *   *INI 中未指定时的行为:* 使用通用服务器配置 \`config/server.ini\` 中的 \`default_filter_type\` 值。

*   \`canny_t1\`, \`canny_t2\`** (可选, 整数, >=0):
    *   Canny 边缘检测的低阈值和高阈值。
    *   **重要:** 这些参数仅在 \`filter_type\` (无论来自 INI 还是通用配置) 解析为 \`canny\` 时才生效。需要根据目标图像和背景进行仔细调整以获得最佳边缘效果。
    *   *INI 中未指定时的行为 (且 \`filter_type\` 为 \`canny\`):* 使用通用服务器配置 \`config/server.ini\` 中的 \`default_canny_t1\` 和 \`default_canny_t2\`.

*   \`threshold\`** (可选, 浮点数, 0.0-1.0):
    *   匹配得分的接受阈值。**此参数对匹配准确性至关重要。** 值太低可能导致误匹配，太高可能导致漏匹配。需要根据模板质量、匹配算法和场景复杂度仔细调整。
    *   *INI 中未指定时的行为:* 使用通用服务器配置 \`config/server.ini\` 中的 \`default_threshold\`.

*   \`match_range_x1\`, \`match_range_y1\`, \`match_range_x2\`, \`match_range_y2\`** (可选, 整数):
    *   定义搜索区域的左上角 (x1, y1) 和右下角 (x2, y2) 坐标。
    *   如果未提供、部分提供或值无效，则默认为全屏搜索。

*   \`offset_x\`, \`offset_y\`** (可选, 整数):
    *   添加到返回坐标 (\`center_x\`, \`center_y\`, \`top_left_x\`, \`top_left_y\`) 的偏移量。默认为 \`0\`.
    *   对于 \`center_x_with_offset\` 等字段，此偏移将被应用。

*   \`match_method\`** (可选, 字符串):
    *   指定 OpenCV 模板匹配算法。
    *   可选值:
        *   \`ccoeff_normed\` (TM_CCOEFF_NORMED): 归一化相关系数匹配 (通常效果较好，是服务器的推荐默认值)。
        *   \`sqdiff_normed\` (TM_SQDIFF_NORMED): 归一化平方差匹配 (值越小表示匹配度越高，与其他方法相反，但API内部会处理，使得高分依然表示好匹配)。
        *   \`ccorr_normed\` (TM_CCORR_NORMED): 归一化相关匹配。
    *   *INI 中未指定时的行为:* 使用通用服务器配置 \`config/server.ini\` 中的 \`default_match_method\`.

*   \`waitforrecheck\`** (可选, 浮点数, >=0.0):
    *   对应 \`/search\` 的 \`waitForRecheck\`. 找到初始匹配后等待多少秒重新验证。**请注意 INI 中参数名为 \`waitforrecheck\`.**
    *   \`0.0\` (默认) 禁用此功能。

## 6. API 端点

### 6.1. 公共 API 端点

这些端点由每个运行的服务器实例在其监听的端口上提供。

#### \`/search_ini\` (GET) - 通过 INI 文件执行图像搜索

**核心推荐端点。** 解析指定的 INI 文件，自动派生模板图像路径，并根据 INI 配置（回退到通用服务器配置）执行图像搜索。

*   **URL:** \`/search_ini\`
*   **方法:** \`GET\`
*   **查询参数:**
    *   \`ini_path\` (**必需**, 字符串): INI 配置文件的**绝对路径**。例如: \`C:\\myapp\\configs\\targetA.ini\`。
*   **行为:**
    1.  读取并解析 \`ini_path\` 指定的 INI 文件（结果会被缓存）。**仅读取 \`[MatchSettings]\` 部分下的标准配置。**
    2.  从 INI 文件名派生模板图像路径 (例如, \`targetA.ini\` -> \`targetA.jpg\`).
    3.  **参数优先级: INI 文件中的标准配置项值具有最高优先级，将覆盖服务器启动时的默认参数。** 如果 INI 文件中未指定某个参数，则使用服务器启动时设置的相应默认值。
    4.  执行图像搜索。
    5.  返回 JSON 格式的搜索结果（详见 [响应体详解](#9-响应体详解-成功匹配)）。

#### \`/search\` 或 \`/\` (GET) - 直接参数化图像搜索 (高级/备用)

允许直接通过 URL 查询参数指定模板图像路径和所有搜索参数，无需 INI 文件。INI 文件中的配置优先级更高，如果同时使用 \`/search_ini\` 和 \`/search\`，\`search_ini\` 的 INI 文件会覆盖 \`/search\` 的 URL 参数。

*   **URL:** \`/search\` 或 \`/\`
*   **方法:** \`GET\`
*   **查询参数:**
    *   \`img\` (**必需**, 字符串): 模板图像的**绝对路径**或基于绝对路径的 **Glob 模式** (例如 \`C:\\templates\\icon_*.png\`).
    *   \`filter_type\` (可选, 字符串): 覆盖通用配置中的 \`default_filter_type\` (\`none\` 或 \`canny\`).
    *   \`match_method\` (可选, 字符串): 覆盖通用配置中的 \`default_match_method\` (\`ccoeff_normed\`, \`sqdiff_normed\`, \`ccorr_normed\`).
    *   \`threshold\` (可选, 浮点数): 覆盖通用配置中的 \`default_threshold\` (0.0-1.0).
    *   \`x1\`, \`y1\`, \`x2\`, \`y2\` (可选, 整数): 定义搜索区域。
    *   \`offsetx\`, \`offsety\` (可选, 整数): 添加到结果坐标的偏移量。
    *   \`waitForRecheck\` (可选, 浮点数): 找到初始匹配后等待多少秒重新验证。
    *   \`canny_t1\`, \`canny_t2\` (可选, 整数): 覆盖通用配置中的 Canny 阈值 (仅在 \`filter_type\` 为 \`canny\` 时有效).

*   **响应:** 匹配结果 JSON 对象或错误响应。

#### \`/health\` (GET) - 健康检查

返回服务器实例的健康状态，包括 MJPEG 流是否活跃、帧是否有效、帧龄、分辨率等信息。

*   **URL:** \`/health\`
*   **方法:** \`GET\`
*   **查询参数:** 无
*   **响应:** 健康状态 JSON 对象 (\`HealthResponse\`).

### 6.2. 内部管理 API 端点 (供管理器使用)

这些端点用于 \`manager.py\` 脚本与子进程中的服务器实例进行通信和控制。**这些端点不应直接暴露给外部网络。**

*   \`/internal/status\` (GET): 获取实例的详细内部状态，包括 MJPEG URL、健康数据等。
*   \`/internal/update_mjpeg_url\` (PUT): 动态更新实例连接的 MJPEG URL.

## 7. 多实例管理器 (Manager) 使用

\`manager.py\` 脚本是用于启动、停止、监控和管理多个服务器实例的主要工具。

### 7.1. 架构概述

使用 \`manager.py\` 时，系统采用以下架构：

*   **Manager 进程:** 运行 \`manager.py\` 脚本的主进程。它不运行 FastAPI 应用，而是负责读取配置、启动子进程、监控子进程状态以及通过内部 API 与子进程通信。
*   **Server Instance 子进程:** 每个由 Manager 启动的子进程运行一个独立的 FastAPI 应用实例 (\`main_instance.py\`). 每个实例监听一个指定的端口，连接到一个特定的 MJPEG 流，并维护自己的状态（如帧缓存）。

### 7.2. 依赖安装

除了原有的服务器依赖项 (\`requirements.txt\`):
```bash
pip install -r requirements_manager.txt
```

### 7.3. 配置文件

Manager 使用两个主要配置文件：

#### \`multi_config.json\` (多实例配置)

这是一个 JSON 格式的文件，包含一个列表，列表中的每个对象定义了一个服务器实例的配置（端口和 MJPEG URL）。

*   **默认路径:** \`multi_config.json\`
*   **示例:**
    ```json
    [
      {
        "port": 60003,
        "mjpeg_url": "http://127.0.0.1:8080/video"
      },
      {
        "port": 60004,
        "mjpeg_url": "http://192.168.1.100:8080/video"
      },
      {
        "port": 60005,
        "mjpeg_url": "http://192.168.1.101:8080/video"
      }
    ]
    ```

#### \`config/server.ini\` (通用服务器配置)

这是一个标准的 INI 文件，包含所有服务器实例共享的通用配置。它分为多个部分，用于组织不同类型的配置参数。

*   **默认路径:** `config/server.ini`
*   **主要部分和配置项:**
    *   `[server]`: 服务器核心配置，如监听地址 `host`。
    *   `[debug]`: 调试相关配置，如是否启用调试保存、保存目录和最大文件数。
    *   `[timeout]`: 超时设置，如空闲超时和唤醒超时。
    *   `[MatchSettings]`: **默认的图像匹配参数。** 这个部分提供了在 `/search_ini` 和 `/search` 端点中未由请求或特定 INI 文件覆盖时的默认参数值。它包含 `filter_type`, `match_method`, `threshold`, `canny_t1`, `canny_t2` 等。
    *   `[cache]`: 缓存相关配置，如模板缓存和 INI 缓存的大小。

*   **示例:**
    ```ini
    [server]
    # Server configuration
    host = 127.0.0.1

    [debug]
    enable_debug_saving = false
    debug_save_dir = debug_images
    max_debug_files = 100

    [timeout]
    idle_timeout = 300.0
    wakeup_timeout = 15.0

    [MatchSettings]
    # Default image matching parameters
    filter_type = none
    match_method = ccoeff_normed
    threshold = 0.8
    canny_t1 = 50
    canny_t2 = 150

    [cache]
    template_cache_size = 1000
    ini_cache_size = 100
    ```

### 7.4. Manager 命令行命令

Manager 脚本通过子命令的方式提供不同的功能：

#### \`start\`

启动一个或所有服务器实例。

*   **用法:** \`python manager.py start [options]\`
*   **Options:**
    *   \`--port <port>\`: 启动指定端口的实例。如果该端口的配置存在于 \`multi_config.json\` 中，则使用该配置；如果不存在但同时指定了 \`--mjpeg-url\`, 则启动一个新的实例并将其添加到 \`multi_config.json\` 中。
    *   \`--mjpeg-url <url>\`: 当使用 \`--port\` 启动一个不在 \`multi_config.json\` 中的新实例时，需要指定此参数。也可以用于覆盖 \`multi_config.json\` 中现有实例的 MJPEG URL（仅本次启动有效，不修改配置文件）。
    *   \`--multi-config <path>\`: 指定多实例配置文件路径 (覆盖默认)。
    *   \`--server-config <path>\`: 指定通用服务器配置文件路径 (覆盖默认)。
    *   \`--log-level <level>\`: 指定 Manager 脚本的日志级别。

*   **默认行为:** 如果不指定 \`--port\`, 则启动 \`multi_config.json\` 中配置的所有实例。

#### \`stop\`

停止一个或所有运行中的服务器实例。

*   **用法:** \`python manager.py stop [options]\`
*   **Options:**
    *   \`--port <port>\`: 停止指定端口的实例。
    *   \`--multi-config <path>\`: 指定多实例配置文件路径 (仅影响要停止的实例列表)。
    *   \`--log-level <level>\`: 指定 Manager 脚本的日志级别。

*   **默认行为:** 如果不指定 \`--port\`, 则尝试停止 \`multi_config.json\` 中配置的所有实例。

#### \`status\`

获取一个或所有服务器实例的当前状态。Manager 会尝试连接到实例的 \`/internal/status\` 端点。

*   **用法:** \`python manager.py status [options]\`
*   **Options:**
    *   \`--port <port>\`: 获取指定端口实例的状态。
    *   \`--multi-config <path>\`: 指定多实例配置文件路径 (仅影响要检查的实例列表)。
    *   \`--log-level <level>\`: 指定 Manager 脚本的日志级别。

*   **默认行为:** 如果不指定 \`--port\`, 则获取 \`multi_config.json\` 中配置的所有实例的状态。

#### \`update\`

动态更新一个运行中服务器实例的 MJPEG URL. Manager 会通过内部 API 向目标实例发送更新请求。

*   **用法:** \`python manager.py update --port <port> --new-mjpeg-url <url> [options]\`
*   **Options:**
    *   \`--port <port> (必须)\`: 要更新的实例的端口。
    *   \`--new-mjpeg-url <url> (必须)\`: 新的 MJPEG URL。
    *   \`--multi-config <path>\`: 指定多实例配置文件路径 (用于在更新成功后同步配置)。
    *   \`--log-level <level>\`: 指定 Manager 脚本的日志级别。

*   **注意:** 更新成功后，Manager 会尝试更新 \`multi_config.json\` 文件中的对应配置。

#### \`create\`

向 \`multi_config.json\` 文件中添加一个新的服务器实例配置。

*   **用法:** \`python manager.py create --port <port> --mjpeg-url <url> [options]\`
*   **Options:**
    *   \`--port <port> (必须)\`: 新实例的端口。
    *   \`--mjpeg-url <url> (必须)\`: 新实例的 MJPEG URL。
    *   \`--multi-config <path>\`: 指定多实例配置文件路径 (覆盖默认)。
    *   \`--log-level <level>\`: 指定 Manager 脚本的日志级别。

*   **注意:** 此命令只修改配置文件，不启动实例。

#### \`delete\`

从 \`multi_config.json\` 文件中删除一个服务器实例配置。

*   **用法:** \`python manager.py delete --port <port> [options]\`
*   **Options:**
    *   \`--port <port> (必须)\`: 要删除配置的实例的端口。
    *   \`--multi-config <path>\`: 指定多实例配置文件路径 (覆盖默认)。
    *   \`--log-level <level>\`: 指定 Manager 脚本的日志级别。

*   **注意:** 此命令只修改配置文件，不停止运行中的实例。如果需要停止，请先使用 \`stop\` 命令。

### 7.5. 示例 Manager 命令

*   **启动 \`multi_config.json\` 中配置的所有实例：**
    ```bash
    python manager.py start --server-config config/server.ini
    ```
*   **启动端口 60004 的实例：**
    ```bash
    python manager.py start --port 60004 --server-config config/server.ini
    ```
*   **启动端口 60006 的新实例，并添加到配置中：**
    ```bash
    python manager.py create --port 60006 --mjpeg-url http://new.stream.url/video
    python manager.py start --port 60006 --server-config config/server.ini
    ```
*   **获取所有运行中实例的状态：**
    ```bash
    python manager.py status
    ```
*   **获取端口 60003 实例的状态：**
    ```bash
    python manager.py status --port 60003
    ```
*   **更新端口 60004 实例的 MJPEG URL：**
    ```bash
    python manager.py update --port 60004 --new-mjpeg-url http://updated.stream.url/video
    ```
*   **停止端口 60005 的实例：**
    ```bash
    python manager.py stop --port 60005
    ```
*   **停止所有实例：**
    ```bash
    python manager.py stop
    ```

## 8. 通用 API 说明

*   **错误处理:** API 会返回标准的 HTTP 状态码和 JSON 格式的错误响应 (\`{"detail": "错误信息"}\`).
*   **性能头部:** 成功匹配的响应可能包含 \`X-Processing-Time\` 和 \`X-Frame-Timestamp\` HTTP 头部，用于调试和性能监控。
*   **绝对路径:** 对于文件路径参数 (\`ini_path\`, \`img\`):
    *   必须提供文件的绝对路径。

## 9. 响应体详解 (成功匹配)

```json
{
  "found": true,                     // 是否找到匹配 (布尔值)
  "center_x": 450,                   // 匹配区域中心的 X 坐标 (整数)
  "center_y": 300,                   // 匹配区域中心的 Y 坐标 (整数)
  "template_name": "login_button.jpg", // 使用的模板文件名 (字符串)
  "template_path": "C:\\\\myapp\\\\templates\\\\login_button.jpg", // 使用的模板文件绝对路径 (字符串)
  "score": 0.9234,                   // 匹配得分 (浮点数, 0.0-1.0)
  "top_left_x": 482,                 // 匹配区域左上角的 X 坐标 (整数)
  "top_left_y": 335,                 // 匹配区域左上角的 Y 坐标 (整数)
  "width": 80,                       // 匹配区域的宽度 (整数)
  "height": 40,                      // 匹配区域的高度 (整数)
  "top_left_x_with_offset": 482,     // 考虑 offset_x 后的左上角 X 坐标 (整数)
  "top_left_y_with_offset": 335,     // 考虑 offset_y 后的左上角 Y 坐标 (整数)
  "offset_applied_x": 0,             // 实际应用的 X 偏移量 (整数)
  "offset_applied_y": 0,             // 实际应用的 Y 偏移量 (整数)
  "verify_wait": 0.0,                // 等待重新验证的时间 (浮点数)
  "verify_confirmed": false,         // 重新验证是否成功 (布尔值)
  "verify_score": null,              // 重新验证的匹配得分 (浮点数或 null)
  "recheck_status": "Not performed", // 重新验证的状态 ("Not performed", "Waiting", "Confirmed", "Failed")
  "recheck_frame_timestamp": null,   // 重新验证时帧的时间戳 (浮点数或 null)
  "search_region_x1": 100,           // 实际使用的搜索区域左上角 X 坐标 (整数)
  "search_region_y1": 200,           // 实际使用的搜索区域左上角 Y 坐标 (整数)
  "search_region_x2": 800,           // 实际使用的搜索区域右下角 X 坐标 (整数)
  "search_region_y2": 600,           // 实际使用的搜索区域右下角 Y 坐标 (整数)
  "search_region_full_search": false, // 是否进行了全屏搜索 (布尔值)
  "filter_type_used": "none",        // 实际使用的滤镜类型 (字符串)
  "match_method_used": "ccoeff_normed",// 实际使用的匹配方法 (字符串)
  "frame_timestamp": 1678886420.123, // 匹配时使用的帧的时间戳 (浮点数)
  "frame_width": 1920,               // 匹配时使用的帧的宽度 (整数)
  "frame_height": 1080,              // 匹配时使用的帧的高度 (整数)
  "threshold": 0.85,                 // 实际使用的匹配阈值 (浮点数)
  "highest_score": 0.9234,           // 在搜索区域内找到的最高匹配得分 (即使低于阈值) (浮点数)
  "error": null                      // 如果发生错误，包含错误信息 (字符串或 null)
}
```

## 10. 错误处理

API 在遇到问题时会返回标准的 HTTP 状态码和包含错误信息的 JSON 响应体。例如：

*   **404 Not Found:** 找不到指定的 INI 文件或模板图像。
    ```json
    {"detail": "Template file not found: C:\\path\\to\\nonexistent.jpg"}
    ```
*   **400 Bad Request:** 请求参数无效（例如，INI 文件格式错误，参数值超出范围）。
    ```json
    {"detail": "Value error: Invalid threshold value"}
    ```
*   **503 Service Unavailable:** 服务器无法获取 MJPEG 帧（例如，MJPEG 流断开或初始化失败）。
    ```json
    {"detail": "Failed to get frame from MJPEG stream"}
    ```
*   **500 Internal Server Error:** 服务器内部发生未知错误。
    ```json
    {"detail": "Internal server error: Some unexpected error"}
    ```

## 11. 版本变更

*   **V6.0:**
    *   引入多进程管理器 (\`manager.py\`) 和单实例启动脚本 (\`main_instance.py\`).
    *   支持通过 \`multi_config.json\` 配置多个服务器实例，每个实例独立运行并连接不同的 MJPEG 流。
    *   在服务器实例中添加内部管理 API (\`/internal/status\`, \`/internal/update_mjpeg_url\`).
    *   更新 \`create_app\` 函数，使其不再依赖全局变量，支持创建独立的应用程序实例。
    *   修改 \`readme.md\` 以反映新的架构和管理器使用方法。
    *   添加 \`requirements_manager.txt\` 文件，列出管理器所需的额外依赖。
