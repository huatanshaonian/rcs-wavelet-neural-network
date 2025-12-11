# 重构日志 (Refactoring Changelog)

## 2025-12-11: UI 结构重构 (UI Structure Refactoring)

### 概述
为了解决 `gui.py` 文件过大（"God Class"）的问题，我们将各个功能标签页（Tabs）的 UI 构建和交互逻辑从主类 `RCSWaveletGUI` 中提取出来，放入独立的模块中。

### 目录结构变更
*   **`gui_managers/managers/`**: (保持不变) 存放业务逻辑管理器，如 `TrainingManager`, `VisualizationManager` 等。
*   **`gui_managers/tabs/`**: (新增) 存放 UI 标签页组件。

### 详细变更记录

#### 1. 数据管理标签页 (Data Management Tab)
*   **原位置**: `gui.py` 中的 `create_data_tab` 方法及相关辅助方法。
*   **新位置**: `gui_managers/tabs/data_management_tab.py` 中的 `DataManagementTab` 类。
*   **移动的方法**:
    *   `create_data_tab` (重构为 `__init__`)
    *   `browse_params_file`
    *   `browse_rcs_dir`
    *   `load_data`
    *   `preview_data`
    *   `show_data_stats`
    *   `show_cache_info`
    *   `clear_cache`
    *   `force_reload_data`
    *   `reset_cuda_manually` (系统管理功能暂时保留在此标签页)
    *   `check_cuda_status`
    *   `clean_gpu_memory`
    *   `on_preprocessing_change`
    *   `update_data_config`
*   **依赖关系**: `DataManagementTab` 接收主应用程序实例 (`app`) 作为参数，并通过 `app.data_config`、`app.cache_manager` 等访问共享状态。

### 如何使用
在 `gui.py` 中，不再直接调用构建方法，而是实例化对应的标签页类：

```python
from gui_managers.tabs.data_management_tab import DataManagementTab

# ... 在 create_widgets 中 ...
self.data_tab = DataManagementTab(self.notebook, self)
self.notebook.add(self.data_tab, text="数据管理")
```

#### 2. 模型训练标签页 (Model Training Tab)
*   **原位置**: `gui.py` 中的 `create_training_tab` 方法及相关辅助方法。
*   **新位置**: `gui_managers/tabs/training_tab.py` 中的 `TrainingTab` 类。
*   **移动的方法**:
    *   `create_training_tab` (重构为 `__init__`)
    *   `start_training`
    *   `_train_model`
    *   `_training_finished`
    *   `_set_random_seeds`
    *   `_initialize_cuda_safely`
    *   `stop_training`
    *   `_monitor_training_stop`
    *   `_on_training_stopped`
    *   `_get_scheduler_info`
    *   `_on_scheduler_changed`
    *   `_update_network_options`
    *   `_on_network_selection_changed`
    *   `test_logging`
    *   `save_model`
    *   `set_default_wavelets`
    *   `set_db4_wavelets`
    *   `set_bior_wavelets`
    *   `set_progressive_wavelets`
    *   `set_edge_wavelets`
    *   `get_current_wavelet_config`
    *   `load_model`
*   **依赖关系**: `TrainingTab` 接收主应用程序实例 (`app`) 作为参数，并通过 `app.training_config`, `app.training_manager` 等访问共享状态和管理器。