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

#### 3. 可视化标签页 (Visualization Tab)
*   **原位置**: `gui.py` 中的 `create_visualization_tab` 方法及相关辅助方法。
*   **新位置**: `gui_managers/tabs/visualization_tab.py` 中的 `VisualizationTab` 类。
*   **移动的方法**:
    *   `create_visualization_tab` (重构为 `__init__`)
    *   `generate_visualization`
    *   `_plot_2d_heatmap`
    *   `_plot_3d_surface`
    *   `_plot_spherical`
    *   `_plot_comparison`
    *   `_plot_difference_analysis`
    *   `_plot_correlation_analysis`
    *   `_plot_training_history`
    *   `_save_fold_plot`
    *   `_display_fold_in_gui`
    *   `_display_simple_training_history`
    *   `_plot_global_statistics_comparison`
    *   `_plot_autoencoder_visualization`
    *   `_plot_ae_latent_space`
    *   `_plot_ae_reconstruction_quality`
    *   `_plot_ae_parameter_mapping`
    *   `_plot_ae_training_progress_vis`
    *   `_plot_autoencoder_prediction_visualization`
    *   `_plot_ae_2d_heatmap`
    *   `_plot_original_rcs_fallback`
    *   `_plot_ae_comparison`
    *   `_plot_attention_weights`
    *   `_plot_wavelet_coefficients_comparison`
    *   `save_current_visualization`
*   **依赖关系**: `VisualizationTab` 接收主应用程序实例 (`app`) 作为参数，并通过 `app.visualization_manager`, `app.statistics_manager` 等访问共享状态和管理器。它还直接管理 Matplotlib `Figure` 和 `Canvas`。

#### 4. 损失配置标签页 (Loss Config Tab)
*   **新位置**: `gui_managers/tabs/loss_config_tab.py` 中的 `LossConfigTab` 类。
*   **移动的方法**: `create_loss_config_tab`, `init_loss_config_vars`, `load_*_preset`, `update_loss_config_preview`, `apply_loss_config`, `reset_loss_config`。

#### 5. 评估标签页 (Evaluation Tab)
*   **新位置**: `gui_managers/tabs/evaluation_tab.py` 中的 `EvaluationTab` 类。
*   **移动的方法**: `create_evaluation_tab`, `start_evaluation`, `_evaluate_*_model`, `generate_report`, `export_results`, `save_detailed_evaluation_report`。

#### 6. 预测标签页 (Prediction Tab)
*   **新位置**: `gui_managers/tabs/prediction_tab.py` 中的 `PredictionTab` 类。
*   **移动的方法**: `create_prediction_tab`, `load_param_template`, `generate_random_params`, `make_prediction`, `_plot_prediction_results`。