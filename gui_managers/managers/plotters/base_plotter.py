
class BasePlotter:
    """所有绘图器的基类"""

    def __init__(self, parent_gui):
        """
        初始化绘图器
        
        Args:
            parent_gui: 主GUI实例
        """
        self.gui = parent_gui

    def get_target_fig_canvas(self, fig=None, canvas=None):
        """获取绘图目标 (Figure和Canvas)"""
        target_fig = fig if fig is not None else getattr(self.gui, 'vis_fig', None)
        target_canvas = canvas if canvas is not None else getattr(self.gui, 'vis_canvas', None)
        return target_fig, target_canvas

    def get_fontsize_scale(self):
        """获取字号缩放因子"""
        try:
            # 尝试从不同位置获取fontsize_scale_var
            if hasattr(self.gui, 'visualization_tab') and hasattr(self.gui.visualization_tab, 'fontsize_scale_var'):
                fontsize_scale = self.gui.visualization_tab.fontsize_scale_var.get()
            elif hasattr(self.gui, 'fontsize_scale_var'):
                fontsize_scale = self.gui.fontsize_scale_var.get()
            else:
                fontsize_scale = 1.0
            
            # 限制范围在0.5-3.0之间
            return max(0.5, min(3.0, fontsize_scale))
        except:
            return 1.0

    def setup_chinese_font(self):
        """配置中文字体"""
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def log(self, message):
        """记录日志"""
        if hasattr(self.gui, 'log_message'):
            self.gui.log_message(message)
        else:
            print(message)

    def handle_error(self, title, error):
        """统一错误处理"""
        from tkinter import messagebox
        error_msg = f"{title}: {str(error)}"
        self.log(error_msg)
        # 尝试显示弹窗，如果在主线程
        try:
            messagebox.showerror("绘图错误", error_msg)
        except:
            pass
