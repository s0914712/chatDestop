import wx
import os
import sys
import random
import itertools
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import math
import numpy as np
from scipy.stats import norm
import io  # 添加這行
import base64
import uuid
from PHIllpine_png import img as PHIllpine

tmp = open('Taiwan.png', 'wb')
tmp.write(base64.b64decode(PHIllpine))
tmp.close()

load_dotenv()
MAP_FILE = 'Taiwan.png'

GRID_ROWS = 4
GRID_COLS = 6
GRID_START_X = 100
GRID_START_Y = 300
GRID_CELL_WIDTH = 50
GRID_CELL_HEIGHT = 50


SEARCH_AREAS = []
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        x1 = GRID_START_X + col * GRID_CELL_WIDTH
        y1 = GRID_START_Y + row * GRID_CELL_HEIGHT
        x2 = x1 + GRID_CELL_WIDTH
        y2 = y1 + GRID_CELL_HEIGHT
        SEARCH_AREAS.append((x1, y1, x2, y2))


class Unit:
    def __init__(self, unit_type, x, y, color):
        self.id = str(uuid.uuid4())
        self.type = unit_type
        self.x = int(x)
        self.y = int(y)
        self.initial_x = int(x)
        self.initial_y = int(y)
        self.color = color

    def move(self, dx, dy):
        self.x = int(self.x + dx)
        self.y = int(self.y + dy)
    def get_move_area(self):
        move_area = []
        for dx in range(-self.move_range, self.move_range + 1, 10):
            for dy in range(-self.move_range, self.move_range + 1, 10):
                if abs(dx) + abs(dy) <= self.move_range:
                    move_area.append((self.x + dx, self.y + dy))
        return move_area
    def toggle_select(self):
        self.selected = not self.selected
class InputDialog(wx.Dialog):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(250, 200))

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.time_input = wx.TextCtrl(panel, size=(30, 30))
        self.speed_input = wx.TextCtrl(panel, size=(30, 30))

        vbox.Add(wx.StaticText(panel, label="Disappearance Time:"), flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.time_input, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        vbox.Add(wx.StaticText(panel, label="Max Speed:"), flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.speed_input, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        ok_button = wx.Button(panel, wx.ID_OK, "OK")
        vbox.Add(ok_button, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=10)

        panel.SetSizer(vbox)
class Search():
    def __init__(self, name):
        self.name = name
        self.img = wx.Image(MAP_FILE, wx.BITMAP_TYPE_PNG)
        if not self.img.IsOk():
            print(f'Could not load map file {MAP_FILE}', file=sys.stderr)
            sys.exit(1)
        self.img_bitmap = wx.Bitmap(self.img)
        self.probabilities = self.initialize_probabilities()
        self.area_actual = 0
        self.sailor_actual = [0, 0]
        self.search_effectiveness = [0 for _ in SEARCH_AREAS]
        self.base_effectiveness = [0.6 + (col * 0.05) for row in range(GRID_ROWS) for col in range(GRID_COLS)]
        self.search_effectiveness = self.base_effectiveness.copy()
    def initialize_probabilities(self):
        # 计算列的中心值
        center = GRID_COLS - 1

        # 创建一个正态分布
        x = np.linspace(0, GRID_COLS - 1, GRID_COLS)
        y = norm.pdf(x, loc=center, scale=1)

        # 为每个搜索区域分配概率
        probabilities = []
        for row in range(GRID_ROWS):
            probabilities.extend(y)

        # 归一化概率
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        return probabilities.tolist()
    def update_effectiveness_after_search(self):
        """每次搜索后更新搜索效率"""
        for i, area in enumerate(SEARCH_AREAS):
            center_x = (area[0] + area[2]) / 2
            center_y = (area[1] + area[3]) / 2
            distance = math.sqrt((center_x - self.sailor_actual[0])**2 + (center_y - self.sailor_actual[1])**2)
            distance_factor = 1 / max(distance, 1)
            self.search_effectiveness[i] = min(self.search_effectiveness[i] + distance_factor * 0.1, 1.0)

    def draw_map(self, dc, last_known):
        """使用 wxPython 在 DC 上绘制地图和搜索区域"""
        # 设置透明背景
        dc.SetBackground(wx.TRANSPARENT_BRUSH)
        dc.Clear()

        # 绘制比例尺
        dc.SetPen(wx.Pen(wx.BLACK, 2))
        # dc.DrawLine(20, 370, 90, 370)
        # dc.DrawText('0', 8, 370)
        # dc.DrawText('50 Nautical Miles', 71, 370)

        # 绘制搜索区域
        for i, area in enumerate(SEARCH_AREAS):
            color = self.get_color_for_probability(self.probabilities[i])
            dc.SetBrush(wx.Brush(color))
            dc.SetPen(wx.Pen(wx.BLACK, 1))
            dc.DrawRectangle(area[0], area[1], area[2] - area[0], area[3] - area[1])


            text_color = wx.BLACK if self.probabilities[i] < 0.5 else wx.WHITE
            dc.SetTextForeground(text_color)
            dc.DrawText(f'{i + 1} ({self.probabilities[i]:.2f})', area[0] + 3, area[1] + 3)

        # 位置
        actual_x, actual_y = self.sailor_actual
        print(f"draw_map: Drawing actual position at {actual_x}, {actual_y}")
        dc.SetTextForeground(wx.BLUE)
        dc.DrawText('*', int(actual_x), int(actual_y))
        dc.DrawText(f'* = Actual Position ({actual_x:.2f}, {actual_y:.2f})', 10, 10)


        dc.SetTextForeground(wx.RED)
        dc.DrawText('+', last_known[0], last_known[1])
        dc.DrawText('+ = Last Known Position', last_known[0], last_known[1])

        dc.SetTextForeground(wx.BLUE)
        dc.DrawText('* = Actual Position', self.sailor_actual[0], self.sailor_actual[1])

    def get_color_for_probability(self, prob):
        """根据概率返回对应深浅的蓝色，背景为70%透明"""
        # 将概率映射到 0-255 的范围
        color_value = int(255 * (1-2*prob))

        background_alpha = int(255 * 0.3)  # 30% 不透明度 = 70% 透明度

        foreground_alpha = int(255 * 0.7)  # 70% 不透明度

        mixed_red = int(255 * 0.7 + color_value * 0.3)
        mixed_green = int(255 * 0.7 + color_value * 0.3)
        mixed_blue = int(255 * 0.7 + 255 * 0.3)

        return wx.Colour(mixed_red, mixed_green, mixed_blue, background_alpha + foreground_alpha)

    def sailor_final_location(self, num_search_areas):
        self.area_actual = random.randint(0, len(SEARCH_AREAS) - 1)
        print(self.area_actual)
        area = SEARCH_AREAS[self.area_actual]
        self.sailor_actual = [
            random.randint(area[0], area[2]),
            random.randint(area[1], area[3])
        ]
        print('self.area_actual are',area,'area',self.sailor_actual)
        self.initial_x = self.sailor_actual[0]  # 保存初始 x 坐标
        return self.sailor_actual

    def update_sailor_position(self, search_num):
        new_x = max(self.initial_x - (search_num * 10), GRID_START_X)  # 确保不会移出地图
        self.sailor_actual[0] = new_x

        # 检查新位置落在哪个搜索区域内
        for i, area in enumerate(SEARCH_AREAS):
            if (area[0] <= self.sailor_actual[0] < area[2] and
                    area[1] <= self.sailor_actual[1] < area[3]):
                self.area_actual = i
                break

        print(f"search_num: {search_num}")
        print(f"sailor_actual: x={self.sailor_actual[0]}, y={self.sailor_actual[1]}")
        print(f"new area_actual: {self.area_actual + 1}")  # +1 是为了使区域编号从1开始
        print(f"EARCH_AREAS location: {SEARCH_AREAS[self.area_actual]}")
        print("------------------------")
    def calc_search_effectiveness(self):
        """设置各搜索区域的搜索效率"""
        for i, area in enumerate(SEARCH_AREAS):
            # 计算区域中心点
            center_x = (area[0] + area[2]) / 2
            center_y = (area[1] + area[3]) / 2

            # 计算与水手的距离
            distance = math.sqrt((center_x - self.sailor_actual[0]) ** 2 + (center_y - self.sailor_actual[1]) ** 2)

            # 更新搜索效率
            distance_factor = 1 / max(distance, 1)  # 避免除以零
            self.search_effectiveness[i] = min(self.base_effectiveness[i] + distance_factor, 1.0)

    def conduct_search(self, area_num, area_array, effectiveness_prob):

        local_y_range = range(area_array[1], area_array[3])
        print(local_y_range)
        local_x_range = range(area_array[0], area_array[2])
        print(local_x_range)
        coords = list(itertools.product(local_x_range, local_y_range))
        print(coords)
        random.shuffle(coords)
        coords = coords[:int((len(coords) * effectiveness_prob))]
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        print(loc_actual)
        if area_num == self.area_actual and loc_actual in coords:
            return f'Found in Area {area_num}.', coords
        return 'Not Found', coords

    def revise_target_probs(self):
        """依搜索效率更新目标概率"""
        for i in range(len(SEARCH_AREAS)):
            if self.search_effectiveness[i] > 0:
                self.probabilities[i] *= (1 - self.search_effectiveness[i])

        # 归一化概率
        total = sum(self.probabilities)
        self.probabilities = [p / total for p in self.probabilities]

class UnitMovementPanel(wx.Panel):
    def __init__(self, parent, on_move):
        super().__init__(parent)
        self.on_move = on_move

        # Create controls
        self.direction_choice = wx.Choice(self, choices=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        self.distance_input = wx.SpinCtrl(self, min=1, max=100, initial=1)
        self.move_button = wx.Button(self, label="Move Unit")

        # Bind events
        self.move_button.Bind(wx.EVT_BUTTON, self.on_move_click)

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Select Direction:"), 0, wx.ALL, 5)
        sizer.Add(self.direction_choice, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="Enter Distance:"), 0, wx.ALL, 5)
        sizer.Add(self.distance_input, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.move_button, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)

    def on_move_click(self, event):
        direction = self.direction_choice.GetStringSelection()
        distance = self.distance_input.GetValue()

        if direction and distance:
            self.on_move({
                'direction': direction,
                'distance': distance
            })


def create_circle_bitmap(color, size):
    bitmap = wx.Bitmap(size, size)
    dc = wx.MemoryDC(bitmap)
    dc.SetBackground(wx.Brush(wx.WHITE))
    dc.Clear()
    dc.SetBrush(wx.Brush(color))
    dc.SetPen(wx.Pen(color))
    dc.DrawCircle(size // 2, size // 2, size // 2 - 1)
    dc.SelectObject(wx.NullBitmap)
    return bitmap
class DrawingPanel(wx.Panel):
    def __init__(self, parent, image_path, size):
        super().__init__(parent, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.image = wx.Image(image_path, wx.BITMAP_TYPE_PNG)
        self.image = self.image.Scale(size[0], size[1], wx.IMAGE_QUALITY_HIGH)
        self.bitmap = wx.Bitmap(self.image)
        self.units = []
        self.current_tool = None
        self.circles = []
        self.last_known_position = (500, 600)  # 初始化 last_known_position
        self.current_tool = None
        self.search_app = Search('PHIllpine_Search')
        self.sailor_x, self.sailor_y = self.search_app.sailor_final_location(num_search_areas=len(SEARCH_AREAS))
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.show_search_areas = False
        self.line_start = None
        self.line_end = None
        self.lines = []  # Store multiple lines
        self.current_line = None
        self.temp_line = None  # For preview while dragging
        self.ruler_start = None
        self.ruler_end = None
        self.selected_unit = None
        self.move_area = []
        self.dragging_unit = None
        self.drag_start = None
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)

    def is_valid_position(self, unit, new_x, new_y):
        # 計算新位置與初始位置的距離
        dx = new_x - unit.initial_x
        dy = new_y - unit.initial_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 檢查距離是否在80像素範圍內
        return distance <= 80

    def OnMouseMove(self, event):
        x, y = event.GetPosition()

        if event.Dragging() and event.LeftIsDown():
            if self.current_tool == 'select' and self.dragging_unit:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                new_x = int(self.dragging_unit.x + dx)
                new_y = int(self.dragging_unit.y + dy)
                if self.is_valid_position(self.dragging_unit, new_x, new_y):
                    self.dragging_unit.x = new_x
                    self.dragging_unit.y = new_y
                    self.drag_start = (x, y)
                else:
                    # 如果新位置無效，將單位移動到80像素範圍的邊界
                    angle = math.atan2(new_y - self.dragging_unit.initial_y, new_x - self.dragging_unit.initial_x)
                    self.dragging_unit.x = int(self.dragging_unit.initial_x + 80 * math.cos(angle))
                    self.dragging_unit.y = int(self.dragging_unit.initial_y + 80 * math.sin(angle))
                    self.drag_start = (self.dragging_unit.x, self.dragging_unit.y)
            elif self.current_tool == 'ruler' and self.ruler_start:
                self.ruler_end = (x, y)
            elif self.current_tool == 'line' and self.line_start:
                self.line_end = (x, y)

            self.Refresh()

        event.Skip
    def OnMouseDown(self, event):
        x, y = event.GetPosition()
        print(f"Mouse down at ({x}, {y}). Current tool: {self.current_tool}")

        if self.current_tool == 'select':
            for unit in self.units:
                if self.is_point_in_unit(x, y, unit):
                    self.dragging_unit = unit
                    self.drag_start = (x, y)
                    break
        elif self.current_tool == 'ruler':
            self.ruler_start = (x, y)
            self.ruler_end = None
        elif self.current_tool == 'line':
            self.line_start = (x, y)
            self.line_end = None
        elif self.current_tool in ['pla_navy', 'pla_plane', 'twn', 'twn_plane']:
            self.add_unit(self.current_tool, x, y)
        elif self.current_tool == 'small search':
            self.circles.append({'x': x, 'y': y, 'radius': 35, 'color': wx.YELLOW})
        elif self.current_tool == 'large search':
            self.circles.append({'x': x, 'y': y, 'radius': 50, 'color': wx.YELLOW})

        self.Refresh()
        event.Skip()

    def OnMouseUp(self, event):
        x, y = event.GetPosition()

        if self.current_tool == 'select':
            self.dragging_unit = None
            self.drag_start = None
        elif self.current_tool == 'ruler' and self.ruler_start:
            self.ruler_end = (x, y)
        elif self.current_tool == 'line' and self.line_start:
            self.line_end = (x, y)
            self.lines.append((self.line_start[0], self.line_start[1], x, y))
            self.line_start = None

        self.Refresh()
        event.Skip()

    def set_tool(self, tool_type):
        print(f"Tool set to: {tool_type}")  # 调试输出
        self.current_tool = tool_type
    def SaveAnnotatedImage(self, filename):
        save_bitmap = wx.Bitmap(self.Size.Width, self.Size.Height)
        mdc = wx.MemoryDC(save_bitmap)
        mdc.Clear()
        mdc.DrawBitmap(self.bitmap, 0, 0)
        for circle in self.circles:
            mdc.SetBrush(wx.Brush(circle['color'], wx.BRUSHSTYLE_TRANSPARENT))
            mdc.SetPen(wx.Pen(circle['color'], 2))
            mdc.DrawCircle(circle['x'], circle['y'], circle['radius'])
        mdc.SelectObject(wx.NullBitmap)
        image = save_bitmap.ConvertToImage()
        image.SaveFile(filename, wx.BITMAP_TYPE_PNG)
    def undo(self):
        if self.lines:
            self.lines.pop()
            self.Refresh()
        elif self.circles:  # Existing UNDO functionality
            self.circles.pop()
            self.Refresh()
    def SetShowSearchAreas(self, show):
        self.show_search_areas = show
        self.Refresh()

    def get_color_for_unit(self, unit_type):
        if unit_type == 'pla_navy':
            return wx.RED
        elif unit_type == 'pla_plane':
            return wx.Colour(255, 150, 150)
        elif unit_type == 'twn':
            return wx.BLUE
        elif unit_type == 'twn_plane':
            return wx.Colour(150, 150, 255)

    def is_point_in_unit(self, x, y, unit):
        return ((x - unit.x) ** 2 + (y - unit.y) ** 2) <= 25  # 5^2 = 25

    def add_unit(self, unit_type, x, y):
        color = self.get_color_for_unit(unit_type)
        new_unit = Unit(unit_type, x, y, color)
        self.units.append(new_unit)

        self.Refresh()

    def select_unit(self, x, y):
        for unit in self.units:
            if self.is_point_in_unit(x, y, unit):
                self.selected_unit = unit
                self.move_area = unit.get_move_area()
                break
        else:
            self.selected_unit = None
            self.move_area = []

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def move_unit(self, x, y):
        if self.selected_unit:
            self.selected_unit.x = x
            self.selected_unit.y = y
            self.selected_unit = None
            self.move_area = []
    def get_selected_unit(self):
        for unit in self.units:
            if unit.selected:
                return unit
        return None

    def move_selected_unit(self, dx, dy):
        unit = self.get_selected_unit()
        if unit:
            unit.move(dx, dy)
            self.Refresh()
    def reset_tool(self):
        self.current_tool = None
        wx.PostEvent(self.GetParent(), wx.PyCommandEvent(wx.EVT_BUTTON.typeId, self.GetId()))

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()
        dc.DrawBitmap(self.bitmap, 0, 0)

        # 绘制单位
        for unit in self.units:
            dc.SetBrush(wx.Brush(unit.color, wx.BRUSHSTYLE_TRANSPARENT))
            dc.SetPen(wx.Pen(unit.color, 2))
            dc.DrawCircle(int(unit.x), int(unit.y), 5)
            if unit == self.dragging_unit:
                dc.SetPen(wx.Pen(wx.WHITE, 1, wx.PENSTYLE_DOT))
                dc.DrawCircle(int(unit.x), int(unit.y), 7)
                # 繪製移動範圍
                dc.SetPen(wx.Pen(wx.CYAN, 1, wx.PENSTYLE_DOT))
                dc.DrawCircle(int(unit.initial_x), int(unit.initial_y), 80)

        # 绘制移动范围
        if self.move_area:
            dc.SetBrush(wx.Brush(wx.CYAN, wx.BRUSHSTYLE_TRANSPARENT))
            dc.SetPen(wx.Pen(wx.CYAN, 1, wx.PENSTYLE_DOT))
            for x, y in self.move_area:
                dc.DrawRectangle(x - 5, y - 5, 10, 10)

        # 绘制线条
        dc.SetPen(wx.Pen(wx.RED, 2))
        for line in self.lines:
            dc.DrawLine(*line)

        # 绘制标尺
        if self.ruler_start and self.ruler_end:
            dc.SetPen(wx.Pen(wx.RED, 2, wx.DOT))
            dc.DrawLine(self.ruler_start, self.ruler_end)
            distance = self.calculate_distance(self.ruler_start, self.ruler_end)
            midpoint = ((self.ruler_start[0] + self.ruler_end[0]) // 2,
                        (self.ruler_start[1] + self.ruler_end[1]) // 2)
            dc.DrawText(f"{distance:.2f} ''", midpoint[0], midpoint[1])

        # 绘制圆圈
        for circle in self.circles:
            if all(key in circle for key in ['x', 'y', 'radius', 'color']):
                dc.SetBrush(wx.Brush(circle['color'], wx.BRUSHSTYLE_TRANSPARENT))
                dc.SetPen(wx.Pen(circle['color'], 2))
                dc.DrawCircle(circle['x'], circle['y'], circle['radius'])
            else:
                print(f"Invalid circle data: {circle}")

        event.Skip()
    def calculate_distance(self, start, end):
        return (((start[0] - end[0])**2 + (start[1] - end[1])**2)**0.5-5)/10

    def reset_line(self):
        self.line_start = None
        self.line_end = None
        self.Refresh()
    def reset_ruler(self):
        self.ruler_start = None
        self.ruler_end = None
        self.Refresh()

    def draw_line(self, dc, line, is_temp=False, is_ruler=False):
        if is_ruler:
            dc.SetPen(wx.Pen(wx.RED, 2, wx.DOT))
        elif is_temp:
            dc.SetPen(wx.Pen(wx.RED, 2, wx.DOT))
        else:
            dc.SetPen(wx.Pen(wx.RED, 2, wx.SOLID))  # Use solid line for permanent lines

        dc.DrawLine(line['start'], line['end'])

        # Calculate and display distance
        distance = self.calculate_distance(line['start'], line['end'])
        midpoint = ((line['start'][0] + line['end'][0]) // 2,
                    (line['start'][1] + line['end'][1]) // 2)
        dc.DrawText(f"{distance:.2f} ''", midpoint[0], midpoint[1])
class GeminiFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='War at Sea Tool(Made by Chen(Taiwan))')
        display = wx.Display(0)
        self.panel = wx.Panel(self)
        screen_size = display.GetGeometry()
        screen_width, screen_height = screen_size.GetWidth(), screen_size.GetHeight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        image_width = int(window_width * 0.6)  # 假设图像占窗口宽度的60%
        image_height = int(window_height * 0.9)  # 假设图像占窗口高度的90%


        #self.movement_panel = UnitMovementPanel(self.panel, self.OnUnitMove)
        # Get screen size and set window size
        display = wx.Display(0)

        self.SetSize((window_width, window_height))



    # 调用父类的初始化方法，设置动态大小

    # 计算图像的新尺寸


        self.search_num = 0

        self.num_search_areas = len(SEARCH_AREAS)
        # Set up the image path

        if getattr(sys, 'frozen', False):
            self.image_path = os.path.join(sys._MEIPASS, 'images', 'Taiwan.png')
        else:
            self.image_path = 'images/Taiwan.png'

        self.status_bar = self.CreateStatusBar()

        # Create toolbar
        self.create_toolbar()

        # Create main layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left layout (map and drawing panel)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.drawing_panel = DrawingPanel(self.panel, self.image_path, size=(image_width, image_height))
        self.drawing_panel.Bind(wx.EVT_BUTTON, self.OnResetTool)

        left_sizer.Add(self.drawing_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Right layout (all controls)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add four player input boxes\
        #player_labels = ["Japan CDR's Intent", "Predict US Intent", "Japan COG", "Japan CV"]
        #self.player_inputs = []

        #for label in player_labels:
            #input_sizer = wx.BoxSizer(wx.VERTICAL)
            #static_text = wx.StaticText(self.panel, label=label)
            #static_text.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            #input_sizer.Add(static_text, 0, wx.BOTTOM, 2)

            #text_ctrl = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE, size=(-1, 30))
            #input_sizer.Add(text_ctrl, 0, wx.EXPAND)

            #right_sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 5)
            #self.player_inputs.append(text_ctrl)
        #self.movement_panel = UnitMovementPanel(self.panel, self.OnUnitMove)
        #right_sizer.Add(self.movement_panel, 0, wx.EXPAND | wx.ALL, 5)
        # Gemini 交互部分
        self.gemini_button = wx.Button(self.panel, label='Talk to find Risk')
        self.gemini_button.Bind(wx.EVT_BUTTON, self.interact_with_gemini)
        right_sizer.Add(self.gemini_button, 0, wx.EXPAND | wx.ALL, 5)

        self.gemini_response = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 200))
        right_sizer.Add(self.gemini_response, 1, wx.EXPAND | wx.ALL, 5)

        # 顯示搜索計劃的切換按鈕
        self.toggle_button = wx.ToggleButton(self.panel, label="Display search plan")
        self.toggle_button.SetValue(True)
        self.toggle_button.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleSearchAreas)
        right_sizer.Add(self.toggle_button, 0, wx.EXPAND | wx.ALL, 5)

        #self.input_button = wx.Button(self.panel, label='Set Sailor Parameters')
        #self.input_button.Bind(wx.EVT_BUTTON, self.OnOpenInputDialog)
        #right_sizer.Add(self.input_button, 0, wx.EXPAND | wx.ALL, 5)
        # 貝葉斯搜索部分
        bayes_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.bayes_input = wx.TextCtrl(self.panel)
        bayes_sizer.Add(self.bayes_input, 1, wx.EXPAND | wx.RIGHT, 5)
        self.bayes_button = wx.Button(self.panel, label='Bayes Search')
        self.bayes_button.Bind(wx.EVT_BUTTON, self.run_bayes_search)
        bayes_sizer.Add(self.bayes_button, 0)
        right_sizer.Add(bayes_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.bayes_output = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        right_sizer.Add(self.bayes_output, 1, wx.EXPAND | wx.ALL, 5)

        self.set_last_known_button = wx.Button(self.panel, label='Set Last Known Position')
        self.set_last_known_button.Bind(wx.EVT_BUTTON, self.OnSetLastKnown)
        right_sizer.Add(self.set_last_known_button, 0, wx.EXPAND | wx.ALL, 5)
        # 將左右布局添加到內容布局
        content_sizer.Add(left_sizer, 2, wx.EXPAND)
        content_sizer.Add(right_sizer, 1, wx.EXPAND)

        main_sizer.Add(content_sizer, 1, wx.EXPAND)

        # 添加額外的 Gemini 對話框和按鈕
        chat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.chat_input = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE, size=(-1, 60))
        chat_sizer.Add(self.chat_input, 1, wx.EXPAND | wx.RIGHT, 5)
        self.chat_button = wx.Button(self.panel, label='Free Chat with Gemini')
        self.chat_button.Bind(wx.EVT_BUTTON, self.chat_with_gemini)
        chat_sizer.Add(self.chat_button, 0)
        main_sizer.Add(chat_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.current_player = 'PLA'
        self.turn_count = 1

        self.turn_info = wx.StaticText(self.panel, label=f"Turn {self.turn_count}: {self.current_player}'s turn")
        main_sizer.Add(self.turn_info, 0, wx.EXPAND | wx.ALL, 5)

        self.end_turn_button = wx.Button(self.panel, label="End Turn")
        self.end_turn_button.Bind(wx.EVT_BUTTON, self.OnEndTurn)
        main_sizer.Add(self.end_turn_button, 0, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(main_sizer)
        self.Centre()
        self.panel.Layout()
        # 初始化 Gemini 模型
        self.init_model()

    def OnOpenInputDialog(self, event):
        dialog = InputDialog(self, "Set Sailor Parameters")
        if dialog.ShowModal() == wx.ID_OK:
            try:
                time_str = dialog.time_input.GetValue().strip()
                speed_str = dialog.speed_input.GetValue().strip()

                if not time_str or not speed_str:
                    raise ValueError("Both fields must be filled.")

                time = int(time_str)
                max_speed = int(speed_str)

                if time <= 0 or max_speed <= 0:
                    raise ValueError("Time and speed must be positive numbers.")

                self.UpdateSailorPosition(time, max_speed)
            except ValueError as e:
                wx.MessageBox(str(e), "Invalid Input", wx.OK | wx.ICON_ERROR)
        dialog.Destroy()

    def GenerateRandomSpeed(self, max_speed):
        # Generate a random value between 0 and max_speed as the mean for the normal distribution
        mean = random.uniform(0, max_speed)
        # Set the standard deviation to 1/3 of the mean, so most values will fall between 0 and max_speed
        std_dev = mean / 3

        # Generate a speed using normal distribution, ensuring it's between 0 and max_speed
        while True:
            speed = np.random.normal(mean, std_dev)
            if 0 <= speed <= max_speed:
                return speed

    def OnUnitMove(self, move_data):
        pass

    def calculate_movement(self, direction, distance):
        angle_map = {'N': 90, 'NE': 45, 'E': 0, 'SE': 315, 'S': 270, 'SW': 225, 'W': 180, 'NW': 135}
        angle = math.radians(angle_map[direction])
        dx = distance * math.cos(angle)
        dy = -distance * math.sin(angle)  # Note: y-axis increases downwards
        return dx, dy

    def OnEndTurn(self, event):
        if self.current_player == 'PLA':
            self.current_player = 'TAIWAN'
        else:
            self.current_player = 'PLA'
            self.turn_count += 1

        self.turn_info.SetLabel(f"Turn {self.turn_count}: {self.current_player}'s turn")
        self.Refresh()
    def UpdateSailorPosition(self, time, max_speed):
        # Use the new method to generate a random speed
        speed = self.GenerateRandomSpeed(max_speed)

        # Calculate total movement distance
        distance = time * speed
        angle = random.uniform(220, 350)  # Random angle between 220 and 350 degrees

        # Convert angle to radians
        angle_rad = math.radians(angle)

        # Calculate new x and y coordinates
        dx = distance * math.cos(angle_rad)
        dy = distance * math.sin(angle_rad)

        # Get current position
        current_x, current_y = self.drawing_panel.search_app.sailor_actual

        # Calculate new position
        new_x = current_x + dx
        new_y = current_y + dy

        # Find the boundaries of all search areas
        min_x = min(area[0] for area in SEARCH_AREAS)
        max_x = max(area[2] for area in SEARCH_AREAS)
        min_y = min(area[1] for area in SEARCH_AREAS)
        max_y = max(area[3] for area in SEARCH_AREAS)

        # Ensure the new position is within the search areas
        new_x = max(min_x, min(new_x, max_x))
        new_y = max(min_y, min(new_y, max_y))

        # Update sailor's position
        self.drawing_panel.search_app.sailor_actual = [new_x, new_y]

        # Update the area_actual based on the new position
        for i, area in enumerate(SEARCH_AREAS):
            if (area[0] <= new_x < area[2] and area[1] <= new_y < area[3]):
                self.drawing_panel.search_app.area_actual = i
                break

        # Refresh the drawing panel to show the updated position
        self.drawing_panel.Refresh()

        # Update sailor's position
        self.drawing_panel.search_app.sailor_actual = [new_x, new_y]

        print(f"UpdateSailorPosition: new position set to {self.drawing_panel.search_app.sailor_actual}")

        # Output debug information
        print(f"Time: {time}, Max Speed: {max_speed}, Actual Speed: {speed:.2f}")
        print(f"Distance Moved: {distance:.2f}, Angle: {angle:.2f}")
        print(f"New Position: ({new_x:.2f}, {new_y:.2f})")
        print(f"Current Area: {self.drawing_panel.search_app.area_actual + 1}")

    def OnSetLastKnown(self, event):
        self.drawing_panel.set_tool('last_known')
        self.set_last_known_button.SetBackgroundColour(wx.YELLOW)
        self.status_bar.SetStatusText("Click on the map to set the Last Known Position")

    def OnResetTool(self, event):
        self.set_last_known_button.SetBackgroundColour(wx.NullColour)
        self.status_bar.SetStatusText("")

    def OnTool(self, event, tool_type):
        print(f"Setting tool to: {tool_type}")  # 调试输出
        self.drawing_panel.set_tool(tool_type)
        self.status_bar.SetStatusText(f"Current tool: {tool_type}")
    def chat_with_gemini(self, event):
        try:
            user_input = self.chat_input.GetValue()

            self.drawing_panel.SaveAnnotatedImage("temp_chat_map.png")

            if not os.path.exists("temp_chat_map.png"):
                raise FileNotFoundError("The annotated image was not saved successfully.")

            with open("temp_chat_map.png", "rb") as image_file:
                image_data = image_file.read()

            image = Image.open(io.BytesIO(image_data))

            prompt = f"""This is a naval battle simulation game. The image shows the current state of the map:
            - Red circles represent PLA warships
            - Light red circles represent PLA  planes
            - Blue circles represent TAIWAN warships
            - Light blue circles represent TAIWAN planes
            - Yellow circles represent TAIWAN search ranges
            -you play the Player "PLA" must reponse for blue tean action
            -3 days ago a accident happen between Taiwan and PLA plane ,the collided and crushed,The rescue mission is undergoing at sounthewest of Taiwan
            -You(PLA) is consider the action for blockade and  landing for unify Taiwan, 
            User question: {user_input}

            Please answer the question based on the current map state and the game context.From your perspective(PLA)
            """

            response = self.model.generate_content([prompt, image])

            self.gemini_response.AppendText(f"\nYou: {user_input}\nGemini: {response.text}\n")

            self.chat_input.Clear()
            os.remove("temp_chat_map.png")

        except Exception as e:
            self.gemini_response.AppendText(f"\nError in Gemini chat: {str(e)}\n")
            if os.path.exists("temp_chat_map.png"):
                os.remove("temp_chat_map.png")

    def add_labeled_tool(self, id, label, bitmap, short_help):
        tool = self.toolbar.AddTool(id, label, bitmap, wx.NullBitmap, kind=wx.ITEM_NORMAL, shortHelp=short_help)
        self.toolbar.AddControl(wx.StaticText(self.toolbar, label=label))
        return tool
    def OnToggleSearchAreas(self, event):
        show_areas = self.toggle_button.GetValue()
        self.drawing_panel.SetShowSearchAreas(show_areas)
        self.drawing_panel.Refresh()
    def init_model(self, model_name="models/gemini-1.5-flash"):
        GOOGLE_API_KEY = 'AIzaSyCnFFq2pzPnWLwatyydhncNRmOJnB5nbsk'
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)

    def create_line_bitmap(self):
        bitmap = wx.Bitmap(24, 24)
        dc = wx.MemoryDC(bitmap)
        dc.SetBackground(wx.Brush(wx.WHITE))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.BLACK, 2))
        dc.DrawLine(2, 22, 22, 2)
        dc.SelectObject(wx.NullBitmap)
        return bitmap

    def create_ruler_bitmap(self):
        bitmap = wx.Bitmap(24, 24)
        dc = wx.MemoryDC(bitmap)
        dc.SetBackground(wx.Brush(wx.WHITE))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.BLACK, 1))
        for i in range(5):
            dc.DrawLine(i*6, 0, i*6, 24)
            dc.SetPen(wx.Pen(wx.BLACK, 2))
            dc.DrawLine(0, 12, 24, 12)
            dc.SelectObject(wx.NullBitmap)
        return bitmap
    def create_toolbar(self):
        self.toolbar = self.CreateToolBar()
        self.toolbar.SetToolBitmapSize((24, 24))
        # Define tool buttons
        tools = [
            ("PLA_NAVY", wx.RED, "Place PLA_NAVY Ship"),
            ("PLA_Plane", wx.Colour(255, 150, 150), "Place PLA_Plane"),
            ("TWN", wx.BLUE, "Place TAIWAN Ship"),
            ("TWN_Plane", wx.Colour(150, 150, 255), "Place TWN_Plane"),
            ("Small search", wx.YELLOW, "Draw Small Yellow Circle", 8),
            ("Large search", wx.YELLOW, "Draw Large Yellow Circle", 36),
            ("Ruler", None, "Measure Distance", 24),
            ("Line", None, "Draw Permanent Line", 24),
            ("Undo", None, "Undo Last Action", 24),
            ("Select", None, "Select and Move Unit"),
        ]
        for tool in tools:
            label, color, short_help = tool[:3]
            size = tool[3] if len(tool) > 3 else 24

            if label == "Ruler":
                bitmap = self.create_ruler_bitmap()
            elif label == "Line":
                bitmap = self.create_line_bitmap()
            elif color is not None:
                bitmap = create_circle_bitmap(color, size)
            else:
                # Special handling for Undo button
                bitmap = wx.ArtProvider.GetBitmap(wx.ART_UNDO, size=(size, size))

            tool_item = self.add_labeled_tool(wx.ID_ANY, label, bitmap, short_help)

            # Bind events
            if label == "Undo":
                self.Bind(wx.EVT_TOOL, self.OnUndo, id=tool_item.GetId())
            else:
                self.Bind(wx.EVT_TOOL, lambda event, t=label.lower(): self.OnTool(event, t), id=tool_item.GetId())

        # Add Set Sailor Parameters button to toolbar
        set_sailor_tool = self.toolbar.AddTool(wx.ID_ANY, "Set Sailor Parameters",
                                               wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, size=(24, 24)),
                                               shortHelp="Set Sailor Parameters")
        self.Bind(wx.EVT_TOOL, self.OnOpenInputDialog, id=set_sailor_tool.GetId())

        self.toolbar.Realize()

    def OnTool(self, event, tool_type):
        self.drawing_panel.set_tool(tool_type)
        if tool_type != 'ruler' and tool_type != 'line':
            self.drawing_panel.reset_ruler()
        self.set_last_known_button.SetBackgroundColour(wx.NullColour)
        self.status_bar.SetStatusText("")


    def OnUndo(self, event):
        self.drawing_panel.undo()

    def interact_with_gemini(self, event):
        try:
            self.drawing_panel.SaveAnnotatedImage("temp_annotated.png")
            if not os.path.exists("temp_annotated.png"):
                raise FileNotFoundError("The annotated image was not saved successfully.")

            with open("temp_annotated.png", "rb") as image_file:
                image_data = image_file.read()

            image = Image.open(io.BytesIO(image_data))

            # 構建包含所有玩家輸入的提示
            user_inputs = [input.GetValue() for input in self.player_inputs]
            user_prompt = f"""圖片顯示中共正對台灣地區實施軍事演習，紅色框起來的部分是 中國宣布想要的禁航區
            - Red circles represent 中國PLA 海軍的位置
            - Light red circles represent 中國空軍的位置
            - Blue circles represent 台灣海軍或是美國海軍的位置
            - Light blue circles represent 台灣空軍的位置
            - Yellow circles represent 台灣想要進行的搜索區域

            PLA CDR's Intent: {user_inputs[0]}
            Predict PLA  Intent: {user_inputs[1]}
            Taiwan COG: {user_inputs[2]}
            Taiwan CV: {user_inputs[3]}

            Based on the above information and the attached map image, analyze the risk from four aspects:
            1. Does this operation have sufficient support? Which unit is at the highest risk?
            2. Considering the enemy's intent, will their route intersect with your reconnaissance route?
            3. Consider the possibilities of flanking and being cut off from the rear.
            4. Any other factors you can think of.
            """

            response = self.model.generate_content([user_prompt, image], stream=True)

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text

            self.gemini_response.SetValue(full_response)
            os.remove("temp_annotated.png")
        except Exception as e:
            error_message = f"Gemini interact error happened: {str(e)}\n"
            if hasattr(e, 'response'):
                error_message += f"Response details: {e.response}\n"
            self.gemini_response.SetValue(error_message)

    def run_bayes_search(self, event):
        choice = self.bayes_input.GetValue()
        output = ""

        if choice == "0":
            self.Close()
        elif choice.isdigit() and 1 <= int(choice) <= self.num_search_areas:
            self.search_num += 1  # 增加搜索次数
            search_area = int(choice) - 1

            # 更新水手位置
            self.drawing_panel.search_app.update_sailor_position(self.search_num)

            self.drawing_panel.search_app.calc_search_effectiveness()

            effectiveness = self.drawing_panel.search_app.search_effectiveness[search_area]

            results = []
            coords = []
            for _ in range(2):  # 进行两次搜索
                result, coord = self.drawing_panel.search_app.conduct_search(
                    search_area,
                    SEARCH_AREAS[search_area],
                    effectiveness
                )
                results.append(result)
                coords.extend(coord)

            # 更新搜索效率
            area_size = (SEARCH_AREAS[search_area][2] - SEARCH_AREAS[search_area][0]) * (
                        SEARCH_AREAS[search_area][3] - SEARCH_AREAS[search_area][1])
            new_effectiveness = len(set(coords)) / area_size
            self.drawing_panel.search_app.search_effectiveness[search_area] = new_effectiveness

            self.drawing_panel.search_app.revise_target_probs()

            # 输出搜索结果
            output += f"Search {self.search_num} Results 1 = {results[0]}\n"
            output += f"Search {self.search_num} Results 2 = {results[1]}\n"
            #output += f"Search {self.search_num} Effectiveness (E):\n"
            #output += ', '.join([f"E{i + 1} = {e:.3f}" for i, e in
                                 #enumerate(self.drawing_panel.search_app.search_effectiveness)]) + '\n'

            if all(result == 'Not Found' for result in results):
                pass
                #output += f"\nNew Target Probabilities (P) for Search {self.search_num + 1}:\n"
                #output += ', '.join(
                    #[f"P{i + 1} = {p:.3f}" for i, p in enumerate(self.drawing_panel.search_app.probabilities)]) + '\n'
            else:
                dc = wx.ClientDC(self.drawing_panel)
                dc.SetPen(wx.Pen(wx.BLUE, 2))
                dc.DrawCircle(self.drawing_panel.sailor_x, self.drawing_panel.sailor_y, 3)
                output += "FOUND。\n"

            self.search_num += 1
        elif choice == "7":
            self.__init__()
            output = "RESTART。\n"
        else:
            output = "INVALID INPUT\n"

        self.bayes_output.SetValue(output)
        self.drawing_panel.Refresh()
if __name__ == '__main__':
    app = wx.App()
    frame = GeminiFrame()
    frame.Show()
    app.MainLoop()