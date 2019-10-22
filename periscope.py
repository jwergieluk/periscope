from typing import Optional, List
import numpy as np
import pygame
import random
import time
import datetime


FONT_NAME = 'inconsolata'
FONT_SIZE = 15
COLOR_BACKGROUND_0 = pygame.Color(0x1e2320ff)
COLOR_BACKGROUND_1 = pygame.Color(0x333333ff)
COLOR_BACKGROUND_2 = pygame.Color(0x3f3f3fff)
COLOR_FONT_NORMAL = pygame.Color(0xdcdcccff)
COLOR_FONT_HIGHLIGHT = pygame.Color(0xfff393ff)
COLOR_LINE_1 = pygame.Color(0x7fbfffff)
COLOR_BORDER = pygame.Color(0x6f6f6fff)


class Widget:
    def __init__(self):
        self._surface = None
        self.w, self.h = None, None
        self._observers = []
        self.do_update = True

    def register_parent(self, observer):
        self._observers.append(observer)

    def notify_parents(self):
        for observer in self._observers:
            observer.update()

    def update(self):
        self.do_update = True
        self.notify_parents()

    def _update(self):
        pass

    @property
    def surface(self):
        if self.do_update:
            self._update()
        return self._surface


class HStack(Widget):
    def __init__(self, children: List[Widget], border_size=2, color_border=COLOR_BORDER):
        super().__init__()
        assert len(children) > 0
        self._children = children
        for c in self._children:
            c.register_parent(self)
        self._border_size = border_size
        self.h = max(e.h for e in self._children) + self._border_size*2
        self.w = sum(e.w for e in self._children) + self._border_size*(len(self._children) + 1)
        self._surface = pygame.Surface((self.w, self.h))
        self._surface.fill(color_border)

    def _update(self):
        index_w = self._border_size
        for c in self._children:
            self._surface.blit(c.surface, (index_w, self._border_size))
            index_w += c.w + self._border_size
        self.do_update = False


class VStack(Widget):
    def __init__(self, children: List[Widget], border_size=2, color_border=COLOR_BORDER):
        super().__init__()
        assert len(children) > 0
        self._children = children
        for c in self._children:
            c.register_parent(self)
        self._border_size = border_size
        self.w = max(e.w for e in self._children) + self._border_size*2
        self.h = sum(e.h for e in self._children) + self._border_size*(len(self._children) + 1)
        self._surface = pygame.Surface((self.w, self.h))
        self._surface.fill(color_border)

    def _update(self):
        index_h = self._border_size
        for c in self._children:
            self._surface.blit(c.surface, (self._border_size, index_h))
            index_h += c.h + self._border_size
        self.do_update = False


class TextField(Widget):
    def __init__(self, w: int, text: Optional[str] = None, color_font=COLOR_FONT_NORMAL,
                 color_background=COLOR_BACKGROUND_1, align='left', font_name=FONT_NAME, font_size=FONT_SIZE):
        super().__init__()
        assert w > 1
        assert align in ('left', 'right')
        assert font_size > 5
        self.color_font = color_font
        self.color_background = color_background
        self.align = align
        self.padding = 1
        self.font = pygame.font.SysFont(font_name, font_size)
        _, self.h = self.font.size('M')
        self.h += 2*self.padding
        self.w = w
        self._surface = pygame.Surface((self.w, self.h))
        if text is not None:
            self.set_content(text)

    def set_content(self, text: str):
        text_surface = self.font.render(text, True, self.color_font, self.color_background)
        self._surface.fill(self.color_background)
        if self.align == 'right':
            self._surface.blit(
                text_surface, (self._surface.get_width() - text_surface.get_width() - 2, self.padding))
        if self.align == 'left':
            self._surface.blit(text_surface, (2, self.padding))
        self.notify_parents()


class LinePlot(Widget):
    def __init__(self, w: int, h: int, color_background=COLOR_BACKGROUND_1):
        super().__init__()
        assert w > 1 and h > 1
        self.w, self.h = w, h
        self.color_background = color_background
        self._surface = pygame.Surface((w, h))
        self._surface.fill(self.color_background)
        self.y_min, self.y_max = None, None
        self.beta = 0.15
        self.epsilon = 0.0001
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE-2)

    @staticmethod
    def exp_avg(x0, x1, beta):
        if x0 is None:
            return x1
        return x0 * (1.0 - beta) + beta * x1

    def set_content(self, x, y, y_range=None):
        assert len(x) == len(y)
        self._surface.fill(self.color_background)

        x_min, x_max = min(x), max(x)
        if y_range is None:
            y_min, y_max = min(y)-self.epsilon, max(y)+self.epsilon
            dy = abs(y_max - y_min)
            y_min -= 0.15*dy
            y_max += 0.15*dy

            self.y_min = self.exp_avg(self.y_min, y_min, self.beta)
            self.y_max = self.exp_avg(self.y_max, y_max, self.beta)
        else:
            self.y_min, self.y_max = y_range

        scale_x = self.w / (x_max - x_min)
        x = (x - x_min)*scale_x
        y = (1.0-(y-self.y_min)/(self.y_max - self.y_min)) * self.h

        points = list(zip(x.astype(int).tolist(), y.astype(int).tolist()))
        pygame.draw.aalines(self._surface, COLOR_LINE_1, False, points)
        text_y_min = self.font.render(f'{self.y_min:.2f}', True, COLOR_FONT_NORMAL, self.color_background)
        text_y_max = self.font.render(f'{self.y_max:.2f}', True, COLOR_FONT_NORMAL, self.color_background)
        self._surface.blit(text_y_max, (0, 0))
        self._surface.blit(text_y_min, (0, self._surface.get_height()-text_y_max.get_height()))
        self.notify_parents()


def get_screen_surface(w: int, h: int) -> pygame.Surface:
    assert w > 1 and h > 1
    pygame.init()
    pygame.mixer.quit()
    surface = pygame.display.set_mode((w, h))
    return surface


class RandomPath:
    def __init__(self):
        self.dx = 0.01
        self.x = np.arange(0.0, 1.0, self.dx)
        self.y = np.random.randn(len(self.x)) * self.dx
        self.y = np.cumsum(self.y)

    def next(self):
        self.x += self.dx
        self.y = np.roll(self.y, -1)
        self.y[-1] = self.y[-2] + random.normalvariate(0.0, self.dx)


def random_around(x: float):
    if abs(x) < 1e-8:
        return random.normalvariate(0.0, 0.1)
    return x*random.lognormvariate(0.0, 0.1)


def demo0():
    pygame.init()
    text_field_0 = TextField(250, 'Text field with width 250')
    pygame.image.save(text_field_0.surface, 'text-field-0.png')

    x = np.linspace(0.0, 2*np.pi, 100)
    line_plot_0 = LinePlot(120, 80)
    line_plot_0.set_content(x, np.sin(x))

    pygame.image.save(line_plot_0.surface, 'line-plot-0.png')

    text_field_1 = TextField(90, 'TextField1')
    text_field_2 = TextField(90, 'TextField2')
    text_field_3 = TextField(90, 'TextField3')

    h_stack = HStack([text_field_1, text_field_2, text_field_3])
    v_stack = VStack([text_field_1, text_field_2, text_field_3])

    text_field_2.set_content('Field2')
    pygame.image.save(h_stack.surface, 'h-stack-0.png')
    pygame.image.save(v_stack.surface, 'v-stack-0.png')


def demo1():
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600

    screen = get_screen_surface(SCREEN_WIDTH, SCREEN_HEIGHT)
    screen.fill(COLOR_BACKGROUND_0)

    width_unit = int(screen.get_width()/16)
    height_unit = int(screen.get_height()/16)
    panel_width = 5*width_unit

    bp_margin_rate = TextField(2*width_unit, '0.1', align='right')
    bp_commission = TextField(2*width_unit, '0.0', align='right')
    bp_financing = TextField(2*width_unit, '0.0', align='right')
    bp_nav = TextField(2*width_unit, '0.0', align='right')
    bp_initial_nav = TextField(2*width_unit, '100000.0', align='right')
    bp_pl = TextField(2*width_unit, '0.0', align='right')
    bp_position_value = TextField(2*width_unit, '0.0', align='right')
    bp_margin_used = TextField(2*width_unit, '0.0', align='right')
    bp_margin_available = TextField(2*width_unit, '0.0', align='right')
    bp_margin_call_percent = TextField(width_unit, '0.0', align='right')
    bp_act_ord = TextField(2*width_unit, '0.0', align='right')
    bp_time_index = TextField(2*width_unit, '0.0', align='right')
    bp_default_numeraire = TextField(width_unit, 'EUR', align='right')
    bp_now = TextField(4*width_unit, '0.0', align='right')

    broker_panel = VStack(
        [TextField(panel_width, 'Broker variables',
                   color_font=COLOR_FONT_HIGHLIGHT, color_background=COLOR_BACKGROUND_2),
         HStack([TextField(3 * width_unit, 'margin rate'), bp_margin_rate], border_size=0),
         HStack([TextField(3 * width_unit, 'commission'), bp_commission], border_size=0),
         HStack([TextField(3 * width_unit, 'Financing'), bp_financing], border_size=0),
         HStack([TextField(3 * width_unit, 'NAV'), bp_nav], border_size=0),
         HStack([TextField(3 * width_unit, 'Initial NAV'), bp_initial_nav], border_size=0),
         HStack([TextField(3 * width_unit, 'P&L'), bp_pl], border_size=0),
         HStack([TextField(3 * width_unit, 'Position value'), bp_position_value], border_size=0),
         HStack([TextField(3 * width_unit, 'Margin used'), bp_margin_used], border_size=0),
         HStack([TextField(3 * width_unit, 'Margin available'), bp_margin_available], border_size=0),
         HStack([TextField(4 * width_unit, 'Margin call percent'), bp_margin_call_percent], border_size=0),
         HStack([TextField(3 * width_unit, '_act_ord'), bp_act_ord], border_size=0),
         HStack([TextField(3 * width_unit, 'Time index'), bp_time_index], border_size=0),
         HStack([TextField(4 * width_unit, 'Default numeraire'), bp_default_numeraire], border_size=0),
         HStack([TextField(1 * width_unit, 'Now'), bp_now], border_size=0),
         ], border_size=1
    )

    ap_price_field = TextField(2*width_unit, '0.0', align='right')
    ap_price_plot = LinePlot(panel_width, 5*height_unit)
    ap_spread_field = TextField(2*width_unit, '0.0', align='right')
    ap_spread_plot = LinePlot(panel_width, 3*height_unit)

    asset_panel = VStack(
        [TextField(panel_width, 'Asset series',
                   color_font=pygame.Color(0xfff393ff), color_background=COLOR_BACKGROUND_2),
         HStack([TextField(3 * width_unit, 'Price'), ap_price_field], border_size=0),
         ap_price_plot,
         HStack([TextField(3 * width_unit, 'Rel. spread'), ap_spread_field], border_size=0),
         ap_spread_plot
         ], border_size=1)

    signal_plot = LinePlot(panel_width, 2*height_unit)
    signal_field_text = TextField(3*width_unit, 'Signal', color_font=pygame.Color(0xfff393ff),
                                  color_background=COLOR_BACKGROUND_2)
    signal_field_value = TextField(2*width_unit, '0.0', color_font=pygame.Color(0xfff393ff),
                                   color_background=COLOR_BACKGROUND_2, align='right')
    signal_panel = VStack(
        [
            HStack([signal_field_text, signal_field_value], border_size=0),
            signal_plot
        ], border_size=1
    )

    screen_container = HStack([asset_panel, broker_panel, signal_panel],
                              border_size=10, color_border=COLOR_BACKGROUND_0)

    series_price_01_mid = RandomPath()
    series_spread_rel = RandomPath()
    signal = RandomPath()

    quit_game = False
    for c in range(1, 10000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True

        series_price_01_mid.next()
        series_spread_rel.next()
        signal.next()

        ap_price_plot.set_content(series_price_01_mid.x, series_price_01_mid.y)
        ap_price_field.set_content(format(float(series_price_01_mid.y[-1]), '.4f'))
        ap_spread_plot.set_content(series_spread_rel.x, series_spread_rel.y, y_range=(-0.5, 0.5))
        ap_spread_field.set_content(format(float(series_spread_rel.y[-1]), '.4f'))

        bp_commission.set_content(f'{random_around(100.0):.1f}')
        bp_financing.set_content(f'{random_around(10.0):.1f}')
        bp_nav.set_content(f'{random_around(100000.0):.0f}')
        bp_pl.set_content(f'{random_around(1000.0):.1f}')
        bp_position_value.set_content(f'{random_around(200000.0):.0f}')
        bp_margin_used.set_content(f'{random_around(10.0):.2f}')
        bp_margin_available.set_content(f'{random_around(90.0):.2f}')
        bp_margin_call_percent.set_content(f'{random_around(10.0):.2f}')
        bp_time_index.set_content(f'{c}')
        bp_now.set_content(f'{datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%S}')

        signal_plot.set_content(signal.x, signal.y, y_range=(-1.0, 1.0))
        signal_field_value.set_content(f'{signal.y[-1]:.2f}')

        screen.blit(screen_container.surface, (0, 0))

        pygame.display.flip()

        pygame.image.save(ap_price_plot.surface, f'/tmp/plot_{c:04d}.png')
        if quit_game:
            break
        time.sleep(0.05)


if __name__ == '__main__':
    demo1()
    pygame.quit()
