class PauseAnimation:
    def __init__(self, fig, ani):
        self.fig = fig
        self.ani = ani
        self.paused = False
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.ani.resume()
        else:
            self.ani.pause()
        self.paused = not self.paused