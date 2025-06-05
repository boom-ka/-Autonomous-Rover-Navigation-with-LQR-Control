from collections import deque
import time
import mujoco
import numpy as np
import pathlib
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QThread
from PySide6.QtGui import (
    QGuiApplication, QSurfaceFormat, QKeyEvent
)
import time

from robot_lqr import RobotLqr


format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2,0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


class Viewport(QOpenGLWindow):

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()

        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.scale = 1.0
        self.__last_pos = None

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60*1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def setScreenScale(self, scaleFactor: float) -> None:
        """ Sets a scale factor that is used to scale the OpenGL window to accommodate
        the high DPI scaling Qt does.
        """
        self.scale = scaleFactor

    def paintGL(self) -> None:
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.running = True

        self.robot = RobotLqr(model, data)

        # robot control parameters
        self.speed = 0.0
        self.yaw = 0.0

        # Define speed and yaw increments
        self.speed_increment = 0.5  # Speed change per key press
        self.max_speed = 4.0        # Maximum speed magnitude
        self.yaw_increment = 0.2    # Reduced steering change per key press
        self.max_yaw = 1.0          # Maximum steering magnitude

        # Reset the simulation timer
        self.reset()

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self) -> None:
        while self.running:
            # don't step the simulation past real time
            # without this the sim usually finishes before it's
            # even visible
            if self.data.time < self.real_time / 1_000_000_000:
                # In the real robot we update the control loop at a 200hz, so do that
                # here too. It's the filters applied to pitch_dot and linear speed error
                # that are not time step independent
                if (time.monotonic_ns() - self.last_robot_update) / 1_000_000_000 >= (1/200):
                    self.last_robot_update = time.monotonic_ns()
                    # update robot with user inputs
                    self.robot.set_velocity_linear_set_point(self.speed)
                    self.robot.set_yaw(self.yaw)
                    # update motor speed with LQR controller
                    self.robot.update_motor_speed()

                # step the simulation
                mujoco.mj_step(self.model, self.data)
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.robot.reset()
        self.speed = 0.0
        self.yaw = 0.0

    def set_speed(self, speed: float) -> None:
        # Clamp speed to maximum values
        self.speed = max(min(speed, self.max_speed), -self.max_speed)

    def set_yaw(self, yaw: float) -> None:
        # Clamp yaw to maximum values
        self.yaw = max(min(yaw, self.max_yaw), -self.max_yaw)

    def increment_speed(self, amount: float) -> float:
        """Increase or decrease the speed and return the new value"""
        new_speed = self.speed + amount
        self.set_speed(new_speed)
        return self.speed
        
    def increment_yaw(self, amount: float) -> float:
        """Increase or decrease the yaw and return the new value"""
        new_yaw = self.yaw + amount
        self.set_yaw(new_yaw)
        return self.yaw


class Window(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath('scene.xml')))
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        # Add a key state dictionary to track continuous key presses
        self.key_states = {
            Qt.Key_Up: False,
            Qt.Key_Down: False,
            Qt.Key_Left: False,
            Qt.Key_Right: False
        }

        layout = QVBoxLayout()
        layout_top = QHBoxLayout()
        layout_top.setSpacing(8)
        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(90)
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        reset_button.clicked.connect(self.reset_simulation)
        layout_top.addWidget(reset_button)
        layout_robot_controls = QVBoxLayout()
        layout_robot_controls.setContentsMargins(0,0,0,0)
        layout_robot_controls.addWidget(self.create_top())
        
        # Add keyboard instructions label
        keyboard_instructions = QLabel("Use arrow keys to control the rover:\nUp/Down: Speed | Left/Right: Steering")
        keyboard_instructions.setAlignment(Qt.AlignCenter)
        layout_robot_controls.addWidget(keyboard_instructions)
        
        layout_top.addLayout(layout_robot_controls)
        layout_top.setContentsMargins(8,0,8,0)
        layout.addLayout(layout_top)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.setContentsMargins(0,4,0,0)
        layout.setStretch(1,1)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.resize(800, 600)

        self.th = UpdateSimThread(self.model, self.data, self)
        self.th.start()
        
        # Set focus to main window to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Set up a timer for key processing to handle continuous key presses
        self.key_timer = QTimer()
        self.key_timer.setInterval(50)  # 50ms interval for smooth control
        self.key_timer.timeout.connect(self.process_keys)
        self.key_timer.start()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard events for rover control"""
        key = event.key()
        if key in self.key_states:
            self.key_states[key] = True
        elif key == Qt.Key_Space:
            # Stop the rover completely
            self.th.set_speed(0.0)
            self.th.set_yaw(0.0)
            self.speed_slider.setValue(0)
            self.yaw_slider.setValue(0)
            # Reset all key states
            for k in self.key_states:
                self.key_states[k] = False
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key release events"""
        key = event.key()
        if key in self.key_states:
            self.key_states[key] = False
            
            # When steering keys are released, gradually return steering to center
            if key == Qt.Key_Left or key == Qt.Key_Right:
                # Only reset steering if both left and right are not pressed
                if not (self.key_states[Qt.Key_Left] or self.key_states[Qt.Key_Right]):
                    # Set yaw to decay towards zero but not immediately
                    current_yaw = self.th.yaw
                    if abs(current_yaw) > 0.1:
                        # Reduce by half
                        new_yaw = current_yaw * 0.5
                        self.th.set_yaw(new_yaw)
                        self.yaw_slider.setValue(int(new_yaw * 1000))
        else:
            super().keyReleaseEvent(event)
    
    def process_keys(self):
        """Process currently pressed keys to allow continuous control"""
        # Process speed keys (vertical)
        if self.key_states[Qt.Key_Up] and not self.key_states[Qt.Key_Down]:
            # Increase forward speed
            new_speed = self.th.increment_speed(self.th.speed_increment * 0.1)
            self.speed_slider.setValue(int(new_speed * 1000))
        elif self.key_states[Qt.Key_Down] and not self.key_states[Qt.Key_Up]:
            # Increase backward speed
            new_speed = self.th.increment_speed(-self.th.speed_increment * 0.1)
            self.speed_slider.setValue(int(new_speed * 1000))
        
        # Process steering keys (horizontal)
        if self.key_states[Qt.Key_Left] and not self.key_states[Qt.Key_Right]:
            # Turn left (negative yaw)
            new_yaw = self.th.increment_yaw(-self.th.yaw_increment * 0.1)
            self.yaw_slider.setValue(int(new_yaw * 1000))
        elif self.key_states[Qt.Key_Right] and not self.key_states[Qt.Key_Left]:
            # Turn right (positive yaw)
            new_yaw = self.th.increment_yaw(self.th.yaw_increment * 0.1)
            self.yaw_slider.setValue(int(new_yaw * 1000))

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {self.data.time:.0f}s"
        )

    def create_top(self):
        layout = QVBoxLayout()
        # layout.setContentsMargins(0,0,0,0)
        label_width = 60

        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        # QSliders only support ints, so scale the values we want by 1000
        # and then remove this scale factor in the valueChanged handler
        self.speed_slider.setMinimum(-4 * 1000)
        self.speed_slider.setMaximum(4 * 1000)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self._speed_changed)
        speed_label = QLabel("Speed")
        speed_label.setFixedWidth(label_width)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)

        yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-1 * 1000)  # Reduced range for more natural car steering
        self.yaw_slider.setMaximum(1 * 1000)
        self.yaw_slider.setValue(0)
        self.yaw_slider.valueChanged.connect(self._yaw_changed)
        yaw_label = QLabel("Steering")  # Changed from "Yaw" to "Steering" to better reflect car-like behavior
        yaw_label.setFixedWidth(label_width)
        yaw_layout.addWidget(yaw_label)
        yaw_layout.addWidget(self.yaw_slider)

        layout.addLayout(speed_layout)
        layout.addLayout(yaw_layout)

        w = QGroupBox("Rover Control")  # Changed from "Robot Control" to "Rover Control"
        w.setLayout(layout)
        return w

    def _speed_changed(self, value: int) -> None:
        speed = value / 1000
        self.th.set_speed(speed)

    def _yaw_changed(self, value: int) -> None:
        yaw = value / 1000
        self.th.set_yaw(yaw)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])
        cam.distance = self.model.stat.extent * 2
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        self.speed_slider.setValue(0)
        self.yaw_slider.setValue(0)
        # Reset state and time.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.th.reset()
        
        # Reset all key states
        for key in self.key_states:
            self.key_states[key] = False


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()
    w.th.stop()