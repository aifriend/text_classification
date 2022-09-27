from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep


class Loader:
    class BC:
        OK = '\033[92m'  # GREEN
        NOTE = '\033[94m'  # BLUE
        WARNING = '\033[93m'  # YELLOW
        FAIL = '\033[91m'  # RED
        RESET = '\033[0m'  # RESET COLOR
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def __init__(self, desc="Loading...", end="Done!", timeout=0.2):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        # self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        # self.steps = ["|", "/", "-", "\\", "|", "/", "-", "\\"]
        self.steps = ["   ", "   ",
                      ".  ", ".  ",
                      ".. ", ".. ",
                      "...", "...",
                      ]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.BC.NOTE}{self.desc}{c}{self.BC.RESET}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.BC.OK}{self.end}{self.BC.RESET}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


if __name__ == "__main__":
    with Loader("Loading with context manager..."):
        for i in range(10):
            sleep(0.25)

    loader = Loader("Loading with object...", "That was fast!", 0.05).start()
    for i in range(10):
        sleep(0.5)
    loader.stop()
