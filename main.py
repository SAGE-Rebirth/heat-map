import sys
import logging
from PyQt6.QtWidgets import QApplication
from gui import MainWindow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("Main")

def main():
    logger.info("Initializing HeatMap Application...")
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    logger.info("Application loop started.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
