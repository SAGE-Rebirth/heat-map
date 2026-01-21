import sys
from PyQt6.QtWidgets import QApplication
from gui import MainWindow
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    logger.info("Initializing HeatMap Application...")
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    logger.info("Application loop started.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
