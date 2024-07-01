import pytesseract
from PIL import ImageGrab
import pyperclip
import tkinter as tk
from tkinter import messagebox
import ctypes
import sys

# Assurez-vous de pointer vers l'emplacement de votre exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'

class ScreenCapture:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-transparent", "white")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        self.root.attributes("-alpha", 0.3)
        self.canvas = tk.Canvas(self.root, cursor="cross", bg='grey11')
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        self.root.destroy()
        self.capture_and_extract_text(self.start_x, self.start_y, end_x, end_y)

    def capture_and_extract_text(self, x1, y1, x2, y2):
        bbox = (x1, y1, x2, y2)
        try:
            image = ImageGrab.grab(bbox)
            # Utilisation de la langue française et d'options de configuration pour améliorer la reconnaissance
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang='fra', config=custom_config)
            pyperclip.copy(text)
            messagebox.showinfo("Texte extrait", "Le texte a été extrait et copié dans le presse-papier.")
        except PermissionError as e:
            messagebox.showerror("Erreur", f"Erreur de permission : {e}")

def main():
    # Vérifier si le script est exécuté en tant qu'administrateur
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False

    if is_admin:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Info", "Cliquez et faites glisser pour sélectionner la zone.")
        root.deiconify()
        app = ScreenCapture(root)
        root.mainloop()
    else:
        # Re-lancer le script avec les privilèges d'administrateur
        messagebox.showinfo("Permission", "L'application doit être exécutée en tant qu'administrateur.")
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

if __name__ == "__main__":
    main()
