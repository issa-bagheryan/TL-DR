from tkinter import filedialog
import tkinter as tk


def select_pdf():
    root = tk.Tk()
    root.withdraw()

    pdf_path = filedialog.askopenfilename(
        filetypes=[
            ("PDF files", "*.pdf")
        ]
    )

    return pdf_path