import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Tkinter Button Example")
root.geometry("300x200")

# Create a label widget
label = tk.Label(root, text="Press the button", font=("Helvetica", 14))
label.pack(pady=20)

# Function to be called when button is clicked
def on_button_click():
    label.config(text="Button Clicked!")

# Create a button widget
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()