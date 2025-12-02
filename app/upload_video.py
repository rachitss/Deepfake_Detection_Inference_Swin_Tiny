"""Legacy Tkinter uploader retained for reference; FastAPI now handles inputs."""

# from tkinter import Label, Button, filedialog
# from tkinterdnd2 import DND_FILES, TkinterDnD
# import os
#
# def handle_drop(event):
#     global video_path
#
#     raw = event.data
#     raw = raw.strip("{}")
#     video_path = os.path.normpath(raw).replace("\\", "/")
#     print("Selected:", video_path)
#     root.destroy()
#
# def handle_button():
#     global video_path
#
#     path = filedialog.askopenfilename(
#         title="Select Video",
#         filetypes=[("Video files", "*.mp4")]
#     )
#     if path:
#         video_path = os.path.normpath(path).replace("\\", "/")
#         print("Selected:", video_path)
#         root.destroy()
#
# def get_video():
#     global root, video_path
#     root = TkinterDnD.Tk()
#     root.title("Drop a Video File")
#
#     label = Label(root, text="Drag and drop a video file here", width=40, height=10)
#     label.pack(padx=20, pady=20)
#
#     button = Button(root, text="Select Video", command=handle_button)
#     button.pack(padx=10, pady=10)
#
#     label.drop_target_register(DND_FILES)
#     label.dnd_bind("<<Drop>>", handle_drop)
#
#     root.mainloop()
#
#     return video_path


def get_video():
    raise RuntimeError(
        "GUI uploader is deprecated. Use the FastAPI service or pass --video to infer.py."
    )
