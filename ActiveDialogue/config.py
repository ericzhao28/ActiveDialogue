import os

if os.path.exists("/home/em/projects/ActiveDialogue/ActiveDialogue"):
    lib_dir = "/home/em/projects/ActiveDialogue/ActiveDialogue"
elif os.path.exists("/home/ubuntu/ActiveDialogue"):
    lib_dir = "/home/ubuntu/ActiveDialogue"
elif os.path.exists("/book/working/ActiveDialogue"):
    lib_dir = "/book/working/ActiveDialogue"
elif os.path.exists("/torch/ActiveDialogue"):
    lib_dir = "/torch/ActiveDialogue"

if os.path.exists("/home/em/projects/ActiveDialogue/ActiveDialogue"):
    mnt_dir = "/home/em/projects/ActiveDialogue/mnt"
elif os.path.exists("/home/ubuntu/ActiveDialogue"):
    mnt_dir = "/home/ubuntu/ActiveDialogue/mnt"
elif os.path.exists("/book/working/ActiveDialogue"):
    mnt_dir = "/book/working/ActiveDialogue/mnt"
elif os.path.exists("/torch/ActiveDialogue"):
    mnt_dir = "/torch/ActiveDialogue/mnt"
