import os

if os.path.exists("/home/em/projects/ActiveDialogue/ActiveDialogue"):
    lib_dir = "/home/em/projects/ActiveDialogue"
elif os.path.exists("/home/eric/projects/ActiveDialogue/ActiveDialogue"):
    lib_dir = "/home/eric/projects/ActiveDialogue"
elif os.path.exists("/home/ubuntu/ActiveDialogue"):
    lib_dir = "/home/ubuntu/ActiveDialogue"
elif os.path.exists("/home/ubuntu/projects/ActiveDialogue"):
    lib_dir = "/home/ubuntu/projects/ActiveDialogue"
elif os.path.exists("/book/working/ActiveDialogue"):
    lib_dir = "/book/working/ActiveDialogue"
elif os.path.exists("/torch/ActiveDialogue"):
    lib_dir = "/torch/ActiveDialogue"
elif os.path.exists("/content/ActiveDialogue"):
    lib_dir = "/content/ActiveDialogue"

if os.path.exists("/home/em/projects/ActiveDialogue/ActiveDialogue"):
    mnt_dir = "/home/em/projects/ActiveDialogue/mnt"
elif os.path.exists("/home/eric/projects/ActiveDialogue/ActiveDialogue"):
    mnt_dir = "/home/eric/projects/ActiveDialogue/mnt"
elif os.path.exists("/home/ubuntu/ActiveDialogue"):
    mnt_dir = "/home/ubuntu/ActiveDialogue/mnt"
elif os.path.exists("/home/ubuntu/projects/ActiveDialogue"):
    mnt_dir = "/home/ubuntu/projects/ActiveDialogue/mnt"
elif os.path.exists("/book/working/ActiveDialogue"):
    mnt_dir = "/book/working/ActiveDialogue/mnt"
elif os.path.exists("/torch/ActiveDialogue"):
    mnt_dir = "/torch/ActiveDialogue/mnt"
elif os.path.exists("/content/ActiveDialogue"):
    mnt_dir = "/content/ActiveDialogue/mnt"

comet_ml_key = "WwYuhoUWbTZMhZPAbwIKPVHmC"
