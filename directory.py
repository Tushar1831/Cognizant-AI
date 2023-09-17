import os

def pathname(filename):
    project_directory = os.getcwd()
    data_directory = os.path.join(project_directory, filename)
    return data_directory
