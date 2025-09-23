import os

if __name__ == "__main__":

    #datasets_path   = "datasets"
    datasets_path   = "./tracking_datasets/person/"

    #types_name      = os.listdir(datasets_path)# list all subfolders
    types_name      = next(os.walk(datasets_path))[1] # list all subfolders exclude files
    types_name      = sorted(types_name,key=int)# types_name is subfolder_name

    list_file = open('cls_train.txt', 'w')
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):# eliminate README.md in the folder
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            #list_file.write(str(cls_id) + ";" + '%s'%(os.path.join(os.path.relpath(datasets_path), type_name, photo_name)))
            list_file.write(str(cls_id) + ";" + '%s'%(os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()
