from numpy import asarray
from numpy import savez_compressed


def load_images(path):
    img_list = list()
    for i in range(df.shape[0]):
        print(i+1,df['image'][i])
        # load and resize the image
        filename=df['image'][i]
        img = cv2.imread(path+filename)
        # img process
        img = img[ : , : , (2, 1, 0)]
        img = img_pre(img)
        # merge
        img_list.append(img.tolist())
        # 防呆 避免記憶體over loading而中斷, 每200次存一次檔案
        if((i+1)%200==0):
            images=asarray(img_list)
            filename = 'tree.npz'
            savez_compressed(filename, images)
            print(images.shape)
    return asarray(img_list)